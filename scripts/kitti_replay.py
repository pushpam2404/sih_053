#!/usr/bin/env python3
"""DRDO ID26053 — offline replay: segmentation network -> C++ grid engine.

This is the end-to-end evidence run. It takes RELLIS-3D scans, labels every point, pushes them
through the real `drdo_map` engine (the same `insert_points` / `classify_dirty` / `purge_distant`
calls `grid_map_node.cpp` makes per frame) and reports what the map ended up holding.

    .venv/bin/python scripts/kitti_replay.py --model models/range_seg.pt
    .venv/bin/python scripts/kitti_replay.py --use-gt-labels        # comparison / no-weights path

── The number this script exists to print ───────────────────────────────────────────────────
`NON-UNKNOWN CELLS`. A cell is born with `semantic_label = 6` (UNKNOWN) in `insert_cell()`, and
free-space carving never changes it — only a labelled hit does, via `c.semantic_label = label`.
So a non-UNKNOWN cell is precisely a cell that received a semantic label from this pipeline.

That count has been 0 for the entire history of this repo whenever the labels came from a
network: `grid_map_node` reads a `label` PointField that nothing ever wrote, so the whole map ran
at traversability 0.40. It is the one number that distinguishes "a segmentation model exists in
the tree" from "segmentation reaches the planner", so it is printed on its own line.

── What the previous version of this file did wrong ─────────────────────────────────────────
It constructed a MinkUNetInference and then did:

    if gt_labels is not None:  labeled = column_stack([pts, gt_labels])
    else:                      labeled = model.infer(pts)

Ground truth won whenever a .label file existed, which for `data/rellis` is every single frame.
The model was built, never called, and the script still printed "[PASS] KITTI replay validated."
It was demonstrating the dataset, not the network. Here the network is the default and the
ground truth is opt-in behind --use-gt-labels, labelled as a comparison baseline everywhere it
is reported.

── Honest scope ─────────────────────────────────────────────────────────────────────────────
* No trained checkpoint ships in this repo (models/* are placeholders that score at chance —
  CLAUDE.md). Without --model this script fails loudly rather than substituting a heuristic,
  matching RangeSegmenter's own no-fallback policy.
* `data/rellis` is synthetic: its labels are independent of geometry. A class histogram from it
  measures the plumbing, not accuracy. Accuracy is scripts/eval.py's job.
* RELLIS .bin scans are raw SENSOR-frame and carry no odometry, so the robot is held at the
  origin for the whole replay. That exercises insert/classify/purge/export exactly as the ROS
  node does, but it is not a drive — for the moving-vehicle case see scripts/map_dashboard.py.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lib"))     # where the built drdo_map*.so lives
sys.path.insert(0, REPO_ROOT)

import drdo_map                                                          # noqa: E402
from drdo_lidar_mapping.segmentation.dataset import discover_scans, load_scan   # noqa: E402
from drdo_lidar_mapping.segmentation.taxonomy import (                   # noqa: E402
    ADL1_NAMES, ADL1_UNKNOWN, NUM_ADL1, fine_to_adl1, remap_raw_to_fine,
)


def gt_adl1(raw: np.ndarray) -> np.ndarray:
    """RELLIS raw ids -> ADL-1, through the same RAW->FINE->ADL-1 chain training uses.

    Never a private table: taxonomy.py owns every mapping in this project (the last local copy
    routed mud to vegetation, i.e. told the planner it could drive into a hazard)."""
    return fine_to_adl1(remap_raw_to_fine(raw))


def frame_ground_z(points: np.ndarray, fixed):
    """Ground height for the engine's z filter.

    Z_REL_MIN/MAX (-2 .. +15 m) are measured RELATIVE to ground_z, and these scans are raw
    sensor-frame, so ground sits near -mount_height rather than 0. Leaving ground_z at 0 on a
    real RELLIS scan (sensor ~1.9 m up) pushes the whole ground plane to zr = -1.9 and starts
    clipping it at -2.0. The 2nd percentile of z is a cheap stand-in for what the ROS node reads
    off the base_link TF; --ground-z pins it when you know the mount height.
    """
    return float(fixed) if fixed is not None else float(np.percentile(points[:, 2], 2.0))


def run_replay(args):
    pairs = []
    if args.scan:
        label = args.label or (os.path.splitext(args.scan)[0] + ".label")
        pairs = [(args.scan, label)]
    else:
        pairs = discover_scans(args.data)
    if args.frames > 0:
        pairs = pairs[: args.frames]
    if not pairs:
        raise SystemExit("no scans to replay")

    # Label source. The network is the default; ground truth is a baseline you must ask for.
    segmenter = None
    if args.model:
        from drdo_lidar_mapping.inference.range_segmenter import RangeSegmenter
        segmenter = RangeSegmenter(args.model, meta_path=args.meta, min_conf=args.min_conf)
    if segmenter is None and not args.use_gt_labels:
        raise SystemExit(
            "no label source: pass --model <checkpoint> (the network, the default source) or\n"
            "--use-gt-labels to replay the dataset's own labels as a comparison baseline.\n"
            "There is deliberately no heuristic fallback — see the module docstring.")

    source = "GROUND TRUTH (baseline)" if args.use_gt_labels else f"NETWORK {args.model}"
    print("=" * 80)
    print(f"  DRDO ID26053 replay — {len(pairs)} scans, labels from {source}")
    if args.use_gt_labels and segmenter is not None:
        print("  --model also given: the network runs too, reported as agreement vs ground truth")
    print("=" * 80)

    drdo_map.reset_map()
    n_pts = inserted = 0
    agree = agree_total = 0
    seg_s = ins_s = 0.0
    label_hist = np.zeros(256, dtype=np.int64)

    for i, (cloud, label) in enumerate(pairs):
        points, raw = load_scan(cloud, label)          # raises on a missing label file, by design
        gz = frame_ground_z(points, args.ground_z)

        net = None
        if segmenter is not None:
            t0 = time.perf_counter()
            net = segmenter.labels(points)             # points are already sensor-frame here
            seg_s += time.perf_counter() - t0
        truth = gt_adl1(raw) if (args.use_gt_labels or net is None) else None

        if args.use_gt_labels:
            labels = truth
            if net is not None:
                agree += int((net == truth).sum())
                agree_total += len(truth)
        else:
            labels = net

        # (N,4) float32 [x, y, z, label] — one GIL-released C++ call per scan, the same batch
        # entry point the dashboard uses. frame_ts starts at 1: 0 is the "never updated" value.
        batch = np.empty((len(points), 4), dtype=np.float32)
        batch[:, :3] = points[:, :3]
        batch[:, 3] = labels
        t0 = time.perf_counter()
        kept = drdo_map.insert_points(batch, args.robot_x, args.robot_y, i + 1, gz)
        drdo_map.classify_dirty(gz)
        # purge_distant moves cells, so it may only run BETWEEN frames, never mid-insert.
        if args.purge_every > 0 and (i + 1) % args.purge_every == 0:
            drdo_map.purge_distant(args.robot_x, args.robot_y, args.purge_margin)
        ins_s += time.perf_counter() - t0

        n_pts += len(points)
        inserted += kept
        label_hist += np.bincount(np.asarray(labels, dtype=np.uint8), minlength=256)
        if args.verbose:
            print(f"  [{i + 1:4d}/{len(pairs)}] {os.path.basename(cloud)}: {len(points)} pts, "
                  f"{kept} inserted, ground_z={gz:+.2f}")

    # ── results ──────────────────────────────────────────────────────────────────────────────
    cells = drdo_map.export_cells()
    cell_labels = cells["semantic_label"]
    n_cells = len(cell_labels)
    cell_hist = np.bincount(cell_labels, minlength=NUM_ADL1)
    non_unknown = int(n_cells - cell_hist[ADL1_UNKNOWN])

    print("-" * 80)
    print(f"POINTS       {n_pts} read, {inserted} inserted "
          f"({100.0 * inserted / max(n_pts, 1):.1f}%; the rest are outside Z_REL_MIN/MAX or "
          f"beyond the 100 m outermost band)")
    print(f"CELLS        {n_cells} valid, load {drdo_map.load_factor():.3f}, "
          f"{drdo_map.insert_fail_count()} pool-full drops")
    if segmenter is not None:
        print(f"TIMING       {1000.0 * seg_s / len(pairs):.1f} ms/scan segmentation, "
              f"{1000.0 * ins_s / len(pairs):.1f} ms/scan engine (insert+classify+purge)")
    else:
        print(f"TIMING       {1000.0 * ins_s / len(pairs):.1f} ms/scan engine (insert+classify+purge)")
    if agree_total:
        print(f"AGREEMENT    network vs ground truth: {100.0 * agree / agree_total:.1f}% of points "
              f"(point accuracy, not mIoU — use scripts/eval.py for the real metric)")

    print("\nsemantic_label over resulting map cells (ADL-1, as the C++ engine stored it):")
    for k in range(NUM_ADL1):
        c = int(cell_hist[k])
        pts_k = int(label_hist[k])
        bar = "#" * int(round(40.0 * c / max(n_cells, 1)))
        print(f"  {k} {ADL1_NAMES[k]:<17} {c:>8} cells ({100.0 * c / max(n_cells, 1):5.1f}%) "
              f"{bar:<40} from {pts_k} points")
    stray = int(cell_hist[NUM_ADL1:].sum()) if len(cell_hist) > NUM_ADL1 else 0
    if stray:
        print(f"  !! {stray} cells carry a label outside 0..7 — the engine clamps to 6 on insert, "
              f"so this means the export or the taxonomy drifted")

    print("-" * 80)
    print(f"NON-UNKNOWN CELLS: {non_unknown} / {n_cells} "
          f"({100.0 * non_unknown / max(n_cells, 1):.1f}%)   <- labels that reached the C++ engine")
    print(f"  source: {source}")
    print("  A cell starts at UNKNOWN(6) and carving never changes that, so every cell counted")
    print("  here was written by a per-point label travelling the full producer -> engine path.")
    print("=" * 80)
    return dict(points=n_pts, inserted=inserted, cells=n_cells, non_unknown=non_unknown,
                cell_hist=cell_hist)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "rellis"),
                   help="RELLIS root (<seq>/os1_cloud_node_kitti_bin/*.bin)")
    p.add_argument("--scan", help="replay a single .bin instead of --data")
    p.add_argument("--label", help="label file for --scan (default: the .label beside it)")
    p.add_argument("--frames", type=int, default=50, help="max scans to replay; 0 = all")
    p.add_argument("--model", help="range-segmentation checkpoint (.pt/.onnx). The DEFAULT label "
                                   "source — without it you must pass --use-gt-labels")
    p.add_argument("--meta", help="projection/normalisation JSON (default <model stem>_meta.json)")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="ADL-1 group probability below which a point becomes UNKNOWN(6)")
    p.add_argument("--use-gt-labels", action="store_true",
                   help="replay the dataset's own labels instead of the network. COMPARISON "
                        "BASELINE ONLY — it measures the map path, never the model")
    p.add_argument("--ground-z", type=float, default=None,
                   help="fixed ground height for the engine's relative z filter "
                        "(default: per-frame 2nd percentile of z)")
    p.add_argument("--robot-x", type=float, default=0.0)
    p.add_argument("--robot-y", type=float, default=0.0)
    p.add_argument("--purge-every", type=int, default=10, help="frames between purge_distant; 0 = never")
    p.add_argument("--purge-margin", type=float, default=20.0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


if __name__ == "__main__":
    r = run_replay(build_parser().parse_args())
    assert r["inserted"] > 0, "no point survived the engine's filters — nothing was mapped"
    assert r["cells"] >= r["non_unknown"], "more labelled cells than cells"
    if r["non_unknown"] == 0:
        raise SystemExit("FAIL: not one label reached the C++ engine — the map is entirely "
                         "UNKNOWN(6), i.e. the pre-segmentation behaviour.")
    print("[REPLAY OK] semantic labels reached the C++ grid engine.")
