#!/usr/bin/env python3
"""DRDO ID26053 — day-1 gate: is this really RELLIS-3D, and what geometry did it record?

    python3 scripts/verify_rellis.py --rellis-root /content/Rellis-3D

Run this the moment the download finishes and BEFORE building a cache or training. Everything
downstream is parameterised by numbers that are cheap to check now and expensive to discover
after a training run: the azimuth width, the vertical FOV, and whether the label files line up
with the clouds at all.

Deliberately dependency-light — numpy only, no torch — so it runs on a bare Colab VM before the
repo's own package is importable.

The five checks, and why each one exists:

  1. LAYOUT      clouds in <seq>/os1_cloud_node_kitti_bin/, labels in
                 <seq>/os1_cloud_node_semantickitti_label_id/. These are SEPARATE directories.
                 Code that derives the label path from the cloud path by swapping the extension
                 finds nothing, and the failure is silent if it defaults instead of raising.

  2. SPLITS      pt_train.lst / pt_val.lst / pt_test.lst present and two-column. RELLIS is 10 Hz,
                 so frame N and N+1 are the same scene 100 ms apart; any home-made random split
                 puts near-duplicates on both sides of the line and inflates val mIoU by
                 memorisation. The official lists are the only defensible split.

  3. WIDTH       points per scan -> the azimuth resolution the sensor actually ran at. The OS1-64
                 can record 512 / 1024 / 2048 columns. Projecting to the wrong width either
                 throws away returns (too narrow) or leaves the image mostly empty and doubles
                 compute for nothing (too wide).

  4. FOV         the real vertical extent, from pitch percentiles. The nominal OS1-64 figure is
                 +/-22.5 deg, but if the true spread is narrower every row outside it is dead
                 pixels. Feed the printed values to build_range_cache.py --fov-up/--fov-down.

  5. LABELS      raw ids present vs the ontology this project maps. An id we do not map becomes
                 ignore, so an unmapped id that is common in the data is silently discarded
                 training signal.
"""
import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

CLOUD_DIR = "os1_cloud_node_kitti_bin"
LABEL_DIR = "os1_cloud_node_semantickitti_label_id"
SPLITS = ("pt_train.lst", "pt_val.lst", "pt_test.lst")

# The dataset's own ontology (benchmarks/.../config/labels/rellis.yaml). Ids 29/30/32 carry no
# name but appear in the data; the official learning_map folds them into grass/grass/water.
RELLIS_NAMES = {
    0: "void", 1: "dirt", 3: "grass", 4: "tree", 5: "pole", 6: "water", 7: "sky", 8: "vehicle",
    9: "object", 10: "asphalt", 12: "building", 15: "log", 17: "person", 18: "fence", 19: "bush",
    23: "concrete", 27: "barrier", 29: "(grass variant)", 30: "(grass variant)",
    31: "puddle", 32: "(water variant)", 33: "mud", 34: "rubble",
}
COMMON_WIDTHS = (512, 1024, 2048)


def human(n):
    return f"{n:,}"


def check_layout(root):
    print("\n[1/5] LAYOUT")
    seqs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    clouds = sorted(glob.glob(os.path.join(root, "**", CLOUD_DIR, "*.bin"), recursive=True))
    labels = sorted(glob.glob(os.path.join(root, "**", LABEL_DIR, "*.label"), recursive=True))
    stray = glob.glob(os.path.join(root, "**", "*.bin"), recursive=True)
    velo = [p for p in stray if "vel_cloud_node" in p]

    print(f"  sequences found      {len(seqs)}: {', '.join(seqs[:8])}{' ...' if len(seqs) > 8 else ''}")
    print(f"  Ouster clouds        {human(len(clouds))}  ({CLOUD_DIR}/)")
    print(f"  Ouster labels        {human(len(labels))}  ({LABEL_DIR}/)")
    if velo:
        print(f"  Velodyne clouds      {human(len(velo))}  (ignored — different ring count and FOV)")
    if not clouds:
        print(f"  !! no clouds under a {CLOUD_DIR}/ directory.")
        if stray:
            print(f"     {human(len(stray))} .bin files exist elsewhere — the archive may have "
                  f"extracted into an extra nesting level. Point --rellis-root one level deeper.")
        # Return the labels anyway: the annotation archive is 174 MB against the clouds' 14 GB, so
        # this is the normal state while the big download runs, and the label audit is exactly
        # what is worth doing in that window.
        return None, (labels or None)
    if not labels:
        print(f"  !! no labels under {LABEL_DIR}/. The annotations are a SEPARATE 174 MB download "
              f"from the point clouds — extract it over the same tree.")
        return clouds, None
    if len(labels) != len(clouds):
        print(f"  !  count mismatch ({human(len(clouds))} clouds vs {human(len(labels))} labels). "
              f"Not fatal — the split lists name their own pairs — but check the extraction.")
    else:
        print("  OK: clouds and labels pair up, in the two separate directories the loader expects.")
    return clouds, labels


def check_splits(root):
    print("\n[2/5] SPLITS")
    found = 0
    for name in SPLITS:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            print(f"  missing  {name}")
            continue
        found += 1
        with open(p) as fh:
            lines = [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]
        ncol = Counter(len(c) for c in lines)
        print(f"  {name:<14} {human(len(lines)):>8} frames, columns={dict(ncol)}")
        if lines and len(lines[0]) < 2:
            print(f"     !! one column only. The loader reads the LABEL path from column 2; a "
                  f"one-column list is the wrong file.")
        elif lines:
            print(f"     e.g. {lines[0][0]}  |  {lines[0][1]}")
    if found < 3:
        print("  !! Download the 75 KB split archive. There is no fallback split on purpose — see "
              "the 10 Hz leakage note in dataset.py.")
    return found == 3


def check_width(clouds, sample=200):
    print("\n[3/5] AZIMUTH WIDTH (points per scan)")
    counts = Counter()
    for p in clouds[:sample]:
        counts[os.path.getsize(p) // 16] += 1          # 4 float32 per point
    for n, k in counts.most_common(5):
        rows = [f"64x{w}={human(64 * w)}" for w in COMMON_WIDTHS if abs(64 * w - n) <= 64 * w * 0.02]
        print(f"  {human(n):>10} points   in {k:>4} of {min(sample, len(clouds))} scans"
              f"{'   <- matches ' + ', '.join(rows) if rows else ''}")
    n = counts.most_common(1)[0][0]
    best = min(COMMON_WIDTHS, key=lambda w: abs(64 * w - n))
    dense = n / (64.0 * best)
    print(f"\n  => use W={best}  (64 x {best} = {human(64 * best)} pixels; "
          f"scans are {dense * 100:.0f}% of that)")
    if dense < 0.5:
        print("  !  scans fill less than half the grid — returns are sparse (sky, absorption) or "
             "the width guess is too wide. Compare against the occupancy the cache reports.")
    return best


def check_fov(clouds, sample=100):
    print("\n[4/5] VERTICAL FOV (pitch percentiles over real returns)")
    pitches = []
    for p in clouds[:sample]:
        a = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        xyz = a[:, :3]
        r = np.linalg.norm(xyz, axis=1)
        ok = r > 1e-3
        pitches.append(np.degrees(np.arcsin(xyz[ok, 2] / r[ok])))
    pit = np.concatenate(pitches)
    lo, hi = np.percentile(pit, [0.1, 99.9])
    amin, amax = pit.min(), pit.max()
    print(f"  {human(len(pit))} returns over {min(sample, len(clouds))} scans")
    print(f"  absolute   {amin:+.2f} .. {amax:+.2f} deg")
    print(f"  0.1/99.9%  {lo:+.2f} .. {hi:+.2f} deg   <- use these, they reject stragglers")
    print(f"\n  => build_range_cache.py --fov-up {hi:.2f} --fov-down {lo:.2f}")
    if hi - lo < 30:
        print(f"  !  spread is {hi - lo:.1f} deg, well under the nominal 45. Using +/-22.5 would "
              f"leave {(1 - (hi - lo) / 45) * 100:.0f}% of image rows empty.")
    return float(hi), float(lo)


def check_labels(labels, sample=100):
    print("\n[5/5] LABEL IDS")
    if not labels:
        print("  skipped — no label files")
        return
    hist = Counter()
    for p in labels[:sample]:
        raw = np.fromfile(p, dtype=np.uint32) & 0xFFFF
        for k, v in zip(*np.unique(raw, return_counts=True)):
            hist[int(k)] += int(v)
    total = sum(hist.values())
    print(f"  {human(total)} labelled points over {min(sample, len(labels))} scans")
    unmapped = []
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
        name = RELLIS_NAMES.get(k)
        pct = 100.0 * v / total
        if name is None:
            unmapped.append((k, pct))
            print(f"  id {k:>3}  {pct:6.2f}%   ** NOT IN THE ONTOLOGY — will become ignore **")
        else:
            print(f"  id {k:>3}  {pct:6.2f}%   {name}")
    void = 100.0 * hist.get(0, 0) / total
    if void > 30:
        print(f"\n  note: void is {void:.1f}% of points. That is normal for RELLIS and is why the "
              f"loss uses ignore_index and the weights are inverse-log frequency.")
    if unmapped:
        worst = max(p for _, p in unmapped)
        if worst > 0.5:
            print(f"\n  !! an unmapped id covers {worst:.2f}% of points — add it to "
                  f"taxonomy.RAW_TO_FINE rather than silently dropping that signal.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rellis-root", required=True)
    ap.add_argument("--sample", type=int, default=100, help="scans to sample for FOV/labels")
    args = ap.parse_args(argv)

    root = os.path.expanduser(args.rellis_root)
    if not os.path.isdir(root):
        raise SystemExit(f"{root} is not a directory")
    print("=" * 78)
    print(f"  RELLIS-3D verification — {root}")
    print("=" * 78)

    clouds, labels = check_layout(root)
    ok_split = check_splits(root)

    # The point clouds are a 14 GB download; the labels and splits are 174 MB and 75 KB. They
    # almost always land first, and the label audit below is the one that can force a taxonomy
    # change before training. So run whatever the data present allows instead of refusing.
    if not clouds:
        check_labels(labels, args.sample)
        print("\n" + "=" * 78)
        print("  PARTIAL — no point clouds yet (the 14 GB archive).")
        print("  The label and split checks above are still meaningful: if an unmapped id covers")
        print("  a real share of points, fix taxonomy.RAW_TO_FINE NOW, while the clouds download.")
        print("  Re-run this once the clouds are extracted to measure W and the FOV.")
        print("=" * 78)
        return 1

    W = check_width(clouds, max(args.sample, 200))
    hi, lo = check_fov(clouds, args.sample)
    check_labels(labels, args.sample)

    print("\n" + "=" * 78)
    print("  NEXT — build the cache with the measured geometry:")
    print(f"    python3 scripts/build_range_cache.py --rellis-root {root} \\")
    print(f"        --out cache --configs-dir configs \\")
    print(f"        --width {W} --fov-up {hi:.2f} --fov-down {lo:.2f} --splits train,val")
    if not ok_split:
        print("\n  (blocked: get the 75 KB split lists first)")
    print("=" * 78)
    return 0 if (clouds and labels and ok_split) else 1


if __name__ == "__main__":
    sys.exit(main())
