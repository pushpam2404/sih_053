#!/usr/bin/env python3
"""DRDO ID26053 — turn raw RELLIS-3D .bin/.label into a compact range-image cache.

WHY THIS SCRIPT EXISTS
Training reads every scan once per epoch. The raw dataset is ~14 GB of 4-float points, and on
Colab it lives behind a Google Drive FUSE mount where per-file latency, not bandwidth, sets the
rate. A 30-epoch run that reads 14 GB through FUSE every epoch leaves the GPU idle waiting on
IO; the schedule does not close. So the projection is done ONCE, offline, into fixed-size planes:

    range      (64, 1024) uint16   2 mm steps       128 KB   (reaches 131 m; see --range-quant-m)
    intensity  (64, 1024) uint8                      64 KB
    label      (64, 1024) uint8    FINE ids          64 KB
                                                 ---------
                                                   256 KB / scan

13,556 scans -> ~3.5 GB, a 4x cut, small enough to copy to the notebook's local disk where the
loader is no longer the bottleneck.

WHAT IS DELIBERATELY NOT STORED
x, y and z. They are reconstructed at train time as build_direction_table(...) * range, which
costs one multiply and saves 768 KB/scan (~10 GB). The reconstruction puts each return at its
pixel centre instead of its true azimuth — up to half a pixel, ~0.3 m at 50 m. That error lands
only in the network's positional INPUT channels and can never reach a reported metric, because
scripts/eval.py deprojects onto the original .bin points via RellisPointDataset.

SIDE PRODUCTS (into configs/)
  class_weights.json  per-FINE-class point counts from the TRAIN split plus the inverse-log
                      weights from taxonomy.class_weights(). This replaces the hardcoded
                      8-vector in scripts/train.py ([0.1, 1.0876, ...]) whose provenance nobody
                      can reconstruct — a weight vector that does not come from the data it is
                      used on is a guess wearing four decimal places.
  norm_stats.json     per-channel mean/std for [range, x, y, z, intensity], TRAIN split only.
                      Computed, not copied from SemanticKITTI: RELLIS is off-road, the range
                      histogram and the Ouster intensity scale are both different, and a wrong
                      mean shifts every input by a constant the first conv has to unlearn.
  projection.json     the H, W and FOV actually used, so training, export and inference cannot
                      disagree about the geometry the weights were trained on.

Usage:
    python3 scripts/build_range_cache.py --rellis-root ~/rellis --out ~/rellis_range
    python3 scripts/build_range_cache.py --rellis-root ~/rellis --out /tmp/c --limit 8   # smoke
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from drdo_lidar_mapping.segmentation.dataset import (  # noqa: E402
    CACHE_MANIFEST, load_scan, read_split_list)
from drdo_lidar_mapping.segmentation import taxonomy  # noqa: E402

MANIFEST_VERSION = 2
CHANNELS = ("range", "x", "y", "z", "intensity")
PLANES = {"range": np.uint16, "intensity": np.uint8, "label": np.uint8}


def _import_projection():
    """Imported lazily and by hand so the error names the missing module, not a traceback."""
    try:
        from drdo_lidar_mapping.segmentation.projection import build_direction_table, project
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "drdo_lidar_mapping/segmentation/projection.py is required by this script "
            f"({exc}). It provides project() and build_direction_table().") from exc
    return project, build_direction_table


# ── Shard layout ─────────────────────────────────────────────────────────────────────────────
# One flat binary per (split, sequence, plane) rather than one .npz per scan. 13,556 scans would
# be 40k small files; on a Drive FUSE mount the per-file open dominates and listing the
# directory alone takes minutes. 3 files per sequence memmap cleanly and copy as one stream.
def _sequence_of(cloud_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(cloud_path))) or "seq"


def _group_by_sequence(pairs: List[Tuple[str, str]]) -> "List[Tuple[str, List[Tuple[str, str]]]]":
    order: List[str] = []
    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for p in pairs:
        seq = _sequence_of(p[0])
        if seq not in buckets:
            buckets[seq] = []
            order.append(seq)
        buckets[seq].append(p)
    return [(s, buckets[s]) for s in order]


def _frame_bytes(H: int, W: int) -> Dict[str, int]:
    return {n: H * W * np.dtype(dt).itemsize for n, dt in PLANES.items()}


def _truncate_to(path: str, nbytes: int) -> None:
    """A run killed mid-write leaves a partial frame. Truncating to the manifest's committed
    count is what makes --limit runs and Colab disconnects resumable instead of corrupt."""
    if os.path.exists(path):
        if os.path.getsize(path) != nbytes:
            with open(path, "r+b") as fh:
                fh.truncate(nbytes)
    else:
        open(path, "wb").close()


# ── Accumulators ─────────────────────────────────────────────────────────────────────────────
class Accum:
    """Running counts/sums, carried in the manifest so a resumed run does not double-count."""

    def __init__(self, d: dict = None):
        d = d or {}
        self.counts = np.asarray(d.get("class_counts", [0] * taxonomy.NUM_FINE), dtype=np.float64)
        self.ignored = float(d.get("ignored_points", 0.0))
        self.n = float(d.get("valid_pixels", 0.0))
        self.s = np.asarray(d.get("sum", [0.0] * len(CHANNELS)), dtype=np.float64)
        self.ss = np.asarray(d.get("sumsq", [0.0] * len(CHANNELS)), dtype=np.float64)
        self.saturated = float(d.get("saturated_range_pixels", 0.0))

    def to_dict(self) -> dict:
        return {"class_counts": self.counts.tolist(), "ignored_points": self.ignored,
                "valid_pixels": self.n, "sum": self.s.tolist(), "sumsq": self.ss.tolist(),
                "saturated_range_pixels": self.saturated}

    def mean_std(self) -> Tuple[List[float], List[float]]:
        n = max(self.n, 1.0)
        mean = self.s / n
        var = np.maximum(self.ss / n - mean * mean, 0.0)
        return mean.tolist(), np.sqrt(var).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rellis-root", required=True, help="dataset root holding 00000/ .. 00004/")
    ap.add_argument("--split-dir", default=None,
                    help="directory holding pt_train.lst / pt_val.lst / pt_test.lst "
                         "(default: --rellis-root)")
    ap.add_argument("--out", required=True, help="cache directory to write")
    ap.add_argument("--splits", default="train,val", help="comma-separated (default train,val)")
    ap.add_argument("--limit", type=int, default=None, help="first N scans per split (smoke test)")
    ap.add_argument("--configs-dir", default=os.path.join(REPO_ROOT, "configs"))
    ap.add_argument("--height", type=int, default=None, help="rings (default: projection's 64)")
    ap.add_argument("--width", type=int, default=None, help="columns (default: projection's 1024)")
    ap.add_argument("--fov-up", type=float, default=22.5, help="OS1-64 vertical FOV, degrees")
    ap.add_argument("--fov-down", type=float, default=-22.5)
    ap.add_argument("--range-quant-m", type=float, default=0.002,
                    help="metres per stored uint16 step. Default 0.002 (2 mm) saturates at "
                         "131.07 m, which covers the map's whole 100 m outer band. 1 mm would "
                         "saturate at 65.535 m — INSIDE that band, and inside the 50-100 m bin "
                         "scripts/eval.py reports — so it would silently cap the far field the "
                         "problem statement specifically asks about. 2 mm is far finer than a "
                         "range-image cell is meaningful to anyway (a 50 cm map cell at 100 m "
                         "spans ~0.35 deg of azimuth), so the extra precision buys nothing and "
                         "the reach is free. The script counts and reports saturated pixels "
                         "either way, so the loss is never silent.")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    project, build_direction_table = _import_projection()
    import drdo_lidar_mapping.segmentation.projection as proj_mod
    H = args.height or int(getattr(proj_mod, "DEFAULT_H", 64))
    W = args.width or int(getattr(proj_mod, "DEFAULT_W", 1024))

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    projection = {"H": H, "W": W, "fov_up_deg": args.fov_up, "fov_down_deg": args.fov_down,
                  "range_quant_m": args.range_quant_m}
    fbytes = _frame_bytes(H, W)
    max_range_m = 65535 * args.range_quant_m

    # ── resume ───────────────────────────────────────────────────────────────────────────────
    mpath = os.path.join(out, CACHE_MANIFEST)
    manifest = {"version": MANIFEST_VERSION, "projection": projection, "splits": {}}
    if os.path.isfile(mpath):
        with open(mpath) as fh:
            prev = json.load(fh)
        if prev.get("version") == MANIFEST_VERSION and prev.get("projection") == projection:
            manifest = prev
            print(f"[CACHE] resuming from {mpath}")
        else:
            print(f"[CACHE] {mpath} was built with different geometry/version — rebuilding")

    # Direction table for the x/y/z statistics. Deliberately the SAME reconstruction the loader
    # uses (dirs * range), not project()'s exact xyz: norm_stats must describe the tensor the
    # network actually sees, otherwise the normalisation is centred on data that never arrives.
    dirs = np.asarray(build_direction_table(H, W, args.fov_up, args.fov_down), dtype=np.float32)

    t_start = time.time()
    for split in splits:
        pairs = read_split_list(args.rellis_root, split, args.split_dir)
        if args.limit:
            pairs = pairs[:args.limit]
        groups = _group_by_sequence(pairs)
        total = sum(len(g) for _, g in groups)
        print(f"\n[CACHE] split={split}  {total} scans in {len(groups)} sequence(s)")

        prev_split = manifest["splits"].get(split, {})
        prev_shards = {s["seq"]: s for s in prev_split.get("shards", [])}
        accum = Accum(prev_split.get("accum"))
        shards: List[dict] = []
        done_total = 0

        for seq, frames in groups:
            rel = {n: os.path.join(split, seq, f"{n}.bin") for n in PLANES}
            os.makedirs(os.path.join(out, split, seq), exist_ok=True)
            prior = prev_shards.get(seq, {})
            start = int(prior.get("count", 0))
            # Never trust a count past the frames this run was asked for.
            start = min(start, len(frames))
            for n in PLANES:
                _truncate_to(os.path.join(out, rel[n]), start * fbytes[n])
            if start:
                print(f"[CACHE]   {seq}: {start}/{len(frames)} already cached, resuming")

            handles = {n: open(os.path.join(out, rel[n]), "ab") for n in PLANES}
            try:
                for i in range(start, len(frames)):
                    cloud, label = frames[i]
                    points, raw = load_scan(cloud, label)
                    fine = taxonomy.remap_raw_to_fine(raw)
                    out_p = project(points, labels=fine, H=H, W=W,
                                    fov_up_deg=args.fov_up, fov_down_deg=args.fov_down)
                    mask = np.asarray(out_p["mask"], dtype=bool)
                    rng_m = np.asarray(out_p["range"], dtype=np.float32) * mask

                    sat = int(np.count_nonzero(rng_m > max_range_m))
                    accum.saturated += sat
                    q = np.clip(np.rint(rng_m / args.range_quant_m), 0, 65535).astype(np.uint16)

                    inten = np.asarray(out_p["intensity"], dtype=np.float32)
                    # RELLIS/Ouster intensity arrives 0..1 in the .bin; anything above 1.5
                    # means the producer used raw counts, so scale by the frame max instead of
                    # clipping 99% of the image to 255.
                    scale = 255.0 if inten.max() <= 1.5 else 255.0 / max(inten.max(), 1e-6)
                    inten_u8 = np.clip(np.rint(inten * scale), 0, 255).astype(np.uint8) * mask

                    lab = np.asarray(out_p["label"], dtype=np.uint8).copy()
                    # Empty pixels are ignore, not class 0. A (64,1024) grid on a ~13k-point
                    # scan is ~80% empty; calling those ASPHALT would make the loss majority-
                    # vote for pavement.
                    lab[~mask] = taxonomy.IGNORE_INDEX

                    handles["range"].write(q.tobytes())
                    handles["intensity"].write(inten_u8.tobytes())
                    handles["label"].write(lab.tobytes())

                    if split == "train":
                        valid = mask
                        r = rng_m[valid].astype(np.float64)
                        xyz = (dirs * rng_m[..., None])[valid].astype(np.float64)
                        iv = (inten_u8[valid].astype(np.float64) / 255.0)
                        cols = [r, xyz[:, 0], xyz[:, 1], xyz[:, 2], iv]
                        accum.n += r.size
                        for c, v in enumerate(cols):
                            accum.s[c] += v.sum()
                            accum.ss[c] += np.square(v).sum()
                        lv = lab[valid]
                        accum.counts += np.bincount(lv[lv < taxonomy.NUM_FINE],
                                                    minlength=taxonomy.NUM_FINE)[:taxonomy.NUM_FINE]
                        accum.ignored += int(np.count_nonzero(lv >= taxonomy.NUM_FINE))

                    done_total += 1
                    n_done = done_total + sum(int(s["count"]) for s in shards)
                    if args.progress_every and (i + 1 - start) % args.progress_every == 0:
                        el = time.time() - t_start
                        rate = n_done / max(el, 1e-6)
                        print(f"[CACHE]   {seq} {i + 1}/{len(frames)}  "
                              f"{rate:.1f} scan/s  eta {(total - n_done) / max(rate, 1e-6):.0f}s")
            finally:
                for h in handles.values():
                    h.close()

            shard = {"seq": seq, "count": len(frames),
                     "frames": [os.path.splitext(os.path.basename(c))[0] for c, _ in frames]}
            shard.update(rel)
            shards.append(shard)
            manifest["splits"][split] = {"shards": shards, "accum": accum.to_dict(),
                                         "total": sum(int(s["count"]) for s in shards)}
            with open(mpath, "w") as fh:            # commit after every sequence, not at the end
                json.dump(manifest, fh, indent=2)

        if split == "train":
            mean, std = accum.mean_std()
            manifest["norm_stats"] = {"channels": list(CHANNELS), "mean": mean, "std": std,
                                      "valid_pixels": accum.n, "split": "train"}
        if accum.saturated:
            print(f"[CACHE] WARNING: {int(accum.saturated)} pixels exceeded {max_range_m:.3f} m "
                  f"and were clipped by the uint16 range quantisation. Rebuild with a larger "
                  f"--range-quant-m (e.g. 0.004 -> 262 m) if returns beyond the map's 100 m "
                  f"outer band matter.")

    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # ── configs/ side products ───────────────────────────────────────────────────────────────
    cdir = os.path.expanduser(args.configs_dir)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "projection.json"), "w") as fh:
        json.dump(projection, fh, indent=2)

    if "train" in splits:
        acc = Accum(manifest["splits"]["train"]["accum"])
        counts = acc.counts.astype(np.int64)
        weights = taxonomy.class_weights(counts)
        with open(os.path.join(cdir, "class_weights.json"), "w") as fh:
            json.dump({"source": "scripts/build_range_cache.py, train split only",
                       "note": "Counts are projected range-image pixels, not raw points: the "
                               "network is trained on pixels, so the loss weights must come "
                               "from the pixel distribution.",
                       "fine_names": list(taxonomy.FINE_NAMES),
                       "counts": counts.tolist(),
                       "ignored_pixels": int(acc.ignored),
                       "weights": [float(w) for w in weights],
                       "formula": "w_c = 1 / log(1.02 + f_c)  (taxonomy.class_weights)"},
                      fh, indent=2)
        with open(os.path.join(cdir, "norm_stats.json"), "w") as fh:
            json.dump(manifest["norm_stats"], fh, indent=2)
        print(f"[CACHE] wrote {cdir}/class_weights.json, norm_stats.json, projection.json")
    else:
        print(f"[CACHE] wrote {cdir}/projection.json only "
              "(class_weights/norm_stats need the train split)")

    built = {s: manifest['splits'][s]['total'] for s in splits if s in manifest['splits']}
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(out) for f in fs)
    print(f"[CACHE] done in {time.time() - t_start:.1f}s — {built}, {size / 1e6:.1f} MB at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
