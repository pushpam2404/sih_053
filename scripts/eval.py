#!/usr/bin/env python3
"""Semantic segmentation evaluation — real model inference, per-class IoU and mIoU.

The previous version of this script never ran the model: it copied the ground-truth label with
65% probability and otherwise drew a random class, then reported the resulting "mIoU". This
version runs the checkpoint on every point and also reports two sanity baselines, so a number
that is no better than chance is visible as such.

Usage:
    python3 scripts/eval.py [--checkpoint models/minkunet18_drdo_ep30.pth]
                            [--data data/rellis] [--last-n 20] [--out eval_miou.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from drdo_lidar_mapping.segmentation.dataset import RELLISDataset  # noqa: E402
from drdo_lidar_mapping.segmentation.model import build_minkunet, load_legacy_mlp  # noqa: E402

CLASS_NAMES = ["GROUND", "GRAVEL_DIRT", "GRASS_LOW", "VEGETATION_DENSE",
               "OBSTACLE_HARD", "WATER_MUD", "UNKNOWN", "SKY_NOISE"]
EVAL_CLASSES = range(7)   # SKY_NOISE (7) is discarded before mapping
# Range bins follow the grid engine's foveation bands (5 cm <= 10 m, 10 cm <= 25 m, 50 cm <= 100 m);
# the problem statement asks for classification accuracy "across varying distances".
RANGE_BINS = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, float("inf"))]


def load_model(path: str) -> torch.nn.Module:
    try:
        return build_minkunet(weights_path=path, strict=True).eval()
    except (RuntimeError, KeyError) as sparse_err:
        model = load_legacy_mlp(path)
        print(f"[EVAL] Not a MinkUNet checkpoint ({sparse_err.__class__.__name__}); "
              f"loaded legacy per-point MLP with {sum(p.numel() for p in model.parameters())} parameters")
        return model


def build_predictor(kind: str, checkpoint: str):
    """Return (predict_fn, emits_fine, description).

    predict_fn maps a (N,4) torch tensor to (N,) ADL-1 predictions, and — when the backend can
    produce them — a parallel (N,) array of FINE predictions, else None. Keeping this behind one
    function is the only structural change to this script: everything below it (the confusion
    matrices, the range bins, the random and majority baselines, the not-better-than-random
    warning) is the reason any number out of this repo is believable, and stays untouched.
    """
    if kind == "legacy":
        model = load_model(checkpoint)

        def predict(pts):
            with torch.no_grad():
                return model(pts).argmax(dim=1).numpy(), None

        return predict, False, f"legacy per-point MLP ({checkpoint})"

    if kind == "range":
        from drdo_lidar_mapping.inference.range_segmenter import RangeSegmenter
        seg = RangeSegmenter(checkpoint)

        def predict(pts):
            # Points are already sensor-frame here: RELLIS .bin files are raw sensor scans, which
            # is exactly the frame the spherical projection and the network were trained in.
            adl1, _obj, _conf, fine = seg.segment(pts.numpy(), return_fine=True)
            return adl1.astype(np.int64), fine.astype(np.int64)

        return predict, True, f"range-image SalsaNext ({checkpoint}, {seg.backend} backend)"

    raise ValueError(f"unknown --model {kind!r}")


def iou_table(conf: np.ndarray) -> dict:
    ious = {}
    for c in EVAL_CLASSES:
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        ious[CLASS_NAMES[c]] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def miou_of(conf: np.ndarray) -> float:
    present = conf.sum(axis=1)[:7] > 0
    vals = [v for c, v in zip(EVAL_CLASSES, iou_table(conf).values()) if present[c]]
    return float(np.nanmean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "models/minkunet18_drdo_ep30.pth"))
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data/rellis"))
    ap.add_argument("--last-n", type=int, default=20, help="evaluate the last N scans (sorted order)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--model", choices=("legacy", "range"), default="legacy",
                    help="legacy per-point MLP (default) or the trained range-image network")
    ap.add_argument("--split", default=None,
                    help="official RELLIS split to evaluate (train|val|test); omit to use --data directly")
    args = ap.parse_args()

    predict, emits_fine, model_desc = build_predictor(args.model, args.checkpoint)
    print(f"[EVAL] model: {model_desc}")
    ds_kw = {"split": args.split} if args.split else {}
    ds = RELLISDataset(args.data, **ds_kw)
    indices = range(max(0, len(ds) - args.last_n), len(ds))
    rng = np.random.default_rng(0)

    conf_model = np.zeros((8, 8), dtype=np.int64)
    conf_random = np.zeros((8, 8), dtype=np.int64)
    conf_range = np.zeros((len(RANGE_BINS), 8, 8), dtype=np.int64)
    label_counts = np.zeros(8, dtype=np.int64)
    n_points = 0

    # The 12-class fine confusion is what backs the pedestrian-vs-vehicle claim; the 8-class
    # ADL-1 one above is what the C++ grid engine actually consumes. Both get reported.
    fine_ds, conf_fine = None, None
    if emits_fine:
        from drdo_lidar_mapping.segmentation.dataset import RellisPointDataset
        from drdo_lidar_mapping.segmentation.taxonomy import FINE_NAMES, NUM_FINE
        fine_ds = RellisPointDataset(args.data, label_space="fine", **ds_kw)
        conf_fine = np.zeros((NUM_FINE + 1, NUM_FINE + 1), dtype=np.int64)   # +1 row for IGNORE

    with torch.no_grad():
        for idx in indices:
            pts, lbs = ds[idx]
            gt = lbs.numpy()
            pred, pred_fine = predict(pts)
            if conf_fine is not None and pred_fine is not None:
                gt_fine = fine_ds[idx][1].numpy()
                # IGNORE_INDEX (255) is void/sky: unlabelled returns, excluded from the metric
                # rather than counted as a class the network got wrong.
                keep = gt_fine < NUM_FINE
                np.add.at(conf_fine, (gt_fine[keep], np.minimum(pred_fine[keep], NUM_FINE)), 1)
            rand = rng.integers(0, 7, size=len(gt))
            np.add.at(conf_model, (gt, pred), 1)
            np.add.at(conf_random, (gt, rand), 1)
            rng_m = np.hypot(pts[:, 0].numpy(), pts[:, 1].numpy())
            bin_idx = np.searchsorted([hi for _, hi in RANGE_BINS], rng_m, side="left")
            np.add.at(conf_range, (np.minimum(bin_idx, len(RANGE_BINS) - 1), gt, pred), 1)
            label_counts += np.bincount(gt, minlength=8)
            n_points += len(gt)

    mask = conf_model.sum(axis=1)[:7] > 0
    ious = iou_table(conf_model)
    rand_ious = iou_table(conf_random)
    miou = float(np.nanmean([ious[CLASS_NAMES[c]] for c in EVAL_CLASSES if mask[c]]))
    rand_miou = float(np.nanmean([rand_ious[CLASS_NAMES[c]] for c in EVAL_CLASSES if mask[c]]))
    majority = int(np.argmax(label_counts[:7]))
    acc = float(np.trace(conf_model[:7, :7]) / max(1, conf_model[:7, :].sum()))
    maj_acc = float(label_counts[majority] / max(1, label_counts[:7].sum()))

    print(f"\n--- Per-class IoU on {len(indices)} scans / {n_points} points ({args.data}) ---")
    for c in EVAL_CLASSES:
        print(f"  {c} {CLASS_NAMES[c]:<17} model={ious[CLASS_NAMES[c]]*100:6.2f}%   "
              f"uniform-random={rand_ious[CLASS_NAMES[c]]*100:6.2f}%")
    print(f"\n[EVAL] mIoU (classes 0-6): model={miou*100:.2f}%   uniform-random baseline={rand_miou*100:.2f}%")
    print(f"[EVAL] point accuracy: model={acc*100:.2f}%   majority-class ({CLASS_NAMES[majority]})={maj_acc*100:.2f}%")
    print("\n--- By range from the sensor ---")
    range_rows = []
    for b, (lo, hi) in enumerate(RANGE_BINS):
        cm = conf_range[b]
        n = int(cm[:7, :].sum())
        acc_b = float(np.trace(cm[:7, :7]) / n) if n else float("nan")
        miou_b = miou_of(cm) if n else float("nan")
        range_rows.append({"range_m": [lo, hi], "points": n, "accuracy": acc_b, "mIoU": miou_b})
        print(f"  {lo:5.0f}-{hi:<5.0f} m  points={n:8d}  accuracy={acc_b*100:6.2f}%  mIoU={miou_b*100:6.2f}%")
    fine_rows, fine_miou = None, None
    if conf_fine is not None:
        n_fine = conf_fine[:NUM_FINE, :NUM_FINE]
        fine_ious = {}
        for c in range(NUM_FINE):
            tp = n_fine[c, c]
            denom = n_fine[:, c].sum() + n_fine[c, :].sum() - tp
            fine_ious[FINE_NAMES[c]] = float(tp / denom) if denom > 0 else float("nan")
        present = [c for c in range(NUM_FINE) if n_fine[c, :].sum() > 0]
        fine_miou = float(np.nanmean([fine_ious[FINE_NAMES[c]] for c in present])) if present else float("nan")
        print(f"\n--- Fine {NUM_FINE}-class IoU (what the network predicts; PERSON and VEHICLE "
              f"are the pedestrian/vehicle claim) ---")
        for c in range(NUM_FINE):
            seen = int(n_fine[c, :].sum())
            note = "" if seen else "   (absent in these scans)"
            print(f"  {c:2d} {FINE_NAMES[c]:<17} IoU={fine_ious[FINE_NAMES[c]]*100:6.2f}%  "
                  f"points={seen:8d}{note}")
        print(f"[EVAL] fine mIoU (present classes only): {fine_miou*100:.2f}%")
        print("[EVAL] note: fine mIoU is expected to sit BELOW the ADL-1 mIoU — it scores 12 "
              "classes including rare ones. Published RELLIS-3D LiDAR baselines: SalsaNext 43.07%, "
              "KPConv 19.07%.")
        fine_rows = fine_ious

    if miou <= rand_miou + 0.02:
        print("[EVAL] WARNING: model is not meaningfully better than random guessing.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "model": args.model, "scans": len(indices),
                       "points": n_points,
                       "per_class_iou": ious, "mIoU": miou, "random_baseline_mIoU": rand_miou,
                       "accuracy": acc, "majority_class_accuracy": maj_acc, "by_range": range_rows,
                       "fine_per_class_iou": fine_rows, "fine_mIoU": fine_miou}, f, indent=2)
        print(f"[EVAL] Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
