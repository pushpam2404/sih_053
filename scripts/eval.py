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
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    ds = RELLISDataset(args.data)
    indices = range(max(0, len(ds) - args.last_n), len(ds))
    rng = np.random.default_rng(0)

    conf_model = np.zeros((8, 8), dtype=np.int64)
    conf_random = np.zeros((8, 8), dtype=np.int64)
    conf_range = np.zeros((len(RANGE_BINS), 8, 8), dtype=np.int64)
    label_counts = np.zeros(8, dtype=np.int64)
    n_points = 0
    with torch.no_grad():
        for idx in indices:
            pts, lbs = ds[idx]
            gt = lbs.numpy()
            pred = model(pts).argmax(dim=1).numpy()
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
    if miou <= rand_miou + 0.02:
        print("[EVAL] WARNING: model is not meaningfully better than random guessing.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "scans": len(indices), "points": n_points,
                       "per_class_iou": ious, "mIoU": miou, "random_baseline_mIoU": rand_miou,
                       "accuracy": acc, "majority_class_accuracy": maj_acc, "by_range": range_rows}, f, indent=2)
        print(f"[EVAL] Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
