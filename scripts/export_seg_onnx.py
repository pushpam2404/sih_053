#!/usr/bin/env python3
"""DRDO ID26053 — export a trained range-segmentation checkpoint to ONNX + its meta sidecar.

    .venv/bin/python scripts/export_seg_onnx.py --checkpoint runs/best.pt --out models/range_seg.onnx

Produces two files that must always travel together:

  <out>.onnx        the graph, exporting **logits** (B, 12, 64, 1024)
  <out>_meta.json   projection geometry, normalisation stats, and both taxonomy LUTs

── Why logits and not an argmax ─────────────────────────────────────────────────────────────
Baking an argmax into the graph would be smaller and faster and would silently destroy the one
safety property this pipeline has. `taxonomy.collapse_probs` sums probability WITHIN each ADL-1
group before choosing, because the obstacle group is split across five fine classes: five
classes at 0.15 hold 0.75 of the mass but lose a naive per-class argmax to a single grass class
at 0.20, and the planner is told a crowd of people is drivable grass. An exported argmax cannot
be un-done downstream. So the graph ends at the logits, every time.

── Why the sidecar is not optional ──────────────────────────────────────────────────────────
The weights alone are not a model. A net trained at one fov_down and run at another is
mis-projected with no error raised and a large, silent accuracy loss; the same is true of the
per-channel normalisation. RangeSegmenter looks for `<stem>_meta.json` and WARNS loudly when it
is missing, but a warning in a log is not a guarantee. Writing both here, from the same
checkpoint and the same configs/ the training run used, is what keeps deploy and training in
agreement.

── Static H/W, dynamic batch ────────────────────────────────────────────────────────────────
TensorRT strongly prefers static spatial dims, and the projection is fixed at 64x1024 anyway.
Only the batch axis is dynamic, which is what the ROS node (batch 1) and any offline sweep
(batch N) actually need.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from drdo_lidar_mapping.segmentation.model import build_segmenter          # noqa: E402
from drdo_lidar_mapping.segmentation.taxonomy import (                     # noqa: E402
    ADL1_NAMES, FINE_NAMES, FINE_TO_ADL1_LUT, FINE_TO_OBJ_LUT, NUM_FINE, OBJ_NAMES,
)

DEFAULT_H, DEFAULT_W = 64, 1024


def load_checkpoint(path: str):
    """Rebuild the network from a train_seg.py checkpoint.

    `width` is read from the checkpoint rather than assumed: a width mismatch would otherwise
    surface as a state_dict shape error at best, or a silently partial load at worst. This repo
    has already shipped random weights once from a non-strict load, so the load is strict.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(f"{path} is not a train_seg.py checkpoint (no 'model' key)")
    width = int(ckpt.get("width", 32))
    n_cls = int(ckpt.get("num_classes", NUM_FINE))
    in_ch = int(ckpt.get("in_channels", 5))
    if n_cls != NUM_FINE:
        raise SystemExit(f"checkpoint has {n_cls} classes, taxonomy expects {NUM_FINE}")
    model = build_segmenter("salsanext_lite", in_channels=in_ch, num_classes=n_cls, width=width)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt, in_ch, width


def read_configs(config_dir: str, in_ch: int):
    """projection.json + norm_stats.json from the run that built the cache."""
    proj_p = os.path.join(config_dir, "projection.json")
    norm_p = os.path.join(config_dir, "norm_stats.json")
    proj = {"H": DEFAULT_H, "W": DEFAULT_W, "fov_up_deg": 22.5, "fov_down_deg": -22.5}
    if os.path.isfile(proj_p):
        proj.update(json.load(open(proj_p)))
    else:
        print(f"[EXPORT] WARNING: {proj_p} missing — falling back to nominal OS1-64 geometry. "
              f"If the cache was built with measured FOV percentiles this is WRONG.")
    if os.path.isfile(norm_p):
        n = json.load(open(norm_p))
        mean, std = list(n["mean"]), list(n["std"])
    else:
        print(f"[EXPORT] WARNING: {norm_p} missing — writing identity normalisation. Inference "
              f"will not match training.")
        mean, std = [0.0] * in_ch, [1.0] * in_ch
    return proj, mean, std


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best.pt / last.pt from train_seg.py")
    ap.add_argument("--out", required=True, help="output .onnx path (sidecar goes beside it)")
    ap.add_argument("--config-dir", default=os.path.join(REPO_ROOT, "configs"),
                    help="where build_range_cache.py wrote projection.json / norm_stats.json")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true", help="skip the onnxruntime parity check")
    args = ap.parse_args(argv)

    model, ckpt, in_ch, width = load_checkpoint(args.checkpoint)
    proj, mean, std = read_configs(args.config_dir, in_ch)
    H, W = int(proj["H"]), int(proj["W"])
    params = sum(p.numel() for p in model.parameters())
    print(f"[EXPORT] SalsaNextLite width={width} in_ch={in_ch} classes={NUM_FINE} "
          f"({params/1e6:.2f} M params), epoch {ckpt.get('epoch', '?')}, "
          f"best ADL-1 mIoU {ckpt.get('best_miou', float('nan')):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    dummy = torch.randn(1, in_ch, H, W)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["feats"], output_names=["logits"],
        dynamic_axes={"feats": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[EXPORT] wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB), "
          f"feats (B,{in_ch},{H},{W}) -> logits (B,{NUM_FINE},{H},{W})")

    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    meta = {
        "model": "salsanext_lite",
        "width": width,
        "in_channels": in_ch,
        "num_fine_classes": NUM_FINE,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "best_adl1_miou": ckpt.get("best_miou"),
        "epoch": ckpt.get("epoch"),
        "projection": {"H": H, "W": W,
                       "fov_up_deg": float(proj["fov_up_deg"]),
                       "fov_down_deg": float(proj["fov_down_deg"])},
        "norm": {"channels": ["range", "x", "y", "z", "intensity"][:in_ch],
                 "mean": mean, "std": std},
        # Carried for provenance and for any non-Python consumer. taxonomy.py remains the source
        # of truth inside this repo — never read these back in place of importing it, or the two
        # copies will drift exactly the way remap.py's private table once did.
        "fine_names": list(FINE_NAMES),
        "fine_to_adl1": FINE_TO_ADL1_LUT.tolist(),
        "adl1_names": list(ADL1_NAMES),
        "fine_to_obj": FINE_TO_OBJ_LUT.tolist(),
        "obj_names": list(OBJ_NAMES),
        "collapse": "softmax over fine, sum within adl1 group, argmax over groups "
                    "(NEVER argmax-then-map: see taxonomy.collapse_probs)",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EXPORT] wrote {meta_path}")

    if not args.no_verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[EXPORT] onnxruntime not installed — parity check skipped")
            return 0
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = np.random.randn(2, in_ch, H, W).astype(np.float32)   # batch 2: exercises the dynamic axis
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"feats": x})[0]
        if got.shape != ref.shape:
            raise SystemExit(f"[EXPORT] FAIL shape {got.shape} vs torch {ref.shape}")
        diff = float(np.abs(got - ref).max())
        agree = float((got.argmax(1) == ref.argmax(1)).mean())
        print(f"[EXPORT] onnxruntime parity: max|diff|={diff:.2e}, argmax agreement {agree*100:.4f}%")
        if diff > 1e-3:
            raise SystemExit("[EXPORT] FAIL: ORT and torch disagree beyond 1e-3")
        print("[EXPORT] OK — graph exports logits and matches torch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
