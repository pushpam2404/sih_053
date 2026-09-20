"""DRDO ID26053 — runtime semantic segmentation for the foveated 2.5D map.

This is the PRODUCER the project has never had. The consumer side already works and is tested:
`ros2/drdo_grid_map/src/grid_map_node.cpp` parses a `label` PointField off the PointCloud2 and
hands it to `insert_lidar_point`, which stores it as `semantic_label` and uses it for cost and
colour. But nothing ever wrote that field — the Ouster driver does not emit one — so in every run
to date `off_label < 0` and *every point entered the map as UNKNOWN(6)*. The semantic half of the
map has been inert. This module fills the field.

Pipeline per scan:

    (N,4) xyzi  ->  spherical projection to a 64x1024 range image  (nearest return wins)
                ->  SalsaNextLite  ->  (12, H, W) FINE logits
                ->  taxonomy.collapse_probs  ->  ADL-1 label / object class / confidence images
                ->  kNN label transfer  ->  per-point labels for all N points

Two things in that chain are load-bearing and easy to get wrong:

1. The collapse is `softmax -> sum within ADL-1 group -> argmax`, NOT `argmax -> map`. With the
   obstacle group split across five fine classes, a naive argmax lets a single grass class at 0.20
   beat five obstacle classes holding 0.75 between them, and a crowd of people is handed to the
   planner as drivable grass. taxonomy.collapse_probs owns this; see its docstring.

2. The kNN transfer is not a refinement. 10-20% of points lose the z-buffer fight during
   projection and would otherwise inherit the label of whatever occluded them, which paints
   salt-and-pepper obstacles across the grid map. Per-point labels after kNN are also the only
   honest thing to report: per-pixel mIoU runs 2-5 points optimistic.

Strict by default, matching `MinkUNetInference(allow_fallback=False)` and
`TrtInferenceRunner(allow_stub=False)` elsewhere in this package: there is no heuristic fallback
path here at all. If the weights will not load, this raises. The repo has been bitten before by a
z-threshold ladder quietly standing in for a network and benchmarks reporting its latency as
"inference".
"""
import json
import os
from typing import Optional, Tuple

import numpy as np

from ..segmentation.taxonomy import (
    ADL1_UNKNOWN,
    NUM_FINE,
    OBJ_NONE,
    collapse_probs,
)

# Kept in sync with configs/projection.json, which build_range_cache.py writes from the data.
# These are only the fallback for a checkpoint shipped without a meta file.
DEFAULT_H, DEFAULT_W = 64, 1024
DEFAULT_FOV_UP, DEFAULT_FOV_DOWN = 22.5, -22.5


class RangeSegmenter:
    """Per-point semantic segmentation from a trained range-image network.

    Args:
        model_path: a TorchScript/`.pt` checkpoint or an `.onnx` graph exporting FINE **logits**.
        meta_path:  JSON beside the weights carrying projection geometry and normalisation stats.
                    Defaults to `<model_path stem>_meta.json`. The geometry and the norm stats
                    MUST travel with the weights — a model trained at one fov_down and run at
                    another is silently mis-projected, with no error and a large accuracy loss.
        backend:    "auto" | "torch" | "onnx".
        min_conf:   ADL-1 group probability below which a point becomes UNKNOWN(6) rather than a
                    confident wrong class. UNKNOWN carries traversability 0.40, so the planner
                    treats it as uncertain ground instead of either free or lethal.
    """

    def __init__(self, model_path: str, meta_path: Optional[str] = None,
                 backend: str = "auto", device: str = "auto",
                 min_conf: float = 0.0, obj_min_conf: float = 0.5):
        self.model_path = os.path.expanduser(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"no segmentation weights at {self.model_path}. Train with scripts/train_seg.py "
                f"or export from Colab; this class deliberately has no heuristic fallback.")

        self.min_conf = min_conf
        self.obj_min_conf = obj_min_conf
        self._load_meta(meta_path)

        if backend == "auto":
            backend = "onnx" if self.model_path.endswith(".onnx") else "torch"
        self.backend = backend
        if backend == "onnx":
            self._init_onnx()
        elif backend == "torch":
            self._init_torch(device)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ── setup ────────────────────────────────────────────────────────────────────────────────
    def _load_meta(self, meta_path: Optional[str]) -> None:
        if meta_path is None:
            stem = os.path.splitext(self.model_path)[0]
            meta_path = f"{stem}_meta.json"
        self.meta_path = meta_path
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            # Not fatal, but worth shouting about: defaults are almost certainly not what the
            # model trained with, and the failure mode is silent degradation rather than a crash.
            print(f"[RangeSegmenter] WARNING: no meta at {meta_path}; falling back to default "
                  f"projection geometry and identity normalisation. Accuracy will be wrong if "
                  f"the model was trained with different values.")
            meta = {}

        proj = meta.get("projection", {})
        self.H = int(proj.get("H", DEFAULT_H))
        self.W = int(proj.get("W", DEFAULT_W))
        self.fov_up = float(proj.get("fov_up_deg", DEFAULT_FOV_UP))
        self.fov_down = float(proj.get("fov_down_deg", DEFAULT_FOV_DOWN))

        norm = meta.get("norm", {})
        self.mean = np.asarray(norm.get("mean", [0.0] * 5), dtype=np.float32).reshape(5, 1, 1)
        self.std = np.asarray(norm.get("std", [1.0] * 5), dtype=np.float32).reshape(5, 1, 1)
        self.std[self.std == 0] = 1.0

    def _init_torch(self, device: str) -> None:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(obj, dict):
            from ..segmentation.model import build_segmenter
            state = obj.get("model", obj.get("model_state_dict", obj))
            width = int(obj.get("width", 32))
            model = build_segmenter("salsanext_lite", num_classes=NUM_FINE, width=width)
            model.load_state_dict(state, strict=True)   # strict: a silent partial load is how
            model = model.to(self.device)               # this repo shipped random weights before
        else:
            model = obj.to(self.device)
        model.eval()
        self.model = model
        self._torch = torch

    def _init_onnx(self) -> None:
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    # ── inference ────────────────────────────────────────────────────────────────────────────
    def _forward(self, img: np.ndarray) -> np.ndarray:
        """(5,H,W) normalised -> (NUM_FINE,H,W) logits."""
        batch = img[None].astype(np.float32)
        if self.backend == "onnx":
            out = self.session.run(None, {self.input_name: batch})[0]
        else:
            torch = self._torch
            with torch.no_grad():
                t = torch.from_numpy(batch).to(self.device)
                out = self.model(t).float().cpu().numpy()
        logits = out[0]
        if logits.shape[0] != NUM_FINE:
            raise RuntimeError(
                f"model emitted {logits.shape[0]} channels, expected {NUM_FINE} FINE logits. "
                f"The graph must export logits, not an argmax — collapse_probs needs the full "
                f"distribution to sum within ADL-1 groups.")
        return logits

    def segment(self, points: np.ndarray, return_fine: bool = False):
        """(N,4) float32 [x,y,z,intensity] in the SENSOR frame -> per-point labels.

        Returns (adl1 uint8 (N,), obj uint8 (N,), conf float32 (N,)), plus a FINE (N,) uint8
        array when `return_fine` — scripts/eval.py reports both the 12-class fine mIoU (which
        backs the pedestrian/vehicle claim) and the 8-class ADL-1 mIoU (what the grid engine
        actually consumes). Reporting only the flattering one is the failure mode this repo has
        already cleaned up once.

        The sensor frame matters. A spherical projection is only valid about the sensor origin,
        and the network trained that way. Handing this a world-registered cloud (what FAST-LIO
        publishes, and what the grid node consumes) produces a projection smeared by the
        vehicle's roll and pitch plus a silent domain shift. Callers holding world-frame points
        must transform into `os_sensor` first and attach the returned labels back to the original
        world-frame points.
        """
        from ..segmentation import projection as P

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f"expected (N,4) [x,y,z,intensity], got {points.shape}")
        n = len(points)
        if n == 0:
            return (np.zeros(0, np.uint8), np.zeros(0, np.uint8), np.zeros(0, np.float32))

        proj = P.project(points, labels=None, H=self.H, W=self.W,
                         fov_up_deg=self.fov_up, fov_down_deg=self.fov_down)

        img = np.concatenate([
            proj["range"][None],
            proj["xyz"].transpose(2, 0, 1),
            proj["intensity"][None],
        ], axis=0).astype(np.float32)
        img = (img - self.mean) / self.std
        img *= proj["mask"][None]           # empty pixels contribute nothing, not a normalised 0

        logits = self._forward(img)

        # Collapse on the FULL distribution, per pixel, BEFORE any label transfer. Doing this
        # after kNN would vote on already-collapsed labels and lose the grouped-probability
        # safety property that collapse_probs exists to provide.
        flat = logits.reshape(NUM_FINE, -1).T
        adl1_px, obj_px, conf_px = collapse_probs(flat, min_conf=self.min_conf,
                                                  obj_min_conf=self.obj_min_conf)
        adl1_img = adl1_px.reshape(self.H, self.W)
        obj_img = obj_px.reshape(self.H, self.W)
        conf_img = conf_px.reshape(self.H, self.W)

        adl1 = P.knn_postprocess(adl1_img, proj["range"], points, H=self.H, W=self.W,
                                 fov_up_deg=self.fov_up, fov_down_deg=self.fov_down)
        obj = P.knn_postprocess(obj_img, proj["range"], points, H=self.H, W=self.W,
                                fov_up_deg=self.fov_up, fov_down_deg=self.fov_down)
        conf = P.unproject(conf_img, proj["idx"], n, fill=0.0).astype(np.float32)

        # Points the projection never covered keep UNKNOWN rather than a neighbour's guess.
        unseen = conf <= 0.0
        adl1 = adl1.astype(np.uint8)
        obj = obj.astype(np.uint8)
        adl1[unseen & (adl1 == 0)] = ADL1_UNKNOWN
        obj[unseen] = OBJ_NONE

        if not return_fine:
            return adl1, obj, conf

        fine_img = logits.argmax(axis=0).astype(np.uint8)
        fine = P.knn_postprocess(fine_img, proj["range"], points, H=self.H, W=self.W,
                                 fov_up_deg=self.fov_up, fov_down_deg=self.fov_down).astype(np.uint8)
        return adl1, obj, conf, fine

    def labels(self, points: np.ndarray) -> np.ndarray:
        """Convenience: ADL-1 labels only, the field the C++ grid engine consumes."""
        return self.segment(points)[0]
