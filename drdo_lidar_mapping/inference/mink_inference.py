"""
MinkUNet-18 inference wrapper for DRDO ID26053.
Backend: TorchSparse++ (torchsparse >= 2.1)
Input:  (N,4) float32 numpy array [x, y, z, intensity]
Output: (N,4) float32 numpy array [x, y, z, label_id]
"""
import numpy as np
import torch

NUM_CLASSES = 8  # ADL-1 class count (GROUND ... SKY_NOISE)


class MinkUNetInference:
    """
    Wraps MinkUNet-18 (TorchSparse++) for per-point semantic labeling.

    Usage:
        model = MinkUNetInference(checkpoint_path=None)       # DUMMY MODE
        model = MinkUNetInference("weights/minkunet18.pth")   # REAL MODE
        labeled = model.infer(raw_cloud_np)   # (N,4) → (N,4)
    """
    VOXEL_SIZE = 0.05  # metres — locked by ADL-3

    def __init__(self, checkpoint_path: str = None, device: str = "auto", allow_fallback: bool = False):
        """allow_fallback: if a checkpoint was requested but TorchSparse++ is missing, use the
        z/range heuristic instead of raising. Default False — the previous behaviour silently
        substituted heuristics, so benchmarks reported "inference" latency for a numpy threshold."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_fp16   = self.device.type == "cuda"
        self.model      = None
        self.allow_fallback = allow_fallback
        self.dummy_mode = (checkpoint_path is None)
        if not self.dummy_mode:
            self._load_model(checkpoint_path)
        print(f"[MinkUNetInference] device={self.device}  fp16={self.use_fp16}"
              f"  mode={'DUMMY-HEURISTIC' if self.dummy_mode else 'REAL'}")

    def _load_model(self, path: str):
        try:
            from torchsparse.models import MinkUNet18  # type: ignore
        except ImportError as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Checkpoint {path} requested but TorchSparse++ is not installed ({e}). "
                    "Pass allow_fallback=True to explicitly accept heuristic labels.") from e
            print(f"[MinkUNetInference] WARNING: TorchSparse++ not installed ({e}); "
                  "labels are a z/range HEURISTIC, not a neural network.")
            self.dummy_mode = True
            return
        self.model = MinkUNet18(in_channels=4, num_classes=NUM_CLASSES)
        ckpt  = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint {path} does not match MinkUNet18: "
                               f"{len(result.missing_keys)} missing / {len(result.unexpected_keys)} unexpected keys")
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()
        print(f"[MinkUNetInference] Checkpoint loaded: {path}")

    def _voxelize(self, points: np.ndarray):
        from torchsparse.utils.quantize import sparse_quantize  # type: ignore
        import torchsparse  # type: ignore
        coords_int = np.floor(points[:, :3] / self.VOXEL_SIZE).astype(np.int32)
        _, unique_map, inverse_map = sparse_quantize(
            coords_int, return_index=True, return_inverse=True)
        vox_feats  = torch.tensor(points[unique_map], dtype=torch.float32)
        if self.use_fp16: vox_feats = vox_feats.half()
        batch_col  = torch.zeros(len(unique_map), 1, dtype=torch.int32)
        vox_coords = torch.cat([batch_col,
                                 torch.tensor(coords_int[unique_map], dtype=torch.int32)], dim=1)
        return torchsparse.SparseTensor(feats=vox_feats.to(self.device),
                                        coords=vox_coords.to(self.device)), inverse_map

    def infer(self, points: np.ndarray) -> np.ndarray:
        """Returns (N,4) float32 [x, y, z, label_id]."""
        N = len(points)
        if points.shape[1] == 3:
            points = np.concatenate([points, np.ones((N,1), np.float32)], axis=1)
        valid_mask    = (points[:,2] >= -2.0) & (points[:,2] <= 15.0)
        points_valid  = points[valid_mask]
        if self.dummy_mode or self.model is None:
            labels_valid = self._dummy_labels(points_valid)
        else:
            labels_valid = self._real_infer(points_valid)
        output         = np.zeros((N, 4), dtype=np.float32)
        output[:, :3]  = points[:, :3]
        output[:,  3]  = 7.0            # default: SKY_NOISE → will be discarded
        output[valid_mask, :3] = points_valid[:, :3]
        output[valid_mask,  3] = labels_valid.astype(np.float32)
        return output

    def _real_infer(self, points: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            sparse_input, inverse_map = self._voxelize(points)
            logits     = self.model(sparse_input).feats
            vox_labels = logits.argmax(dim=1)
            return vox_labels[torch.tensor(inverse_map, device=self.device)].cpu().numpy().astype(np.uint8)

    def _dummy_labels(self, points: np.ndarray) -> np.ndarray:
        """Deterministic labeling for no-checkpoint pipeline tests."""
        labels = np.zeros(len(points), dtype=np.uint8)
        z      = points[:, 2]
        dist   = np.sqrt(points[:,0]**2 + points[:,1]**2)
        labels[z < 0.10] = 0                               # GROUND
        labels[(z >= 0.10) & (z < 0.30)]           = 1    # GRAVEL_DIRT
        labels[(z >= 0.30) & (dist <  10.0)]        = 4   # OBSTACLE_HARD
        labels[(z >= 0.30) & (dist >= 10.0)]        = 3   # VEGETATION_DENSE
        return labels
