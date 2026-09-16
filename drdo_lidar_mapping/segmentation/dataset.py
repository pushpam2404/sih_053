import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .remap import remap_rellis

class RELLISDataset(Dataset):
    """RELLIS-3D off-road LiDAR dataset loader with automatic label remapping."""

    def __init__(self, rellis_root: str):
        self.rellis_root = os.path.expanduser(rellis_root)
        self.bin_files = sorted(glob.glob(
            os.path.join(self.rellis_root, "**/*.bin"), recursive=True))
        if len(self.bin_files) == 0:
            raise RuntimeError(f"No .bin point cloud files found in {self.rellis_root}")

    def __len__(self) -> int:
        return len(self.bin_files)

    def __getitem__(self, idx: int):
        bin_path = self.bin_files[idx]
        label_path = bin_path.replace(".bin", ".label")

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        if os.path.exists(label_path):
            raw = np.fromfile(label_path, dtype=np.uint32)
            labels = remap_rellis((raw & 0xFFFF).astype(np.int32))
        else:
            labels = np.full(len(points), 6, dtype=np.uint8)

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
