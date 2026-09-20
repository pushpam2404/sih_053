"""DRDO ID26053 — RELLIS-3D loaders.

Two datasets, because evaluation and training want different things:

  RellisPointDataset  raw .bin / .label, one row per LiDAR return. Used by scripts/eval.py:
                      metrics must be reported on the original points, never on a range image,
                      because projection drops the ~8% of returns that collide into an occupied
                      pixel and a mIoU computed on survivors is not the mIoU of the sensor.
  RellisRangeDataset  the compact (64, 1024) cache built by scripts/build_range_cache.py. Used
                      for training: 256 KB/scan instead of ~1 MB, so 13,556 scans are 3.5 GB and
                      fit next to the notebook instead of being streamed through Drive FUSE.

RELLISDataset is the old name, kept so scripts/eval.py and scripts/train.py keep working.

── The bug this file exists to kill ─────────────────────────────────────────────────────────
The previous version did:

    label_path = bin_path.replace(".bin", ".label")
    if os.path.exists(label_path): ...
    else: labels = np.full(len(points), 6, dtype=np.uint8)   # UNKNOWN

Real RELLIS-3D does NOT keep labels beside the clouds. Per sequence:

    <seq>/os1_cloud_node_kitti_bin/000000.bin                    <- points
    <seq>/os1_cloud_node_semantickitti_label_id/000000.label     <- labels

so on real data that `exists()` is False for EVERY frame, the else fires, and every point in
the dataset silently becomes UNKNOWN. Nothing crashes. scripts/eval.py runs to completion and
prints a confident per-class mIoU against an all-6 ground truth — a number that is not wrong by
a few points, it is measuring nothing at all. (Our synthetic data/rellis scans happen to put
the .label beside the .bin, which is exactly why the bug survived: it only appears on the real
dataset it was written for.)

The fix is structural, not a better guess: the `else` is gone and a missing label file raises,
and the split lists below carry the label path as their own column so the path is never
derived from the cloud path at all. Fabricating ground truth is never an acceptable fallback.
"""
import glob
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import IGNORE_INDEX, NUM_FINE, fine_to_adl1, remap_raw_to_fine

# torch is optional at import time: scripts/build_range_cache.py imports the scan-listing
# helpers below and must run on a preprocessing box with no torch installed. The Dataset
# classes raise a clear error if it is actually missing. Same guard style as perception/*'s
# scipy fallbacks (see CLAUDE.md, Conventions).
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAVE_TORCH = True
except ImportError:                                            # pragma: no cover
    torch = None                                               # type: ignore[assignment]
    _HAVE_TORCH = False

    class _TorchDataset:                                       # type: ignore[no-redef]
        pass


# ── Layout of the real dataset ───────────────────────────────────────────────────────────────
# The Ouster OS1-64 is the sensor this project targets (CLAUDE.md: Ouster OS1-64, ROS 2 Humble).
# RELLIS also ships a Velodyne cloud per frame under vel_cloud_node_kitti_bin; a recursive
# "**/*.bin" glob sweeps both, doubling the frame count with clouds that have a different ring
# count, a different FOV and NO matching .label directory. Restrict to the Ouster folder.
CLOUD_DIR = "os1_cloud_node_kitti_bin"
LABEL_DIR = "os1_cloud_node_semantickitti_label_id"
VELODYNE_DIR = "vel_cloud_node_kitti_bin"

SPLIT_FILES = {"train": "pt_train.lst", "val": "pt_val.lst", "test": "pt_test.lst"}

_SPLIT_HELP = (
    "The official RELLIS-3D split lists (pt_train.lst / pt_val.lst / pt_test.lst, ~75 KB total)\n"
    "were not found in {d}.\n"
    "Download them from the RELLIS-3D release page and drop them there, e.g.\n"
    "    <rellis-root>/pt_train.lst\n"
    "There is deliberately NO fallback split. RELLIS is recorded at 10 Hz, so frame N and frame\n"
    "N+1 are the same scene 100 ms apart. A random_split or an index cut puts near-duplicate\n"
    "frames on both sides of the train/val line, and the resulting val mIoU is inflated by\n"
    "memorisation — it is a leak, not a result. Only the official sequence-disjoint lists give\n"
    "a number that means anything."
)


# ── Scan listing ─────────────────────────────────────────────────────────────────────────────
def read_split_list(rellis_root: str, split: str,
                    split_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """Parse pt_<split>.lst into [(cloud_path, label_path), ...], both absolute.

    Each line holds TWO whitespace-separated columns: the cloud path and the label path, both
    relative to the dataset root. Column 2 is used verbatim. That is the structural kill for the
    silent-UNKNOWN bug documented at the top of this file — no string surgery on the cloud path
    can produce a wrong-but-existing label path if the label path is simply read from the file.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    root = os.path.expanduser(rellis_root)
    sdir = os.path.expanduser(split_dir) if split_dir else root
    path = os.path.join(sdir, SPLIT_FILES[split])
    if not os.path.isfile(path):
        raise FileNotFoundError(_SPLIT_HELP.format(d=sdir))

    pairs: List[Tuple[str, str]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 'cloud_path label_path', got {line!r}. "
                    "A one-column list is the wrong file — the pt_*.lst lists carry both paths.")
            cloud, label = cols[0], cols[1]
            pairs.append((cloud if os.path.isabs(cloud) else os.path.join(root, cloud),
                          label if os.path.isabs(label) else os.path.join(root, label)))
    if not pairs:
        raise RuntimeError(f"{path} contains no frames")
    return pairs


def discover_scans(rellis_root: str) -> List[Tuple[str, str]]:
    """Enumerate <root>/**/os1_cloud_node_kitti_bin/*.bin and pair each with its label file.

    NOT A SPLIT. Use read_split_list() for anything that reports a number. This exists for
    smoke tests and for scripts/eval.py's default --data data/rellis, whose 500 synthetic scans
    have no .lst files.

    Label resolution, in order: the sibling os1_cloud_node_semantickitti_label_id/ directory
    (real RELLIS), then a .label beside the .bin (our synthetic scans). If neither exists the
    frame is listed with the *expected* real-layout path and load_scan() raises on it. A frame
    is never listed with a None label, because that is where the old code invented ground truth.
    """
    root = os.path.expanduser(rellis_root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"RELLIS root {root} does not exist")
    bins = sorted(glob.glob(os.path.join(root, "**", CLOUD_DIR, "*.bin"), recursive=True))
    if not bins:
        # Be specific about the likely cause: a flat dump of .bin files is not the RELLIS layout.
        stray = glob.glob(os.path.join(root, "**", "*.bin"), recursive=True)
        hint = (f" ({len(stray)} .bin files exist but none under a {CLOUD_DIR}/ directory; "
                f"{len([s for s in stray if VELODYNE_DIR in s])} of them are Velodyne clouds, "
                "which this project does not use)") if stray else ""
        raise RuntimeError(f"no Ouster clouds found under {root}/*/{CLOUD_DIR}/{hint}")

    pairs: List[Tuple[str, str]] = []
    for b in bins:
        seq_dir = os.path.dirname(os.path.dirname(b))
        stem = os.path.splitext(os.path.basename(b))[0]
        real = os.path.join(seq_dir, LABEL_DIR, stem + ".label")
        beside = os.path.splitext(b)[0] + ".label"
        pairs.append((b, real if os.path.exists(real) else
                      (beside if os.path.exists(beside) else real)))
    return pairs


# ── Scan IO ──────────────────────────────────────────────────────────────────────────────────
def load_scan(cloud_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one frame. Returns (points (N,4) float32 [x,y,z,intensity], raw_ids (N,) uint16).

    Raises on a missing or mismatched label file. There is no default-label branch: see the
    module docstring.
    """
    if not os.path.isfile(cloud_path):
        raise FileNotFoundError(f"point cloud missing: {cloud_path}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(
            f"label file missing: {label_path}\n"
            f"  for cloud: {cloud_path}\n"
            "RELLIS-3D keeps labels in a separate per-sequence directory "
            f"({LABEL_DIR}/), not beside the clouds. Check the sequence is complete — the "
            "loader will NOT substitute UNKNOWN, because that silently turns evaluation into a "
            "measurement of nothing (it is the bug this loader was rewritten to remove).")

    raw_bytes = np.fromfile(cloud_path, dtype=np.float32)
    if raw_bytes.size % 4 != 0:
        raise ValueError(f"{cloud_path}: {raw_bytes.size} float32 values is not a multiple of 4 "
                         "([x,y,z,intensity] per point)")
    points = raw_bytes.reshape(-1, 4)

    # SemanticKITTI-format .label: uint32 per point, low 16 bits = semantic id, high 16 =
    # instance id. & 0xFFFF strips the instance id; without it every labelled instance becomes
    # an out-of-range class.
    raw = np.fromfile(label_path, dtype=np.uint32)
    if raw.shape[0] != points.shape[0]:
        raise ValueError(
            f"label/point count mismatch: {raw.shape[0]} labels vs {points.shape[0]} points\n"
            f"  cloud: {cloud_path}\n  label: {label_path}\n"
            "These are almost always a cloud and a label from different sensors (an Ouster "
            "label against a Velodyne cloud) or a truncated download. Silently zipping them "
            "would mislabel every point after the first difference.")
    return points, (raw & 0xFFFF).astype(np.uint16)


def _require_torch() -> None:
    if not _HAVE_TORCH:
        raise ImportError("torch is required for the Dataset classes; the scan helpers "
                          "(read_split_list / discover_scans / load_scan) work without it")


# ── Per-point dataset (evaluation) ───────────────────────────────────────────────────────────
class RellisPointDataset(_TorchDataset):
    """Raw per-point RELLIS frames.

    __getitem__ -> (points (N,4) float32 tensor, labels (N,) int64 tensor)

    label_space:
      "fine"  the 12 training classes, 255 = ignore (void / sky). Default.
      "adl1"  the 8 classes the C++ engine consumes; ignore collapses to UNKNOWN(6), so every
              value is in 0..7 and can index an 8x8 confusion matrix.
      "raw"   untouched RELLIS ontology ids, for auditing the mapping itself.
    """

    def __init__(self, rellis_root: str, split: Optional[str] = None,
                 split_dir: Optional[str] = None, label_space: str = "fine",
                 limit: Optional[int] = None):
        _require_torch()
        if label_space not in ("fine", "adl1", "raw"):
            raise ValueError(f"label_space must be fine|adl1|raw, got {label_space!r}")
        self.rellis_root = os.path.expanduser(rellis_root)
        self.split = split
        self.label_space = label_space
        self.pairs = (read_split_list(self.rellis_root, split, split_dir) if split
                      else discover_scans(self.rellis_root))
        if limit is not None:
            self.pairs = self.pairs[:limit]

    def __len__(self) -> int:
        return len(self.pairs)

    def frame_paths(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]

    def labels_for(self, raw: np.ndarray) -> np.ndarray:
        if self.label_space == "raw":
            return raw.astype(np.int64)
        fine = remap_raw_to_fine(raw)
        return (fine if self.label_space == "fine" else fine_to_adl1(fine)).astype(np.int64)

    def __getitem__(self, idx: int):
        cloud, label = self.pairs[idx]
        points, raw = load_scan(cloud, label)
        return (torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)),
                torch.from_numpy(self.labels_for(raw)))


class RELLISDataset(RellisPointDataset):
    """Backward-compatible name for scripts/eval.py and scripts/train.py.

    Defaults to label_space="adl1" — NOT just an alias — because both callers were written
    against the old ADL-1 output: eval.py allocates an 8x8 confusion matrix and indexes it with
    the label, and train.py builds an 8-wide class-weight vector. Handing them FINE ids (0..11,
    plus 255 for ignore) would IndexError on the first frame. New code should construct
    RellisPointDataset directly and ask for the label space it actually wants.
    """

    def __init__(self, rellis_root: str, split: Optional[str] = None,
                 split_dir: Optional[str] = None, label_space: str = "adl1",
                 limit: Optional[int] = None):
        super().__init__(rellis_root, split=split, split_dir=split_dir,
                         label_space=label_space, limit=limit)


# ── Range-image dataset (training) ───────────────────────────────────────────────────────────
CACHE_MANIFEST = "manifest.json"


class RellisRangeDataset(_TorchDataset):
    """The compact range-image cache written by scripts/build_range_cache.py.

    __getitem__ -> (features (C,H,W) float32 tensor, labels (H,W) int64 tensor)
                   with C = len(channels), default [range, x, y, z, intensity].

    Labels are FINE ids; empty pixels (no return) carry IGNORE_INDEX(255), so a plain
    CrossEntropyLoss(ignore_index=255) already excludes them and no separate mask is needed.

    x/y/z are NOT stored. They are reconstructed as direction_table * range, which costs one
    (H,W,3) multiply and saves 768 KB/scan — 10 GB over the dataset, i.e. the difference
    between a cache that sits on local disk and one that has to be streamed from Drive. The
    reconstruction places each return at its pixel centre rather than its true azimuth, an
    error of up to half a pixel (~0.3 m at 50 m). That only perturbs the network's positional
    INPUT channels; it can never touch a reported metric, because evaluation runs on the
    original .bin points via RellisPointDataset.
    """

    CHANNELS = ("range", "x", "y", "z", "intensity")

    def __init__(self, cache_root: str, split: str = "train",
                 channels: Sequence[str] = CHANNELS, normalize: bool = True,
                 norm_stats: Optional[dict] = None):
        _require_torch()
        self.cache_root = os.path.expanduser(cache_root)
        self.split = split
        bad = [c for c in channels if c not in self.CHANNELS]
        if bad:
            raise ValueError(f"unknown channels {bad}; available {list(self.CHANNELS)}")
        self.channels = tuple(channels)

        mpath = os.path.join(self.cache_root, CACHE_MANIFEST)
        if not os.path.isfile(mpath):
            raise FileNotFoundError(
                f"no range cache at {self.cache_root} ({CACHE_MANIFEST} missing). Build it with\n"
                "    python3 scripts/build_range_cache.py --rellis-root <root> --out "
                f"{self.cache_root}")
        with open(mpath) as fh:
            self.manifest = json.load(fh)
        if split not in self.manifest.get("splits", {}):
            raise KeyError(f"split {split!r} is not in the cache; built splits: "
                           f"{sorted(self.manifest.get('splits', {}))}")

        proj = self.manifest["projection"]
        self.H, self.W = int(proj["H"]), int(proj["W"])
        self.fov_up, self.fov_down = float(proj["fov_up_deg"]), float(proj["fov_down_deg"])
        # Quantisation step of the stored uint16 range, in metres. Recorded rather than assumed:
        # 1 mm saturates at 65.535 m, which is inside the map's 100 m outer band, so the builder
        # is allowed to widen it and the reader must honour whatever it chose.
        self.range_quant_m = float(proj.get("range_quant_m", 0.001))

        self._shards = self.manifest["splits"][split]["shards"]
        self._index: List[Tuple[int, int]] = []       # (shard, frame within shard)
        for si, sh in enumerate(self._shards):
            self._index.extend((si, f) for f in range(int(sh["count"])))
        self._open: dict = {}

        self.norm = None
        if normalize:
            self.norm = norm_stats if norm_stats is not None else self.manifest.get("norm_stats")
            if self.norm is None:
                raise RuntimeError("normalize=True but the cache carries no norm_stats; rebuild "
                                   "the cache or pass norm_stats=... explicitly")
        self._dirs = None

    def __len__(self) -> int:
        return len(self._index)

    def _dir_table(self) -> np.ndarray:
        if self._dirs is None:
            from .projection import build_direction_table      # lazy: separate module, own owner
            self._dirs = np.asarray(
                build_direction_table(self.H, self.W, self.fov_up, self.fov_down),
                dtype=np.float32)
        return self._dirs

    def _memmap(self, si: int, name: str, dtype) -> np.ndarray:
        key = (si, name)
        if key not in self._open:
            path = os.path.join(self.cache_root, self._shards[si][name])
            self._open[key] = np.memmap(path, dtype=dtype, mode="r").reshape(-1, self.H, self.W)
        return self._open[key]

    def raw_frame(self, idx: int) -> dict:
        """Un-normalised planes, for visualisation and for the cache's own unit tests."""
        si, fi = self._index[idx]
        rng = self._memmap(si, "range", np.uint16)[fi].astype(np.float32) * self.range_quant_m
        inten = self._memmap(si, "intensity", np.uint8)[fi].astype(np.float32) / 255.0
        lab = self._memmap(si, "label", np.uint8)[fi]
        xyz = self._dir_table() * rng[..., None]
        return {"range": rng, "xyz": xyz, "intensity": inten, "label": lab,
                "mask": rng > 0.0}

    def __getitem__(self, idx: int):
        f = self.raw_frame(idx)
        planes = {"range": f["range"], "x": f["xyz"][..., 0], "y": f["xyz"][..., 1],
                  "z": f["xyz"][..., 2], "intensity": f["intensity"]}
        feat = np.stack([planes[c] for c in self.channels], axis=0)
        if self.norm is not None:
            mean = np.asarray([self.norm["mean"][self.CHANNELS.index(c)] for c in self.channels],
                              dtype=np.float32)[:, None, None]
            std = np.asarray([self.norm["std"][self.CHANNELS.index(c)] for c in self.channels],
                             dtype=np.float32)[:, None, None]
            feat = (feat - mean) / np.maximum(std, 1e-6)
        # Empty pixels must not leak a normalised non-zero value into the conv stack.
        feat *= f["mask"][None, :, :]
        lab = f["label"].astype(np.int64)
        lab[(lab >= NUM_FINE) & (lab != IGNORE_INDEX)] = IGNORE_INDEX
        return torch.from_numpy(np.ascontiguousarray(feat)), torch.from_numpy(lab)
