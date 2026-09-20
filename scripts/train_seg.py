#!/usr/bin/env python3
"""DRDO ID26053 — range-image semantic segmentation training (SalsaNext-Lite on RELLIS-3D).

This is the real training entry point. `scripts/train.py` is kept untouched because
tests/python/test_deployment.py and the Phase-4 artefacts reference it, but it cannot be used to
train anything: it hoists 20 scans into a Python list, runs batch_size=1, builds a val split and
never evaluates it, and weights the loss with a hardcoded 8-vector whose provenance is not
recorded anywhere. Every one of those is fixed here rather than in place, so the old numbers
stay reproducible.

What this script is built around, in order of how much it matters:

  1. Lovasz-softmax alongside weighted CE. PERSON is well under 0.1% of RELLIS points. Weighted
     CE makes each person pixel expensive but still optimises per-pixel likelihood, and the
     cheapest way to cut that loss is to get the boundary of a large grass region slightly more
     right. Lovasz is a convex surrogate for the IoU of the class itself, so a class with 400
     pixels in a batch contributes as much to it as a class with 400,000. If the pedestrian
     claim in README is going to be defensible, this term is why.

  2. Both mIoU numbers, every epoch. FINE (12-class) backs the pedestrian/vehicle claim; ADL-1
     (8-class, after taxonomy.collapse) is what the C++ grid engine actually consumes and is the
     only one that shows up in the map. They move differently: merging six obstacle classes into
     one flatters ADL-1 by a lot. Reporting only the ADL-1 number would hide a model that cannot
     tell a person from a fence post; reporting only FINE would hide the number that matters
     downstream. Print both or neither.

  3. Atomic, resume-safe checkpointing. Colab disconnects. torch.save() straight onto last.pt
     that is interrupted mid-flush leaves a truncated file that torch.load() refuses, and the
     previous good checkpoint is already gone — a full night lost to a 200 ms window. Every save
     writes <name>.tmp and then os.replace()s it, which is atomic within a filesystem, and best.pt
     is a separate file so a bad late epoch cannot overwrite the best one.

Usage (Colab):
    !python scripts/train_seg.py --cache /content/rellis_cache --epochs 50 --batch-size 16 \
        --out /content/drive/MyDrive/drdo_runs/salsanext_w32 --resume auto

Usage (this Mac, smoke test only — no GPU, so this proves the code runs, not that it learns):
    .venv/bin/python scripts/train_seg.py --cache <cache> --limit 32 --epochs 1 --batch-size 2
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from drdo_lidar_mapping.segmentation.taxonomy import (
    ADL1_GROUPS, ADL1_NAMES, FINE_NAMES, FINE_TO_ADL1_LUT, IGNORE_INDEX, NUM_ADL1, NUM_FINE,
    class_weights as inv_log_weights,
)

DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


# ── device ───────────────────────────────────────────────────────────────────────────────────
def pick_device(explicit: Optional[str] = None) -> torch.device:
    """cuda -> mps -> cpu.

    mps is in the chain so the whole script (dataloader, loss, metrics, checkpointing) can be
    smoke-tested on the macOS dev host described in CLAUDE.md before a Colab session is spent on
    it. It is not a training target: no CUDA on this machine means no fp16 AMP and no
    channels_last, and a 64x1024 SalsaNext step is seconds, not milliseconds.
    """
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Lovasz-softmax ───────────────────────────────────────────────────────────────────────────
# Implemented here rather than pulled in as a dependency: it is 40 lines, the published
# reference (Berman et al., CVPR 2018) is unmaintained, and a Colab `pip install` of a random
# mirror is exactly the kind of thing that breaks a training session at 3 a.m.
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss, for one class.

    `gt_sorted` is the ground-truth indicator (1 = this class) permuted into descending order of
    prediction error. The returned vector is the per-position increment of the Jaccard loss, so a
    dot product with the sorted errors gives the loss itself.
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-9)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor,
                   ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Multi-class Lovasz-softmax over a (B, C, H, W) logit map and (B, H, W) labels.

    Two decisions that matter:

      - Computed in float32 even under autocast. The sort + cumsum chain is a running difference
        of large partial sums; in fp16 the union cumsum over ~1M pixels overflows the 65504 range
        outright, and the loss comes back as inf on the first step.

      - "present" class averaging: a class absent from the batch is skipped, not counted as a
        perfect 0. Counting it would mean a batch with no PERSON pixels gets a lower loss than
        one that contains people and segments them well, which inverts the incentive this term
        exists to create.
    """
    logits = logits.float()
    C = logits.shape[1]
    # (B,C,H,W) -> (N,C), (B,H,W) -> (N,)
    probs = logits.softmax(dim=1).permute(0, 2, 3, 1).reshape(-1, C)
    flat = labels.reshape(-1)
    valid = flat != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0          # keeps the graph connected; no supervised pixel
    probs = probs[valid]
    flat = flat[valid]

    losses = []
    for c in range(C):
        fg = (flat == c).float()
        if fg.sum() == 0:
            continue                        # class absent from this batch — see docstring
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


class SegLoss(nn.Module):
    """weighted CE(ignore_index=255) + lovasz_weight * Lovasz-softmax.

    CE alone gives a smooth, well-conditioned gradient everywhere and is what actually gets the
    network off the ground in the first epochs; Lovasz alone is flat wherever a class is absent
    and trains badly from scratch. The sum is what SalsaNext ships with and the 1.0 default
    weight is theirs.
    """

    def __init__(self, weights: torch.Tensor, lovasz_weight: float = 1.0,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.register_buffer("weights", weights)
        self.lovasz_weight = float(lovasz_weight)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # A batch in which every pixel is IGNORE_INDEX makes F.cross_entropy divide a zero sum by
        # a zero weight total and return nan. One nan backward is permanent: it writes nan into
        # AdamW's exp_avg / exp_avg_sq, and from then on every parameter update is nan no matter
        # what the next batch contains. The run keeps going, the loss prints as nan, and the
        # checkpoints are worthless. It costs one comparison to make that impossible.
        n_valid = int((labels != self.ignore_index).sum())
        if n_valid == 0:
            zero = logits.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        ce = F.cross_entropy(logits.float(), labels, weight=self.weights.float(),
                             ignore_index=self.ignore_index)
        lv = lovasz_softmax(logits, labels, self.ignore_index) if self.lovasz_weight > 0 \
            else logits.sum() * 0.0
        return ce + self.lovasz_weight * lv, ce.detach(), lv.detach()


# ── metrics ──────────────────────────────────────────────────────────────────────────────────
class ConfusionMatrix:
    """Streaming confusion matrix on the training device.

    Accumulated with a single bincount over (gt * n + pred) rather than a per-class loop: at
    16 x 64 x 1024 pixels a Python loop over 12 classes per batch is measurable next to the
    forward pass, and on cuda it forces a sync per class.
    """

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.mat = torch.zeros(n * n, dtype=torch.int64, device=device)

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        valid = gt < self.n                    # drops IGNORE_INDEX(255) and any stray id
        gt, pred = gt[valid].to(torch.int64), pred[valid].to(torch.int64)
        self.mat += torch.bincount(gt * self.n + pred, minlength=self.n * self.n)

    def iou(self) -> Tuple[np.ndarray, float]:
        m = self.mat.reshape(self.n, self.n).double().cpu().numpy()
        tp = np.diag(m)
        union = m.sum(1) + m.sum(0) - tp
        # A class with no ground-truth AND no prediction is NaN, not 0. ADL-1 classes 6 (UNKNOWN)
        # and 7 (SKY_DUST) can never be emitted by the network — averaging them in as zeros would
        # drag a genuine 0.62 down to 0.47 and make every reported ADL-1 number wrong by a
        # constant the reader cannot see.
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, tp / np.maximum(union, 1e-9), np.nan)
        present = ~np.isnan(iou)
        return iou, float(np.mean(iou[present])) if present.any() else float("nan")


def collapse_logits_to_adl1(logits: torch.Tensor, groups) -> torch.Tensor:
    """(B, NUM_FINE, H, W) fine logits -> (B, H, W) ADL-1 prediction.

    Sums probability WITHIN each ADL-1 group before the argmax — the torch twin of
    taxonomy.collapse_probs(). It must match, because this is the number the report quotes and
    collapse_probs() is what runs on the vehicle. Taking argmax over the fine classes first and
    then mapping through FINE_TO_ADL1_LUT gives a different (and optimistic-looking but
    operationally wrong) answer whenever obstacle evidence is split across several fine classes;
    tests/python/test_segmentation.py pins a case where the two disagree.
    """
    probs = logits.float().softmax(dim=1)
    grouped = torch.stack([probs[:, idx].sum(dim=1) for idx in groups], dim=1)
    return grouped.argmax(dim=1)


def make_adl1_lut(device: torch.device) -> torch.Tensor:
    """256-entry FINE -> ADL-1 lookup as a torch tensor, IGNORE_INDEX preserved."""
    lut = torch.full((256,), IGNORE_INDEX, dtype=torch.int64, device=device)
    lut[:NUM_FINE] = torch.from_numpy(FINE_TO_ADL1_LUT.astype(np.int64)).to(device)
    return lut


# ── class weights ────────────────────────────────────────────────────────────────────────────
def load_class_weights(config_dir: str, device: torch.device,
                       explicit: Optional[str] = None) -> torch.Tensor:
    """Load the FINE class weights written by scripts/build_range_cache.py.

    The file is the provenance. train.py's magic 8-vector (0.1, 1.0876, ... 10.0, 0.0) has none:
    nobody can say which dataset it was counted on or whether 10.0 was measured or guessed, so it
    cannot be regenerated after a taxonomy change. Here the counts come out of the cache builder,
    the weights come out of taxonomy.class_weights() (inverse-log frequency, the RangeNet/SalsaNext
    rule), and both land in configs/class_weights.json next to the run.

    Accepted shapes, most specific first: {"weights": [...]}, {"counts": [...]}, or a bare list.
    """
    path = explicit or os.path.join(config_dir, "class_weights.json")
    if not os.path.isfile(path):
        # Falling back to uniform weights is loud on purpose. Unweighted CE on RELLIS converges
        # to "everything is grass" and reports a plausible-looking pixel accuracy while PERSON
        # IoU is 0.0 — a silent default here would produce exactly the kind of number this
        # project exists not to publish.
        print(f"[WARN] no class weights at {path} — using UNIFORM weights. Rare classes "
              f"(PERSON especially) will not be learned. Run scripts/build_range_cache.py to "
              f"generate it.", flush=True)
        return torch.ones(NUM_FINE, dtype=torch.float32, device=device)

    with open(path) as fh:
        blob = json.load(fh)
    if isinstance(blob, dict) and "weights" in blob:
        w = np.asarray(blob["weights"], dtype=np.float32)
        src = "weights"
    elif isinstance(blob, dict) and "counts" in blob:
        w = inv_log_weights(np.asarray(blob["counts"], dtype=np.float64))
        src = "counts -> taxonomy.class_weights()"
    elif isinstance(blob, (list, tuple)):
        w = np.asarray(blob, dtype=np.float32)
        src = "bare list"
    else:
        raise ValueError(f"{path}: expected a list, or a dict with 'weights' or 'counts'")

    if w.shape != (NUM_FINE,):
        raise ValueError(f"{path}: expected {NUM_FINE} weights (taxonomy.NUM_FINE), got {w.shape}. "
                         "A stale file from a previous taxonomy is a retrain, not a reshape.")
    print(f"[weights] {path} ({src}): " +
          ", ".join(f"{n}={v:.3f}" for n, v in zip(FINE_NAMES, w)), flush=True)
    return torch.from_numpy(np.ascontiguousarray(w)).to(device)


# ── dataset wrapper ──────────────────────────────────────────────────────────────────────────
class AugmentedRange(Dataset):
    """Applies the image-space augmentations from segmentation.augmentation.

    Wraps rather than subclasses RellisRangeDataset so the cache reader stays owned by
    dataset.py, and runs inside the DataLoader worker processes so augmentation cost is hidden
    behind the forward pass instead of serialising with it.
    """

    def __init__(self, base, train: bool = True, lasermix_p: float = 0.5,
                 dropout_p: float = 0.05, y_channel: int = 2):
        self.base = base
        self.train = train
        self.lasermix_p = lasermix_p
        self.dropout_p = dropout_p
        self.y_channel = y_channel

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        from drdo_lidar_mapping.segmentation.augmentation import (
            lasermix_image, random_dropout, random_flip, random_roll)

        img, lab = self.base[i]
        if not self.train:
            return img, lab

        if self.lasermix_p > 0 and np.random.rand() < self.lasermix_p:
            j = np.random.randint(len(self.base))
            if j != i:
                img, lab = lasermix_image((img, lab), self.base[j])
        # Roll before flip: both are exact index permutations, so the order does not change the
        # distribution, but rolling first keeps the flip's y-negation operating on the same
        # tensor layout the dataset produced.
        img, lab = random_roll(img, lab)
        img, lab = random_flip(img, lab, y_channel=self.y_channel)
        img, lab = random_dropout(img, lab, p=self.dropout_p)
        return img, lab


def _worker_init(worker_id: int) -> None:
    """Re-seed numpy per worker.

    DataLoader forks the workers, so without this every worker inherits the SAME numpy RNG state
    and all `num_workers` of them draw the identical roll/flip/dropout sequence. The augmentation
    then has 1/num_workers of the diversity it looks like it has, and nothing anywhere reports it.
    torch's own RNG is already re-seeded per worker; numpy's is not.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 31 - 1)
    np.random.seed(seed)


# ── checkpointing ────────────────────────────────────────────────────────────────────────────
def atomic_save(state: dict, path: str) -> None:
    """Write to <path>.tmp, fsync, then os.replace().

    os.replace() is atomic within a filesystem, so a reader (or the next resume) sees either the
    old complete file or the new complete file and never a half-flushed one. A plain
    torch.save(state, "last.pt") interrupted by a Colab disconnect truncates the only copy.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def rng_state() -> dict:
    """Capture numpy + torch RNG so a resumed run continues the same augmentation stream.

    Without this a resume replays the augmentations of epoch 0 for every epoch after a
    disconnect. It does not crash and it does not show up in the loss, it just quietly removes
    the regularisation from the second half of training.
    """
    st = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: Optional[dict]) -> None:
    if not st:
        return
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as exc:               # different GPU count on the new Colab VM
            print(f"[resume] cuda RNG not restored ({exc}); continuing with a fresh cuda seed")


# ── schedule ─────────────────────────────────────────────────────────────────────────────────
def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int, min_factor: float = 0.01):
    """Linear warmup then cosine decay, stepped PER OPTIMIZER STEP, not per epoch.

    Per-step is what makes a mid-epoch resume land on the right LR: the checkpoint stores the
    scheduler's own step count, so restarting at epoch 12 of 50 resumes the cosine where it was
    instead of at its epoch-12 boundary value.

    The warmup exists because AdamW's second moment is still garbage for the first few hundred
    steps; at lr=2e-3 with batch 16 that has reliably produced a NaN inside the first epoch.
    min_factor keeps the tail LR at 1% rather than exactly 0 — a final epoch at lr=0 trains
    nothing but still costs the 3 minutes.
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(1.0, t)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── train / validate ─────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp_dtype,
              use_amp: bool, log_every: int = 50) -> Dict[str, float]:
    model.train()
    tot = tot_ce = tot_lv = 0.0
    n = 0
    t0 = time.time()
    for it, (img, lab) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(img)
        # Loss OUTSIDE the autocast region and in fp32: see lovasz_softmax's docstring — the
        # cumsum overflows fp16 and returns inf on step 1.
        loss, ce, lv = criterion(logits, lab)

        scaler.scale(loss).backward()
        # Unscale before clipping or the clip threshold is applied to scaled gradients and does
        # nothing at scale=65536.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tot += float(loss.detach())
        tot_ce += float(ce)
        tot_lv += float(lv)
        n += 1
        if log_every and (it % log_every == 0):
            print(f"    it {it:5d}/{len(loader)}  loss={float(loss.detach()):.4f} "
                  f"(ce={float(ce):.4f} lovasz={float(lv):.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  {(time.time()-t0)/max(1,it+1):.2f}s/it",
                  flush=True)
    n = max(1, n)
    return {"loss": tot / n, "ce": tot_ce / n, "lovasz": tot_lv / n, "sec": time.time() - t0}


@torch.no_grad()
def validate(model, loader, device, groups, adl1_lut, amp_dtype, use_amp: bool) -> Dict:
    model.eval()
    cm_fine = ConfusionMatrix(NUM_FINE, device)
    cm_adl1 = ConfusionMatrix(NUM_ADL1, device)
    for img, lab in loader:
        img = img.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(img)
        logits = logits.float()
        cm_fine.update(lab, logits.argmax(dim=1))
        # Ground truth is collapsed through the table; the PREDICTION is collapsed by summing
        # group probabilities, exactly as the vehicle does. Never argmax-then-map.
        cm_adl1.update(adl1_lut[lab.clamp(max=255)], collapse_logits_to_adl1(logits, groups))
    iou_f, miou_f = cm_fine.iou()
    iou_a, miou_a = cm_adl1.iou()
    return {"iou_fine": iou_f, "miou_fine": miou_f, "iou_adl1": iou_a, "miou_adl1": miou_a}


def format_iou(iou: np.ndarray, names) -> str:
    return "  ".join(f"{n}={'  n/a' if np.isnan(v) else f'{v:.3f}'}" for n, v in zip(names, iou))


# ── main ─────────────────────────────────────────────────────────────────────────────────────
def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True,
                   help="range-image cache root (scripts/build_range_cache.py --out)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--width", type=int, default=32, help="SalsaNextLite width (multiple of 4)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lovasz-weight", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--resume", default=None,
                   help="'auto' to pick up <out>/last.pt, or an explicit checkpoint path")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "salsanext_lite"))
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N training scans (smoke test)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", default="val")
    p.add_argument("--class-weights", default=None,
                   help="override configs/class_weights.json")
    p.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    p.add_argument("--device", default=None, help="force cuda/mps/cpu (default: auto)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = pick_device(args.device)
    # AMP is fp16 + GradScaler on cuda only. MPS autocast exists but the fp16 path there has no
    # GradScaler and has produced silent NaNs; cpu bf16 autocast is slower than fp32 for these
    # conv shapes. Both non-cuda devices are smoke-test targets, so the honest setting is off.
    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.float16
    print(f"[device] {device}  amp={use_amp}", flush=True)

    from drdo_lidar_mapping.segmentation.dataset import RellisRangeDataset
    from drdo_lidar_mapping.segmentation.model import SalsaNextLite

    train_base = RellisRangeDataset(args.cache, split="train")
    if args.limit:
        train_base = Subset(train_base, list(range(min(args.limit, len(train_base)))))
    try:
        val_base = RellisRangeDataset(args.cache, split=args.val_split)
        if args.limit:
            val_base = Subset(val_base, list(range(min(args.limit, len(val_base)))))
    except KeyError as exc:
        raise SystemExit(f"[FATAL] {exc}\nA training run without a validation split reports no "
                         "mIoU at all. Rebuild the cache with a val split rather than training "
                         "blind — train.py already made that mistake (it built a val split and "
                         "never evaluated it).")

    train_ds = AugmentedRange(train_base, train=True)
    print(f"[data] train={len(train_ds)} scans  val={len(val_base)} scans  cache={args.cache}",
          flush=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0,
                              worker_init_fn=_worker_init)
    val_loader = DataLoader(val_base, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = SalsaNextLite(in_channels=5, num_classes=NUM_FINE, width=args.width).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] SalsaNextLite width={args.width}  {n_par/1e6:.2f} M parameters", flush=True)

    weights = load_class_weights(args.config_dir, device, args.class_weights)
    criterion = SegLoss(weights, args.lovasz_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_with_warmup(optimizer, args.epochs * steps_per_epoch,
                                   int(args.warmup_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    groups = [torch.from_numpy(g).to(device) for g in ADL1_GROUPS]
    adl1_lut = make_adl1_lut(device)

    last_path = os.path.join(args.out, "last.pt")
    best_path = os.path.join(args.out, "best.pt")
    log_path = os.path.join(args.out, "train_log.json")

    start_epoch, best_miou, history = 0, -1.0, []
    resume_from = last_path if args.resume == "auto" else args.resume
    if resume_from and os.path.isfile(resume_from):
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        best_miou = ck.get("best_miou", -1.0)
        start_epoch = ck["epoch"] + 1
        history = ck.get("history", [])
        restore_rng(ck.get("rng"))
        print(f"[resume] {resume_from}: continuing at epoch {start_epoch}, "
              f"best ADL-1 mIoU so far {best_miou:.4f}", flush=True)
    elif args.resume == "auto":
        print(f"[resume] auto requested, no {last_path} yet — starting from scratch", flush=True)
    elif resume_from:
        raise SystemExit(f"[FATAL] --resume {resume_from} does not exist. Pass 'auto' to start "
                         "fresh when there is nothing to resume from.")

    # Epoch budget: one epoch is len(train_loader) steps. On a Colab T4 a width-32 step at batch
    # 16 is ~0.20 s, so 13,556 scans / 16 = 847 steps is ~2.8 min + ~20 s of validation. That is
    # the reason for 50 short epochs instead of 10 long ones: a disconnect costs one epoch.
    print(f"[plan] {args.epochs} epochs x {steps_per_epoch} steps, "
          f"checkpoint after every epoch -> {last_path}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== epoch {epoch+1}/{args.epochs} ===", flush=True)
        tr = run_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device,
                       amp_dtype, use_amp)
        va = validate(model, val_loader, device, groups, adl1_lut, amp_dtype, use_amp)

        print(f"  train loss={tr['loss']:.4f} (ce={tr['ce']:.4f} lovasz={tr['lovasz']:.4f}) "
              f"{tr['sec']:.0f}s", flush=True)
        print(f"  FINE  mIoU={va['miou_fine']:.4f}   {format_iou(va['iou_fine'], FINE_NAMES)}",
              flush=True)
        print(f"  ADL-1 mIoU={va['miou_adl1']:.4f}   {format_iou(va['iou_adl1'], ADL1_NAMES)}",
              flush=True)

        history.append({"epoch": epoch, "train_loss": tr["loss"], "ce": tr["ce"],
                        "lovasz": tr["lovasz"], "miou_fine": va["miou_fine"],
                        "miou_adl1": va["miou_adl1"], "sec": tr["sec"]})

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "rng": rng_state(),
            "history": history,
            # Everything needed to rebuild the architecture from the file alone. Without it a
            # checkpoint trained at width=16 loads into a width=32 model as 200 shape errors and
            # nobody remembers which run it came from. `width` is duplicated at the top level
            # because inference/range_segmenter.py reads ckpt["width"] there, not inside args.
            "width": args.width,
            "in_channels": 5,
            "num_classes": NUM_FINE,
            "args": vars(args),
            "taxonomy": {"num_fine": NUM_FINE, "fine_names": list(FINE_NAMES)},
        }
        atomic_save(state, last_path)

        # Selection is on ADL-1 mIoU: that is the number the C++ grid engine's output quality
        # actually depends on. FINE mIoU is reported every epoch and stored in history, so the
        # pedestrian claim can still be read off the same run.
        if not math.isnan(va["miou_adl1"]) and va["miou_adl1"] > best_miou:
            best_miou = va["miou_adl1"]
            state["best_miou"] = best_miou
            atomic_save(state, best_path)
            print(f"  [best] ADL-1 mIoU {best_miou:.4f} -> {best_path}", flush=True)

        with open(log_path + ".tmp", "w") as fh:
            json.dump(history, fh, indent=2)
        os.replace(log_path + ".tmp", log_path)

    print(f"\n[done] best ADL-1 mIoU={best_miou:.4f}  ({best_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
