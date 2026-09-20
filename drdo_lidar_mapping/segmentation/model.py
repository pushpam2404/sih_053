import torch
import torch.nn as nn

# NUM_FINE is the ONLY thing this module is allowed to know about labels. taxonomy.py is
# the single source of truth (CLAUDE.md); a hardcoded 12 here would be a second mapping.
from .taxonomy import NUM_FINE

class MinkUNet18Scaffold(nn.Module):
    """MinkUNet18 Host & Development Scaffold.

    Provides a clean, fully trace-compatible neural network matching the 4-channel
    LiDAR input (x, y, z, intensity) and 8 DRDO output classes. When TorchSparse++
    is present on CUDA / Jetson environments, it wraps the native SparseTensor layers.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        Args:
            x: (N, in_channels) features or LiDAR points.
            coords: optional (N, 4) sparse voxel coordinates.
        Returns:
            logits: (N, num_classes)
        """
        return self.net(x)

def build_minkunet(in_channels: int = 4, num_classes: int = 8, weights_path: str = None,
                   strict: bool = True) -> nn.Module:
    """Instantiate MinkUNet18 with optional checkpoint loading.

    strict=True (default) raises if the checkpoint does not match the architecture. The previous
    strict=False silently loaded zero tensors from models/minkunet18_drdo_ep30.pth (6 missing,
    4 unexpected keys) and returned a randomly initialised network.
    """
    try:
        from torchsparse.models import MinkUNet18
        model = MinkUNet18(in_channels=in_channels, num_classes=num_classes)
    except ImportError:
        model = MinkUNet18Scaffold(in_channels=in_channels, num_classes=num_classes)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        result = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint {weights_path} does not match {type(model).__name__}: "
                f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys "
                f"({loaded} tensors loaded). Use load_legacy_mlp() for the 2-layer host MLP checkpoint.")

    return model


def load_legacy_mlp(weights_path: str, in_channels: int = 4, num_classes: int = 8) -> nn.Module:
    """Load the checkpoint actually produced by scripts/train.py on hosts without TorchSparse++:
    a per-point Linear(4,64)-ReLU-Linear(64,8) MLP over raw (x, y, z, intensity). This is NOT a
    sparse-convolutional MinkUNet — it has no spatial context at all."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    hidden = state["0.weight"].shape[0]
    model = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
    model.load_state_dict(state, strict=True)
    return model.eval()


# ═════════════════════════════════════════════════════════════════════════════════════════════
# SalsaNext-Lite — the network that is actually trained (scripts/train_seg.py)
# ═════════════════════════════════════════════════════════════════════════════════════════════
# Everything above this line is the legacy per-point MLP path and the TorchSparse wrapper. Both
# are kept because scripts/train.py, scripts/eval.py, scripts/export_onnx.py and
# tests/python/test_deployment.py import them by name; nothing below replaces them.
#
# WHY A 2D RANGE-IMAGE CNN AND NOT A SPARSE CONVNET
#
#   - torchsparse / MinkowskiEngine build against a specific CUDA + PyTorch pair. On Colab that
#     pair moves under you between sessions, and on this Apple Silicon host they do not build at
#     all. build_minkunet() above has silently fallen back to a 3-layer MLP on every machine
#     this project has ever run on (see CLAUDE.md, "Deep learning is deliberately parked").
#   - Sparse convolutions have no usable ONNX representation. deploy/ and scripts/export_onnx.py
#     exist to produce a TensorRT engine for the Orin; a network that cannot be exported makes
#     that whole path decorative. SalsaNext-Lite is Conv2d / BatchNorm / LeakyReLU /
#     PixelShuffle only, so it exports at opset 17 and maps onto TensorRT layers 1:1.
#   - Runtime: a 64x1024 image is 65k pixels against ~131k points per OS1-64 scan, and the dense
#     2D form is what GPUs are built for. The 10 Hz budget in README is reachable here and was
#     never reachable with a MinkUNet.
#
# The cost of the trade is projection loss (10-20% of returns lose the z-buffer fight), which is
# paid back by projection.knn_postprocess() at inference time. See projection.py.


class RangeConv2d(nn.Module):
    """Conv2d with CIRCULAR padding on W (azimuth) and ZERO padding on H (elevation).

    This is the single detail most range-image reimplementations get wrong, and it is not
    cosmetic:

      - The W axis of a range image is azimuth. Column 0 and column W-1 are 0.35 deg apart on an
        OS1-64 at W=1024 — physically adjacent. Zero-padding there tells every conv in the stack
        that the scene ends at the seam behind the robot, so a vehicle straddling the seam is
        seen as two half-vehicles against a wall of zeros. With 5 conv stages of receptive-field
        growth the artefact is tens of columns wide, and because the seam sits at a fixed yaw in
        the SENSOR frame it follows the robot around and never averages out over the dataset.
      - The H axis is elevation and genuinely ends: there is no beam above row 0 or below row 63.
        Wrapping it would connect the sky to the ground, which is worse than a zero border.

    Implementation note: the circular half is done with cat(slice, x, slice) rather than
    F.pad(mode="circular") because ONNX's Pad only gained mode "wrap" in opset 19, and the
    TensorRT path this repo targets exports at opset 17. Slice+Concat is supported everywhere
    and costs one extra copy of a (B, C, H, 2*pad) strip.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.pad_w = dilation * (kw - 1) // 2
        pad_h = dilation * (kh - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw), stride=stride,
                              padding=(pad_h, 0), dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_w > 0:
            x = torch.cat((x[..., -self.pad_w:], x, x[..., :self.pad_w]), dim=-1)
        return self.conv(x)


def _cbr(in_ch: int, out_ch: int, kernel_size=3, dilation: int = 1) -> nn.Sequential:
    """conv -> LeakyReLU -> BatchNorm, the SalsaNext ordering.

    BN after the activation rather than before it is what the reference implementation does. It
    is kept identical so published SalsaNext hyper-parameters transfer without a re-tune.
    """
    return nn.Sequential(
        RangeConv2d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class ResContextBlock(nn.Module):
    """Stem block: a 1x1 projection plus a dilated 3x3 pair, joined by a residual add.

    Three of these run at full 64x1024 resolution before any downsampling. That is deliberate:
    a PERSON at 30 m occupies roughly 3x2 pixels, so the only stage that can still see it as
    more than one blurred pixel is the one before the first stride-2.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.LeakyReLU(inplace=True))
        self.branch = nn.Sequential(_cbr(out_channels, out_channels, dilation=1),
                                    _cbr(out_channels, out_channels, dilation=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shortcut(x)
        return s + self.branch(s)


class ResBlock(nn.Module):
    """Encoder block: three dilated branches concatenated, fused 1x1, residual add, then an
    optional dropout + stride-2 average pool.

    Dilations are 1 / 2 / 3 rather than SalsaNext's 1 / 2 / (2x2 kernel, dilation 2). The 2x2
    kernel needs an asymmetric pad that the circular-W wrapper above cannot express symmetrically;
    a 3x3 at dilation 3 covers a wider field for the same parameter count and keeps every conv
    odd-sized, which is also what keeps the ONNX graph free of explicit Pad nodes.

    Pooling is AvgPool2d(2, 2) — an exact halving with no padding, so 64x1024 -> 32x512 -> ...
    -> 4x64 with no off-by-one and no asymmetric border. SalsaNext's 3x3/stride-2/pad-1 pool
    would zero-pad across the azimuth seam and quietly undo the circular convs above it.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2,
                 pool: bool = True, use_dropout: bool = True):
        super().__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.LeakyReLU(inplace=True))
        self.b1 = _cbr(in_channels, out_channels, dilation=1)
        self.b2 = _cbr(out_channels, out_channels, dilation=2)
        self.b3 = _cbr(out_channels, out_channels, dilation=3)
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(out_channels))
        self.pool = nn.AvgPool2d(2, 2) if pool else None
        self.drop = nn.Dropout2d(p=dropout) if (use_dropout and dropout > 0.0) else None

    def forward(self, x: torch.Tensor):
        """Returns (downsampled, skip) when pooling, otherwise a single tensor.

        The skip is taken BEFORE the pool and before the dropout, so the decoder always sees the
        full-resolution activation and the dropout only regularises the path that continues down.
        """
        s = self.shortcut(x)
        a = self.b1(x)
        b = self.b2(a)
        c = self.b3(b)
        out = s + self.fuse(torch.cat((a, b, c), dim=1))
        if self.pool is None:
            return out
        down = self.drop(out) if self.drop is not None else out
        return self.pool(down), out


class UpBlock(nn.Module):
    """Decoder block: PixelShuffle(2) upsample, concat the encoder skip, then the same dilated
    3-branch fuse as the encoder.

    PixelShuffle rather than ConvTranspose2d or interpolate:
      - ConvTranspose2d produces the classic checkerboard, which on a range image shows up as a
        striped traversability grid — visible in the costmap, not just in a loss curve.
      - Upsample(mode="nearest"/"bilinear") exports as ONNX Resize, whose TensorRT support has
        historically been the flaky part of this pipeline. PixelShuffle exports as DepthToSpace,
        which is a plain reshape+transpose on every backend.
    PixelShuffle(2) divides the channel count by 4, so in_channels must be a multiple of 4.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dropout: float = 0.2, use_dropout: bool = True):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError(f"UpBlock in_channels={in_channels} must be divisible by 4 for "
                             "PixelShuffle(2)")
        self.up = nn.PixelShuffle(2)
        cat_ch = in_channels // 4 + skip_channels
        self.b1 = _cbr(cat_ch, out_channels, dilation=1)
        self.b2 = _cbr(out_channels, out_channels, dilation=2)
        self.b3 = _cbr(out_channels, out_channels, dilation=3)
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(out_channels))
        self.drop = nn.Dropout2d(p=dropout) if (use_dropout and dropout > 0.0) else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.drop is not None:
            x = self.drop(x)
        x = torch.cat((x, skip), dim=1)
        a = self.b1(x)
        b = self.b2(a)
        c = self.b3(b)
        return self.fuse(torch.cat((a, b, c), dim=1))


class SalsaNextLite(nn.Module):
    """SalsaNext topology in pure torch.nn.Conv2d, for (B, 5, 64, 1024) range images.

    Input channels are [range, x, y, z, intensity] in that order — the default CHANNELS tuple of
    dataset.RellisRangeDataset. Output is (B, num_classes, 64, 1024) LOGITS at full input
    resolution.

    THE OUTPUT IS LOGITS AND MUST STAY LOGITS. Do not fold an argmax (or even a softmax) into
    the exported graph. taxonomy.collapse_probs() sums probability WITHIN each ADL-1 group
    before taking its argmax, because the fine head splits obstacle evidence across
    person/vehicle/pole/structure/tree/debris — six classes at 0.13 carry 0.78 of the mass but
    lose a naive per-class argmax to one grass class at 0.20, and the table then reports a crowd
    of people as drivable grass at traversability 0.65. That collapse needs the full
    distribution; an argmax-ed export deletes the information it runs on.

    `width` trades capacity for latency: every stage is a multiple of it, so parameters and MACs
    scale ~quadratically. width=32 is the reference SalsaNext size; width=16 is the knob to reach
    for first if the Orin misses 10 Hz, before touching W or H.
    """

    def __init__(self, in_channels: int = 5, num_classes: int = NUM_FINE, width: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        if width % 4 != 0:
            raise ValueError(f"width={width} must be a multiple of 4 (PixelShuffle(2) in the "
                             "last UpBlock divides 2*width by 4)")
        w = int(width)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = w

        # Stem: 3 context blocks at full resolution, no downsampling.
        self.ctx1 = ResContextBlock(in_channels, w)
        self.ctx2 = ResContextBlock(w, w)
        self.ctx3 = ResContextBlock(w, w)

        # Encoder: 4 stride-2 stages -> 64x1024 becomes 4x64 at the bottleneck. The first block
        # runs without dropout (SalsaNext does the same): dropping whole channels at full
        # resolution costs the small-object detail the stem just spent three blocks building.
        self.enc1 = ResBlock(w, 2 * w, dropout, pool=True, use_dropout=False)
        self.enc2 = ResBlock(2 * w, 4 * w, dropout, pool=True)
        self.enc3 = ResBlock(4 * w, 8 * w, dropout, pool=True)
        self.enc4 = ResBlock(8 * w, 8 * w, dropout, pool=True)
        self.bottleneck = ResBlock(8 * w, 8 * w, dropout, pool=False)

        # Decoder: mirror stages, each consuming the pre-pool skip of its encoder twin.
        self.dec1 = UpBlock(8 * w, 8 * w, 4 * w, dropout)          # 4x64   -> 8x128
        self.dec2 = UpBlock(4 * w, 8 * w, 4 * w, dropout)          # 8x128  -> 16x256
        self.dec3 = UpBlock(4 * w, 4 * w, 2 * w, dropout)          # 16x256 -> 32x512
        self.dec4 = UpBlock(2 * w, 2 * w, w, dropout, use_dropout=False)   # 32x512 -> 64x1024

        self.logits = nn.Conv2d(w, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ctx1(x)
        x = self.ctx2(x)
        x = self.ctx3(x)

        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)
        b = self.bottleneck(d4)

        u = self.dec1(b, s4)
        u = self.dec2(u, s3)
        u = self.dec3(u, s2)
        u = self.dec4(u, s1)
        return self.logits(u)          # LOGITS — see the class docstring before changing this.


def build_segmenter(name: str = "salsanext_lite", weights_path: str = None,
                    map_location: str = "cpu", strict: bool = True, **kw) -> nn.Module:
    """Factory for the range-image segmenters.

    `strict=True` is the default for the same reason build_minkunet() carries it: a checkpoint
    that does not match the architecture must raise, not quietly return a randomly initialised
    network that then gets benchmarked as if it were trained.

    Accepts either a bare state_dict or a scripts/train_seg.py checkpoint (which nests the
    weights under "model" alongside the optimizer/scheduler/RNG state).
    """
    builders = {"salsanext_lite": SalsaNextLite}
    if name not in builders:
        raise ValueError(f"unknown segmenter {name!r}; available: {sorted(builders)}")
    model = builders[name](**kw)

    if weights_path:
        ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        try:
            result = model.load_state_dict(state, strict=False)
        except RuntimeError as exc:
            # strict=False still raises on a SHAPE mismatch, which is the common case: the same
            # layer names at a different `width`. Re-raise with the one fact that fixes it.
            ck_w = ckpt.get("args", {}).get("width") if isinstance(ckpt, dict) else None
            raise RuntimeError(
                f"Checkpoint {weights_path} has the right layer names but the wrong shapes. "
                f"It was trained with width={ck_w}; this model was built with "
                f"width={getattr(model, 'width', '?')}.") from exc
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                f"Checkpoint {weights_path} does not match {name}: "
                f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected "
                f"keys. A width= mismatch is the usual cause — the checkpoint records it under "
                f"ckpt['args']['width'].")
    return model
