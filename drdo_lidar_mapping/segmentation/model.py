import torch
import torch.nn as nn

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
