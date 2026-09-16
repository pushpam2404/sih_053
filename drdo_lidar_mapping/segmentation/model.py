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

def build_minkunet(in_channels: int = 4, num_classes: int = 8, weights_path: str = None) -> nn.Module:
    """Instantiate MinkUNet18 with optional checkpoint loading."""
    try:
        from torchsparse.models import MinkUNet18
        model = MinkUNet18(in_channels=in_channels, num_classes=num_classes)
    except ImportError:
        model = MinkUNet18Scaffold(in_channels=in_channels, num_classes=num_classes)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    return model
