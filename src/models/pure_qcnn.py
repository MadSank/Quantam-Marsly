"""Model 4: Pure Amplitude-Encoded QCNN and matched tiny CNN / MLP controls."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantum.circuits import BatchedQCNNHead


def downsample_and_l2_normalize(
    images: torch.Tensor, size: int = 16, eps: float = 1e-8
) -> torch.Tensor:
    """Downsamples images to size x size and L2-normalizes amplitude vectors across channels."""
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[1] > 1:
        images = images[:, :1, :, :]

    x = F.adaptive_avg_pool2d(images, (size, size)).flatten(start_dim=1)
    norms = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)

    uniform_amp = 1.0 / math.sqrt(size * size)
    normalized = torch.where(norms < eps, torch.full_like(x, uniform_amp), x / torch.clamp(norms, min=eps))
    return normalized


class PureQCNNModel(nn.Module):
    """Amplitude-encoded 8/10-qubit Pure QCNN and matched classical controls."""

    def __init__(self, config: dict[str, Any], variant: str = "pure_qcnn", use_sun: bool = True) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.use_sun = use_sun

        m4_cfg = config.get("models", {}).get("model4", {})
        mlp_control_hidden = int(m4_cfg.get("mlp_control_hidden", 32))

        if variant == "pure_qcnn":
            n_qubits = 10 if self.use_sun else 8
            self.head = BatchedQCNNHead(
                n_qubits=n_qubits,
                encoding="amplitude",
                use_entanglement=True,
            )
        elif variant == "tiny_cnn_control":
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # -> 16
            )
            in_head = 16 + (3 if self.use_sun else 0)
            self.head = nn.Linear(in_head, 3)
        elif variant == "mlp_control":
            in_features = 256 + (3 if self.use_sun else 0)
            self.head = nn.Sequential(
                nn.Linear(in_features, mlp_control_hidden),
                nn.GELU(),
                nn.Linear(mlp_control_hidden, 3),
            )
        else:
            raise ValueError(f"Unknown variant '{variant}' for Model 4.")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Processes normalized 16x16 amplitude vectors through pure QCNN or matched controls."""
        images = batch["image"]
        x_norm = downsample_and_l2_normalize(images, size=16)

        if self.variant == "pure_qcnn":
            sun_rad = batch.get("sun_angles_rad") if self.use_sun else None
            return self.head(x_norm, sun_angles_rad=sun_rad)

        elif self.variant == "tiny_cnn_control":
            x_2d = x_norm.view(-1, 1, 16, 16)
            feat = torch.flatten(self.features(x_2d), start_dim=1)
            if self.use_sun:
                feat = torch.cat([feat, batch["sun_features"]], dim=-1)
            return self.head(feat)

        elif self.variant == "mlp_control":
            if self.use_sun:
                x_in = torch.cat([x_norm, batch["sun_features"]], dim=-1)
            else:
                x_in = x_norm
            return self.head(x_in)

        else:
            raise ValueError(f"Unknown variant: {self.variant}")
