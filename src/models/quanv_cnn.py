"""Model 3: Quanvolutional CNN and matched classical filter / plain CNN controls."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantum.circuits import FixedQuanvolutionFilter


def precompute_or_load_quanv_cache(
    images_np: np.ndarray,
    cache_path: Path,
    n_layers: int = 2,
    seed: int = 2026,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Precomputes or loads cached 4-channel 16x16 quanvolutional feature maps."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quanv_filter = FixedQuanvolutionFilter(n_layers=n_layers, seed=seed)

    if images_np.ndim == 3:
        images_tensor = torch.from_numpy(images_np).unsqueeze(1).float()
    else:
        images_tensor = torch.from_numpy(images_np).float()

    n_samples = images_tensor.shape[0]

    # Downsample to 32x32
    downsampled_32 = F.adaptive_avg_pool2d(images_tensor, (32, 32))

    # Extract non-overlapping 2x2 stride-2 patches -> (N, 4, 256)
    patches = F.unfold(downsampled_32, kernel_size=2, stride=2)
    patches = patches.permute(0, 2, 1).contiguous().view(-1, 4)

    # Evaluate quanvolution filter
    out_patches = quanv_filter(patches, chunk_size=chunk_size)

    # Reshape back to (N, 4, 16, 16)
    out_maps = (
        out_patches.view(n_samples, 256, 4)
        .permute(0, 2, 1)
        .contiguous()
        .view(n_samples, 4, 16, 16)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    tmp_path = cache_path.with_name(f"tmp_{cache_path.name}")
    np.save(tmp_path, out_maps)
    os.replace(tmp_path, cache_path)

    return out_maps


class QuanvCNNModel(nn.Module):
    """Quanvolutional CNN with matched classical random filter and plain CNN controls."""

    def __init__(self, config: dict[str, Any], variant: str = "quanv", use_sun: bool = True) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.use_sun = use_sun

        m3_cfg = config.get("models", {}).get("model3", {})
        quanv_seed = int(m3_cfg.get("quanv_seed", 2026))

        if variant == "classical_filter_control":
            gen = torch.Generator().manual_seed(quanv_seed)
            self.classical_filter = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=True)
            nn.init.xavier_uniform_(self.classical_filter.weight, generator=gen)
            nn.init.zeros_(self.classical_filter.bias)
            self.classical_filter.requires_grad_(False)
        elif variant == "quanv":
            self.quanv_filter = FixedQuanvolutionFilter(
                n_layers=int(m3_cfg.get("n_random_layers", 2)), seed=quanv_seed
            )

        # CNN trunk
        if variant in {"quanv", "classical_filter_control"}:
            self.block1 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
            )
        elif variant == "plain_cnn_control":
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
            )
        else:
            raise ValueError(f"Unknown variant '{variant}' for Model 3.")

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8x8 -> 4x4
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 64
        )

        in_head = 64 + (3 if self.use_sun else 0)
        self.head = nn.Sequential(
            nn.Linear(in_head, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Processes input feature maps or downsamples images through filters and CNN."""
        if self.variant == "quanv":
            if "quanv_map" in batch:
                x = batch["quanv_map"]
            else:
                img = batch["image"]
                x_32 = F.adaptive_avg_pool2d(img, (32, 32))
                b_size = x_32.shape[0]
                patches = F.unfold(x_32, kernel_size=2, stride=2).permute(0, 2, 1).contiguous().view(-1, 4)
                out_p = self.quanv_filter(patches)
                x = out_p.view(b_size, 256, 4).permute(0, 2, 1).contiguous().view(b_size, 4, 16, 16)
        elif self.variant == "classical_filter_control":
            img = batch["image"]
            x_32 = F.adaptive_avg_pool2d(img, (32, 32))
            x = torch.tanh(self.classical_filter(x_32))
        elif self.variant == "plain_cnn_control":
            img = batch["image"]
            x = F.adaptive_avg_pool2d(img, (32, 32))
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        pooled = torch.flatten(h3, start_dim=1)

        if self.use_sun:
            h_fused = torch.cat([pooled, batch["sun_features"]], dim=-1)
        else:
            h_fused = pooled

        return self.head(h_fused)
