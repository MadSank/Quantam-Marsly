"""Model 1: 1-channel adapted ResNet-18 baseline with solar feature fusion."""

import logging
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def make_resnet_input_stem(
    old_conv: nn.Conv2d, input_channels: int, *, pretrained: bool
) -> nn.Conv2d:
    """Return an input stem with correct pretrained or random initialization.

    ResNet18 starts with a three-channel stem. Keep that stem intact when the
    architecture also uses three channels. A pretrained one-channel stem is
    initialized by averaging the pretrained RGB filters; a random one-channel
    stem receives a fresh Conv2d initialization rather than an RGB adaptation.
    """
    input_channels = int(input_channels)
    if input_channels == old_conv.in_channels:
        return old_conv

    if pretrained and not (old_conv.in_channels == 3 and input_channels == 1):
        raise ValueError(
            "Pretrained ResNet18 stem adaptation is supported only for 1 or 3 input channels"
        )

    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
    return new_conv


class Model1ResNet18(nn.Module):
    """ResNet-18 adapted for 1-channel grayscale HiRISE tiles and optional solar features."""

    def __init__(self, config: dict[str, Any], use_sun: bool = True) -> None:
        super().__init__()
        self.config = config
        self.use_sun = use_sun

        m1_cfg = config.get("models", {}).get("model1", {})
        pretrained = bool(m1_cfg.get("pretrained", True))
        self.input_channels = int(m1_cfg.get("input_channels", 1))
        mlp_hidden = int(m1_cfg.get("mlp_hidden", 128))
        dropout = float(m1_cfg.get("dropout", 0.1))

        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception as e:
                logging.warning("Could not load pretrained ResNet18 weights: %s. Using random init.", e)

        try:
            self.backbone = resnet18(weights=weights)
        except Exception as e:
            logging.warning("Offline or error downloading weights (%s). Falling back to weights=None.", e)
            self.backbone = resnet18(weights=None)

        self.backbone.conv1 = make_resnet_input_stem(
            self.backbone.conv1, self.input_channels, pretrained=pretrained
        )

        # Replace classification head with identity to extract 512-d embeddings
        self.backbone.fc = nn.Identity()

        # Fusion and prediction head
        in_dim = 512 + (3 if self.use_sun else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),
        )

    def extract_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts 512-d pooled feature representation from the backbone."""
        if image.ndim == 3:
            image = image.unsqueeze(1)
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected image shape (batch,{self.input_channels},height,width), got {tuple(image.shape)}"
            )
        return self.backbone(image)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass taking batch dict with 'image' and optional 'sun_features'."""
        embeddings = self.extract_embeddings(batch["image"])
        if self.use_sun:
            sun_feat = batch["sun_features"]
            h = torch.cat([embeddings, sun_feat], dim=-1)
        else:
            h = embeddings
        return self.head(h)
