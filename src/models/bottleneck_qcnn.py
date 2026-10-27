"""Model 2: Frozen ResNet-18 backbone + 8-d bottleneck + QCNN head & 4 matched controls."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn import Model1ResNet18
from src.quantum.circuits import BatchedQCNNHead


class BottleneckQCNNModel(nn.Module):
    """Hybrid Quantum-Classical Bottleneck model with 5 rigorous head variants."""

    def __init__(self, config: dict[str, Any], variant: str = "qcnn", use_sun: bool = True) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.use_sun = use_sun

        m2_cfg = config.get("models", {}).get("model2", {})
        bottleneck_dim = int(m2_cfg.get("bottleneck_dim", 8))
        mlp_control_hidden = int(m2_cfg.get("mlp_control_hidden", 4))
        perm_seed = int(m2_cfg.get("perm_seed", 1337))

        in_dim = 512 + (3 if self.use_sun else 0)
        self.bottleneck = nn.Linear(in_dim, bottleneck_dim)

        if variant == "qcnn":
            self.head = BatchedQCNNHead(
                n_qubits=8,
                encoding="angle",
                use_entanglement=True,
                qubit_permutation=None,
            )
        elif variant == "linear_control":
            self.head = nn.Linear(8, 3)
        elif variant == "mlp_control":
            self.head = nn.Sequential(
                nn.Linear(8, mlp_control_hidden),
                nn.GELU(),
                nn.Linear(mlp_control_hidden, 3),
            )
        elif variant == "no_entanglement_control":
            self.head = BatchedQCNNHead(
                n_qubits=8,
                encoding="angle",
                use_entanglement=False,
                qubit_permutation=None,
            )
        elif variant == "permuted_qubits_control":
            perm = np.random.default_rng(perm_seed).permutation(8).tolist()
            self.head = BatchedQCNNHead(
                n_qubits=8,
                encoding="angle",
                use_entanglement=True,
                qubit_permutation=perm,
            )
        else:
            raise ValueError(f"Unknown variant '{variant}' for Model 2.")

    def encode_bottleneck(
        self, embeddings: torch.Tensor, sun_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Projects backbone embeddings to 8-d angle features in [0, pi]."""
        if self.use_sun and sun_features is not None:
            h = torch.cat([embeddings, sun_features], dim=-1)
        else:
            h = embeddings

        z = self.bottleneck(h)
        x_angle = (torch.tanh(z) + 1.0) * (torch.pi / 2.0)
        return x_angle

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass taking batch dict with 'embedding' or 'image' and 'sun_features'."""
        if "embedding" in batch:
            emb = batch["embedding"]
        elif hasattr(self, "backbone"):
            emb = self.backbone(batch["image"])
        else:
            raise KeyError("Batch must contain precomputed 'embedding' or model must have backbone.")

        sun_feat = batch.get("sun_features") if self.use_sun else None
        x_angle = self.encode_bottleneck(emb, sun_feat)
        return self.head(x_angle)


def precompute_and_cache_backbone_embeddings(
    backbone_model: Model1ResNet18,
    dataset: Dataset,
    cache_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Precomputes 512-d embeddings for all tiles in dataset and saves atomically to disk."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    backbone_model.to(device)
    backbone_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            emb = backbone_model.extract_embeddings(imgs)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))

    stacked = np.concatenate(all_embeddings, axis=0)

    tmp_path = cache_path.with_name(f"tmp_{cache_path.name}")
    np.save(tmp_path, stacked)
    os.replace(tmp_path, cache_path)

    return stacked
