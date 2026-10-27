"""Mars QCNN Terrain model factory."""

from typing import Any

import torch.nn as nn

from src.models.bottleneck_qcnn import BottleneckQCNNModel
from src.models.cnn import Model1ResNet18
from src.models.pure_qcnn import PureQCNNModel
from src.models.quanv_cnn import QuanvCNNModel


def build_model(
    model_name: str, variant: str, config: dict[str, Any], use_sun: bool = True
) -> nn.Module:
    """Instantiates requested classical or quantum hybrid model by name and variant."""
    name = model_name.lower()
    var = variant.lower()

    if name == "model1":
        return Model1ResNet18(config=config, use_sun=use_sun)
    elif name == "model2":
        return BottleneckQCNNModel(config=config, variant=var, use_sun=use_sun)
    elif name == "model3":
        return QuanvCNNModel(config=config, variant=var, use_sun=use_sun)
    elif name == "model4":
        return PureQCNNModel(config=config, variant=var, use_sun=use_sun)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from 'model1', 'model2', 'model3', 'model4'."
        )


__all__ = [
    "build_model",
    "Model1ResNet18",
    "BottleneckQCNNModel",
    "QuanvCNNModel",
    "PureQCNNModel",
]
