"""Core utilities for configuration, seeding, parameter counting, and checkpointing."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(
    config_path: str | Path = "configs/default.yaml",
    overrides: list[str] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Loads YAML configuration, applies dot-notation overrides, and handles smoke mode."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override format '{item}'. Expected 'key=value'.")
            key, val_str = item.split("=", 1)
            parsed_val = yaml.safe_load(val_str)
            keys = key.strip().split(".")
            curr = cfg
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            curr[keys[-1]] = parsed_val

    is_smoke = smoke or bool(cfg.get("smoke", {}).get("enabled", False))
    if is_smoke:
        if "smoke" not in cfg:
            cfg["smoke"] = {}
        cfg["smoke"]["enabled"] = True
        if "training" not in cfg:
            cfg["training"] = {}
        cfg["training"]["epochs"] = cfg["smoke"].get("epochs", 2)
        cfg["training"]["batch_size"] = cfg["smoke"].get("batch_size", 20)
        cfg["training"]["seeds"] = cfg["smoke"].get("seeds", [0])

        if "paths" in cfg:
            for key in ["processed_dir", "cache_dir", "runs_dir", "results_dir"]:
                if key in cfg["paths"]:
                    original_val = str(cfg["paths"][key])
                    norm_path = Path(original_val)
                    if norm_path.name != "smoke":
                        cfg["paths"][key] = str(norm_path / "smoke")

    return cfg


def set_all_seeds(seed: int) -> dict[str, Any]:
    """Sets random seeds deterministically across Python, NumPy, PyTorch, and PennyLane."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pennylane_version = "not_installed"
    try:
        import pennylane as qml

        pennylane_version = qml.__version__
    except ImportError:
        pass

    return {
        "seed": seed,
        "torch": torch.__version__,
        "pennylane": pennylane_version,
        "numpy": np.__version__,
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Counts total trainable, total frozen, and head trainable parameters."""
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Module):
        head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    else:
        head_trainable = total_trainable

    return {
        "total_trainable": total_trainable,
        "total_frozen": total_frozen,
        "head_trainable": head_trainable,
    }


def save_checkpoint_atomic(state: dict[str, Any], filepath: str | Path) -> None:
    """Saves checkpoint to a temporary file and atomically replaces destination."""
    dst = Path(filepath)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f"{dst.name}.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, dst)


class EpochTimerCSV:
    """Logs epoch wall-clock timing and losses to a CSV file atomically."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "epoch,train_loss,val_loss,lr,epoch_wall_time_sec,cumulative_wall_time_sec\n"
                )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        wall_time_sec: float,
        cumulative_sec: float,
    ) -> None:
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.6e},"
                f"{wall_time_sec:.4f},{cumulative_sec:.4f}\n"
            )
            f.flush()
