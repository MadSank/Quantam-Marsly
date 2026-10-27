from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.controlled_evaluation import (
    binary_metrics,
    fit_training_image_bounds,
    select_validation_threshold,
)
from src.data.dataset import MarsTerrainDataset
from src.train import capture_rng_state, restore_rng_state
import torch
import yaml


def test_controlled_normalization_touches_only_image_and_masks_invalid_cells(tmp_path: Path) -> None:
    shard = np.zeros((1, 3, 2, 2), dtype=np.float16)
    shard[0, 0] = [[0.0, 0.25], [0.5, 0.75]]
    shard[0, 1] = [[1.0, 0.5], [1.0, 1.0]]
    shard[0, 2] = [[1.0, 0.0], [0.5, 1.0]]
    path = tmp_path / "raw_if.npy"
    np.save(path, shard)
    frame = pd.DataFrame([{"tile_id": "t0", "image_id": "site_01", "shard_path": str(path), "shard_idx": 0}])
    cfg = {
        "preprocessing": {"target_cols": ["mean_slope_5m", "roughness_m", "hazard_fraction"]},
        "features": {"use_sun_features": False},
        "augmentation": {"enable_photometric": False},
        "input_normalization": {"if_low": 0.25, "if_high": 0.75},
    }
    sample = MarsTerrainDataset(frame, config=cfg, is_train=False)[0]["image"].numpy()
    assert np.allclose(sample[0], [[0.0, 0.0], [0.5, 1.0]])
    assert np.allclose(sample[1], shard[0, 1])
    assert np.allclose(sample[2], shard[0, 2])


def test_fold_normalizer_uses_only_rows_passed_as_training(tmp_path: Path) -> None:
    shard = np.zeros((2, 3, 2, 2), dtype=np.float16)
    shard[0, 0] = [[0.1, 0.2], [0.3, 0.4]]
    shard[0, 2] = 1.0
    shard[1, 0] = 0.9
    shard[1, 2] = 1.0
    path = tmp_path / "train_and_heldout.npy"
    np.save(path, shard)
    rows = pd.DataFrame([
        {"site_id": "train_site", "shard_path": str(path), "shard_idx": 0},
        {"site_id": "heldout_site", "shard_path": str(path), "shard_idx": 1},
    ])
    bounds = fit_training_image_bounds(rows, rows.iloc[[0]])
    assert bounds["fit_sites"] == ["train_site"]
    assert bounds["fit_valid_pixel_count"] == 4
    assert bounds["if_low"] < 0.2 < bounds["if_high"] < 0.9
    assert bounds["clip"] is False


def test_nonclipping_normalization_preserves_out_of_range_values_and_masks(tmp_path: Path) -> None:
    shard = np.zeros((1, 3, 2, 3), dtype=np.float32)
    shard[0, 0] = [[-0.5, 0.25, 0.5], [0.75, 1.0, 1.25]]
    shard[0, 1] = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]
    shard[0, 2] = [[1, 1, 0], [1, 1, 1]]
    path = tmp_path / "nonclipping.npy"
    np.save(path, shard)
    frame = pd.DataFrame([{"tile_id": "tile", "image_id": "site", "shard_path": str(path), "shard_idx": 0}])
    cfg = {
        "features": {"use_sun_features": False},
        "augmentation": {"enable_photometric": False},
        "input_normalization": {"if_low": 0.25, "if_high": 0.75, "clip": False},
    }
    sample = MarsTerrainDataset(frame, config=cfg)[0]["image"].numpy()
    assert np.allclose(sample[0], [[-1.5, 0.0, 0.0], [1.0, 1.5, 2.0]])
    assert np.allclose(sample[1], shard[0, 1])
    assert np.allclose(sample[2], shard[0, 2])


def test_nonclipping_normalization_serializes_and_reloads(tmp_path: Path) -> None:
    bounds = {"if_low": 0.1, "if_high": 0.2, "clip": False, "fit_sites": ["train"]}
    config_path = tmp_path / "fold.yaml"
    config_path.write_text(yaml.safe_dump({"input_normalization": bounds}), encoding="utf-8")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert loaded["input_normalization"] == bounds

    shard = np.zeros((1, 3, 1, 2), dtype=np.float32)
    shard[0, 0, 0] = [0.0, 0.3]
    shard[0, 2, 0] = 1.0
    path = tmp_path / "serialized.npy"
    np.save(path, shard)
    frame = pd.DataFrame([{"tile_id": "tile", "image_id": "site", "shard_path": str(path), "shard_idx": 0}])
    sample = MarsTerrainDataset(frame, config=loaded)[0]["image"].numpy()
    assert np.allclose(sample[0, 0], [-1.0, 2.0])


def test_safety_metrics_use_expected_denominators_and_nan_for_zero() -> None:
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (1, 1, 1, 1)
    assert m["false_positive_rate"] == 0.5
    assert m["false_discovery_rate"] == 0.5
    assert m["safe_recall"] == 0.5
    assert m["specificity"] == 0.5
    m0 = binary_metrics(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(m0["safe_precision"])
    assert np.isnan(m0["safe_recall"])
    assert np.isnan(m0["false_discovery_rate"])


def test_validation_threshold_meets_fpr_target_without_test_data() -> None:
    val = pd.DataFrame({
        "pred_mean_slope_5m": [1.0, 2.0, 7.0, 10.0],
        "pred_hazard_fraction": [0.0, 0.01, 0.02, 0.0],
        "true_safe": [1, 1, 0, 0],
    })
    threshold, m = select_validation_threshold(val, target_fpr=0.0)
    assert m["false_positive_rate"] == 0.0
    assert threshold <= max(7.0 / 8.0, 0.02 / 0.05) or m["safe_recall"] == 0.0


def test_rng_state_roundtrip_preserves_next_torch_random_value() -> None:
    random.seed(29)
    np.random.seed(31)
    torch.manual_seed(19)
    state = capture_rng_state()
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected = torch.rand(4)
    restore_rng_state(state)
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    actual = torch.rand(4)
    assert torch.equal(expected, actual)
