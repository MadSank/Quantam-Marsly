"""Tests for grouped cross-validation, train-only TargetScaler, and sun-angle augmentation."""

from typing import Any

import numpy as np

from src.data.dataset import TargetScaler, transform_sun_azimuth
from src.data.splits import assign_grouped_folds, get_fold_indices


def test_zero_image_id_overlap(synthetic_workspace: dict[str, Any]) -> None:
    """Verifies that across all 5 folds, train, val, and test splits share zero image_ids."""
    metadata_df = synthetic_workspace["metadata_df"]
    n_splits = 5
    df_with_folds = assign_grouped_folds(metadata_df, n_splits=n_splits, seed=42)

    for test_fold in range(n_splits):
        splits = get_fold_indices(df_with_folds, test_fold=test_fold, n_splits=n_splits)
        train_images = set(splits["train"]["image_id"])
        val_images = set(splits["val"]["image_id"])
        test_images = set(splits["test"]["image_id"])

        assert len(train_images) > 0, f"Fold {test_fold}: train set has 0 images"
        assert len(val_images) > 0, f"Fold {test_fold}: val set has 0 images"
        assert len(test_images) > 0, f"Fold {test_fold}: test set has 0 images"

        assert train_images.isdisjoint(val_images), (
            f"Fold {test_fold}: Leakage between train and val: {train_images & val_images}"
        )
        assert train_images.isdisjoint(test_images), (
            f"Fold {test_fold}: Leakage between train and test: {train_images & test_images}"
        )
        assert val_images.isdisjoint(test_images), (
            f"Fold {test_fold}: Leakage between val and test: {val_images & test_images}"
        )


def test_standardization_from_training_folds_only(synthetic_workspace: dict[str, Any]) -> None:
    """Verifies that TargetScaler is computed strictly on train folds and reconstructs accurately."""
    metadata_df = synthetic_workspace["metadata_df"]
    config = synthetic_workspace["config"]
    target_cols = config["preprocessing"]["target_cols"]
    log1p_cols = config["preprocessing"]["log1p_targets"]

    splits = get_fold_indices(metadata_df, test_fold=0, n_splits=5)
    train_df = splits["train"]
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    scaler = TargetScaler(target_cols=target_cols, log1p_cols=log1p_cols)
    scaler.fit(train_df)

    # Corrupt val and test with huge values (1e6) to guarantee scaler did not see them
    for col in target_cols:
        val_df[col] = 1e6
        test_df[col] = 1e6

    # Verify scaler statistics match train_df exactly
    raw_train = train_df[target_cols].to_numpy(dtype=np.float64).copy()
    for idx, col in enumerate(target_cols):
        if col in log1p_cols:
            raw_train[:, idx] = np.log1p(np.maximum(raw_train[:, idx], 0.0))

    expected_mean = raw_train.mean(axis=0)
    expected_std = np.where(raw_train.std(axis=0) < 1e-8, 1.0, raw_train.std(axis=0))

    assert np.allclose(scaler.mean_, expected_mean, atol=1e-6), "Scaler mean leaked outside train_df"
    assert np.allclose(scaler.std_, expected_std, atol=1e-6), "Scaler std leaked outside train_df"

    # Verify inverse_transform recovers raw train targets
    scaled_train = scaler.transform(train_df)
    recovered_train = scaler.inverse_transform(scaled_train)
    raw_original = train_df[target_cols].to_numpy(dtype=np.float32)

    assert np.allclose(recovered_train, raw_original, rtol=1e-4, atol=1e-4), (
        "inverse_transform failed to reconstruct original targets within tolerance."
    )


def test_sun_azimuth_geometric_augmentation() -> None:
    """Verifies analytic updating of solar azimuth under rot90 and reflections."""
    initial_az = 90.0  # Sun is East

    # Counter-clockwise 90 deg rotation shifts East (90 deg) to South (180 deg) or North (0 deg)
    # rot90_k = 1: (90 - 90) % 360 = 0.0
    rot1 = transform_sun_azimuth(initial_az, rot90_k=1)
    assert np.isclose(rot1, 0.0), f"Expected 0.0, got {rot1}"

    # rot90_k = 2: (90 - 180) % 360 = 270.0
    rot2 = transform_sun_azimuth(initial_az, rot90_k=2)
    assert np.isclose(rot2, 270.0), f"Expected 270.0, got {rot2}"

    # Horizontal flip (reflection across North-South axis): (360 - 90) % 360 = 270.0
    flip_h = transform_sun_azimuth(initial_az, flip_h=True)
    assert np.isclose(flip_h, 270.0), f"Expected 270.0, got {flip_h}"

    # Vertical flip (reflection across East-West axis): (180 - 90) % 360 = 90.0
    flip_v = transform_sun_azimuth(initial_az, flip_v=True)
    assert np.isclose(flip_v, 90.0), f"Expected 90.0, got {flip_v}"
