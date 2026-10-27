"""Milestone 1 Acceptance Tests: Data Loading Layer."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Add visualization_app to path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.mars_data import (
    load_mars_metadata,
    load_mars_patch_image_by_tile,
    load_mars_predictions,
    load_mars_aggregate_metrics,
    load_mars_per_site_metrics,
    load_mars_site_mean_metrics,
    load_mars_paired_differences,
    load_mars_validation_calibrated_metrics,
    load_mars_epoch_log,
    load_mars_config,
    load_mars_dataset_metadata,
)
from data_loading.moon_data import (
    load_moon_results,
    load_moon_image,
    load_moon_recomputed_metrics,
    load_moon_metrics,
    load_moon_provenance,
    get_moon_predictions_df,
    load_moon_history,
    softmax,
    CORRECTED_CNN_TRAINABLE_PARAMS,
)


def test_mars_metadata_loading():
    df = load_mars_metadata()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 15233
    assert "tile_id" in df.columns
    assert "site_id" in df.columns
    assert "shard_path" in df.columns
    assert "shard_idx" in df.columns


def test_mars_predictions_loading():
    df = load_mars_predictions(fold=0, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2195
    assert "cnn_pred_mean_slope_5m" in df.columns
    assert "qcnn_pred_mean_slope_5m" in df.columns
    assert "true_mean_slope_5m" in df.columns
    assert "agree_safe" in df.columns
    assert "both_correct_safe" in df.columns


def test_mars_patch_image_loading():
    df_meta = load_mars_metadata()
    first_tile_id = df_meta["tile_id"].iloc[0]
    img, meta = load_mars_patch_image_by_tile(first_tile_id, df_meta)
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_mars_aggregate_metrics():
    df = load_mars_aggregate_metrics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "model_name" in df.columns or "model" in df.columns or "metric" in df.columns


def test_moon_results_loading():
    data = load_moon_results(seed=42)
    assert isinstance(data, dict)
    assert "test" in data
    assert len(data["test"]["sample_ids"]) == 779
    assert len(data["test"]["ground_truth"]) == 779
    assert "cnn" in data["test"]["predictions"]
    assert "hybrid_qcnn" in data["test"]["predictions"]
    assert "frozen_embedding_mlp" in data["test"]["predictions"]
    assert "parameter_counts" in data
    assert data["parameter_counts"]["cnn"]["trainable_during_training"] == CORRECTED_CNN_TRAINABLE_PARAMS


def test_moon_image_loading():
    data = load_moon_results(seed=42)
    first_sample_id = data["test"]["sample_ids"][0]
    img = load_moon_image(first_sample_id)
    assert isinstance(img, Image.Image)
    assert img.size == (227, 227)
    assert img.mode == "RGB"


def test_moon_predictions_df_and_softmax():
    df = get_moon_predictions_df(seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 779
    assert "cnn_confidence" in df.columns
    assert "hybrid_confidence" in df.columns
    assert "mlp_confidence" in df.columns
    
    # Verify probabilities sum to ~1.0
    cnn_prob_sum = df["cnn_prob_ejecta"] + df["cnn_prob_oldcrater"] + df["cnn_prob_none"]
    assert np.allclose(cnn_prob_sum, 1.0, atol=1e-5)


def test_moon_aggregate_metrics():
    df = load_moon_recomputed_metrics()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3
    assert "Model" in df.columns or "model" in df.columns


def test_moon_cross_seed_sample_id_alignment():
    """Verify that all 3 seeds share identical 779 test sample IDs and ground truth targets."""
    results_42 = load_moon_results(seed=42)
    results_43 = load_moon_results(seed=43)
    results_44 = load_moon_results(seed=44)

    assert len(results_42["test"]["sample_ids"]) == 779
    assert len(results_43["test"]["sample_ids"]) == 779
    assert len(results_44["test"]["sample_ids"]) == 779

    assert results_42["test"]["sample_ids"] == results_43["test"]["sample_ids"]
    assert results_42["test"]["sample_ids"] == results_44["test"]["sample_ids"]

    assert results_42["test"]["ground_truth"] == results_43["test"]["ground_truth"]
    assert results_42["test"]["ground_truth"] == results_44["test"]["ground_truth"]


def test_mars_metadata_and_signed_errors_alignment():
    """Verify that Mars predictions dataframe includes correctly aligned site, image, row, col and signed errors."""
    df = load_mars_predictions(fold=0, seed=42)
    assert len(df) == 2195
    assert "site_id" in df.columns
    assert "image_id" in df.columns
    assert "row" in df.columns
    assert "col" in df.columns

    # Verify no nulls in spatial coordinates
    assert not df["site_id"].isnull().any()
    assert not df["image_id"].isnull().any()
    assert not df["row"].isnull().any()
    assert not df["col"].isnull().any()

    # Verify signed error calculations
    assert "cnn_err_slope" in df.columns
    assert "qcnn_err_slope" in df.columns
    assert "cnn_err_roughness" in df.columns
    assert "qcnn_err_roughness" in df.columns
    assert "cnn_err_hazard" in df.columns
    assert "qcnn_err_hazard" in df.columns

    expected_cnn_err_slope = df["cnn_pred_mean_slope_5m"] - df["true_mean_slope_5m"]
    assert np.allclose(df["cnn_err_slope"], expected_cnn_err_slope, atol=1e-5)

    expected_qcnn_err_slope = df["qcnn_pred_mean_slope_5m"] - df["true_mean_slope_5m"]
    assert np.allclose(df["qcnn_err_slope"], expected_qcnn_err_slope, atol=1e-5)

