"""Data loaders for Mars experiments (retraining seedset and HiRISE dataset)."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import yaml
import numpy as np
import pandas as pd
import streamlit as st

from data_loading.paths import (
    MARS_REPAIRED_ROOT,
    MARS_METRICS_DIR,
    MARS_RUNS_DIR,
    MARS_CONFIGS_DIR,
    MARS_PROTOCOL_DIR,
    MARS_PROGRESS_FILE,
    MARS_METADATA_PARQUET,
    MARS_DATASET_METADATA,
    MARS_SHARDS_DIR,
)

# Fold-to-site mapping as defined by leave-one-site-out protocol
FOLD_SITE_MAPPING = {
    0: {"test": "site_01", "val": "site_02", "train": ["site_03", "site_04", "site_05"]},
    1: {"test": "site_02", "val": "site_03", "train": ["site_01", "site_04", "site_05"]},
    2: {"test": "site_03", "val": "site_04", "train": ["site_01", "site_02", "site_05"]},
    3: {"test": "site_04", "val": "site_05", "train": ["site_01", "site_02", "site_03"]},
    4: {"test": "site_05", "val": "site_01", "train": ["site_02", "site_03", "site_04"]},
}

# Safety thresholds (documented research proxy)
SLOPE_SAFE_THRESHOLD = 8.0     # Mean slope < 8.0 deg
HAZARD_SAFE_THRESHOLD = 0.05   # Hazard fraction < 0.05


@st.cache_data
def load_mars_metadata() -> pd.DataFrame:
    """Load the full 15,233-row Mars metadata dataframe."""
    if not MARS_METADATA_PARQUET.exists():
        raise FileNotFoundError(f"Missing Mars metadata parquet: {MARS_METADATA_PARQUET}")
    df = pd.read_parquet(MARS_METADATA_PARQUET)
    return df


@st.cache_resource
def _get_shard_mmap(shard_filename: str) -> np.ndarray:
    """Memory-map a Mars shard array by filename."""
    shard_path = MARS_SHARDS_DIR / shard_filename
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard array at {shard_path}")
    return np.load(shard_path, mmap_mode="r")


def load_mars_patch_image(shard_ref: str, shard_idx: int) -> np.ndarray:
    """Load a single 64x64 patch from a shard and apply display percentile stretch.
    
    Channel 0 = Calibrated I/F (grayscale terrain)
    Channel 1 = DTM validity mask
    Channel 2 = Image validity mask
    
    Returns uint8 array (64, 64) in range [0, 255].
    """
    shard_filename = Path(shard_ref).name
    shard_array = _get_shard_mmap(shard_filename)
    tile_data = shard_array[shard_idx]  # shape (3, 64, 64), float16
    
    ch0 = tile_data[0].astype(np.float32)
    ch2_valid = tile_data[2] > 0
    
    valid_pixels = ch0[ch2_valid] if np.any(ch2_valid) else ch0.ravel()
    
    if len(valid_pixels) > 0 and (np.max(valid_pixels) > np.min(valid_pixels)):
        p2, p98 = np.percentile(valid_pixels, (2.0, 98.0))
        denom = p98 - p2 if (p98 - p2) > 1e-6 else 1e-6
        stretched = np.clip((ch0 - p2) / denom, 0.0, 1.0)
    else:
        stretched = np.zeros_like(ch0)
        
    return (stretched * 255.0).astype(np.uint8)


def load_mars_patch_image_by_tile(tile_id: str, metadata_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Look up a tile_id in metadata, load its image, and return metadata dict."""
    if metadata_df is None:
        metadata_df = load_mars_metadata()
        
    match = metadata_df[metadata_df["tile_id"] == tile_id]
    if match.empty:
        raise KeyError(f"tile_id '{tile_id}' not found in Mars metadata")
        
    row = match.iloc[0]
    shard_path = row["shard_path"]
    shard_idx = int(row["shard_idx"])
    img = load_mars_patch_image(shard_path, shard_idx)
    return img, row.to_dict()


@st.cache_data
def load_mars_predictions(fold: int, seed: int) -> pd.DataFrame:
    """Load paired CNN and QCNN test predictions for a given fold and seed.
    
    Returns a unified DataFrame with ground truth and both models' predictions.
    """
    cnn_path = MARS_RUNS_DIR / "model1_resnet18" / f"fold_{fold}_seed_{seed}" / "test_predictions.parquet"
    qcnn_path = MARS_RUNS_DIR / "model2_qcnn" / f"fold_{fold}_seed_{seed}" / "test_predictions.parquet"
    
    if not cnn_path.exists():
        raise FileNotFoundError(f"Missing CNN predictions at {cnn_path}")
    if not qcnn_path.exists():
        raise FileNotFoundError(f"Missing QCNN predictions at {qcnn_path}")
        
    df_cnn = pd.read_parquet(cnn_path)
    df_qcnn = pd.read_parquet(qcnn_path)
    
    # Verify alignment
    if not (df_cnn["tile_id"].values == df_qcnn["tile_id"].values).all():
        raise ValueError(f"Tile IDs in CNN and QCNN predictions for fold {fold} seed {seed} do not match!")
        
    merged = pd.DataFrame({
        "tile_id": df_cnn["tile_id"],
        "image_id": df_cnn["image_id"],
        "fold": fold,
        "seed": seed,
        "test_site": FOLD_SITE_MAPPING[fold]["test"],
        # Ground truth
        "true_mean_slope_5m": df_cnn["true_mean_slope_5m"].astype(float),
        "true_roughness_m": df_cnn["true_roughness_m"].astype(float),
        "true_hazard_fraction": df_cnn["true_hazard_fraction"].astype(float),
        "true_safe": df_cnn["true_safe"].astype(int),
        # CNN predictions
        "cnn_pred_mean_slope_5m": df_cnn["pred_mean_slope_5m"].astype(float),
        "cnn_pred_roughness_m": df_cnn["pred_roughness_m"].astype(float),
        "cnn_pred_hazard_fraction": df_cnn["pred_hazard_fraction"].astype(float),
        "cnn_pred_safe": df_cnn["pred_safe"].astype(int),
        # QCNN predictions
        "qcnn_pred_mean_slope_5m": df_qcnn["pred_mean_slope_5m"].astype(float),
        "qcnn_pred_roughness_m": df_qcnn["pred_roughness_m"].astype(float),
        "qcnn_pred_hazard_fraction": df_qcnn["pred_hazard_fraction"].astype(float),
        "qcnn_pred_safe": df_qcnn["pred_safe"].astype(int),
    })
    
    # Derived comparison metrics
    merged["agree_safe"] = merged["cnn_pred_safe"] == merged["qcnn_pred_safe"]
    merged["cnn_correct_safe"] = merged["cnn_pred_safe"] == merged["true_safe"]
    merged["qcnn_correct_safe"] = merged["qcnn_pred_safe"] == merged["true_safe"]
    merged["both_correct_safe"] = merged["cnn_correct_safe"] & merged["qcnn_correct_safe"]
    merged["both_wrong_safe"] = (~merged["cnn_correct_safe"]) & (~merged["qcnn_correct_safe"])
    
    # Signed errors (pred - true)
    merged["cnn_signed_err_slope"] = merged["cnn_pred_mean_slope_5m"] - merged["true_mean_slope_5m"]
    merged["qcnn_signed_err_slope"] = merged["qcnn_pred_mean_slope_5m"] - merged["true_mean_slope_5m"]
    merged["cnn_signed_err_roughness"] = merged["cnn_pred_roughness_m"] - merged["true_roughness_m"]
    merged["qcnn_signed_err_roughness"] = merged["qcnn_pred_roughness_m"] - merged["true_roughness_m"]
    merged["cnn_signed_err_hazard"] = merged["cnn_pred_hazard_fraction"] - merged["true_hazard_fraction"]
    merged["qcnn_signed_err_hazard"] = merged["qcnn_pred_hazard_fraction"] - merged["true_hazard_fraction"]

    # Absolute errors (|pred - true|)
    merged["cnn_abs_err_slope"] = merged["cnn_signed_err_slope"].abs()
    merged["qcnn_abs_err_slope"] = merged["qcnn_signed_err_slope"].abs()
    merged["cnn_abs_err_roughness"] = merged["cnn_signed_err_roughness"].abs()
    merged["qcnn_abs_err_roughness"] = merged["qcnn_signed_err_roughness"].abs()
    merged["cnn_abs_err_hazard"] = merged["cnn_signed_err_hazard"].abs()
    merged["qcnn_abs_err_hazard"] = merged["qcnn_signed_err_hazard"].abs()

    # Merge spatial grid metadata (row, col, coverage) from metadata.parquet
    meta_sub = load_mars_metadata()[["tile_id", "row", "col", "valid_coverage", "dtm_valid_coverage", "image_valid_coverage"]].drop_duplicates(subset=["tile_id"])
    merged = merged.merge(meta_sub, on="tile_id", how="left")
    
    # Aliases for convenience & test compatibility
    merged["site_id"] = merged["test_site"]
    merged["cnn_err_slope"] = merged["cnn_signed_err_slope"]
    merged["qcnn_err_slope"] = merged["qcnn_signed_err_slope"]
    merged["cnn_err_roughness"] = merged["cnn_signed_err_roughness"]
    merged["qcnn_err_roughness"] = merged["qcnn_signed_err_roughness"]
    merged["cnn_err_hazard"] = merged["cnn_signed_err_hazard"]
    merged["qcnn_err_hazard"] = merged["qcnn_signed_err_hazard"]

    return merged


MARS_DEMO_PRESETS = [
    "Manual / Filter Selection",
    "Both Models Agree Safe (True Positive)",
    "Model Disagreement (CNN != QCNN)",
    "CNN False Positive (Unsafe predicted Safe)",
    "QCNN False Negative (Safe predicted Unsafe)",
]


def get_mars_demo_presets(df_preds: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return dictionary of demo preset subsets for Mars test patches.
    
    Categories:
    - Both Models Agree Safe (True Positive): both models predict safe and ground truth is safe
    - Model Disagreement: CNN predicted safe != QCNN predicted safe
    - CNN False Positive: CNN predicted safe but ground truth is unsafe
    - QCNN False Negative: QCNN predicted unsafe but ground truth is safe
    """
    return {
        "Manual / Filter Selection": df_preds,
        "Both Models Agree Safe (True Positive)": df_preds[
            (df_preds["both_correct_safe"]) & (df_preds["true_safe"] == 1)
        ],
        "Model Disagreement (CNN != QCNN)": df_preds[~df_preds["agree_safe"]],
        "CNN False Positive (Unsafe predicted Safe)": df_preds[
            (df_preds["cnn_pred_safe"] == 1) & (df_preds["true_safe"] == 0)
        ],
        "QCNN False Negative (Safe predicted Unsafe)": df_preds[
            (df_preds["qcnn_pred_safe"] == 0) & (df_preds["true_safe"] == 1)
        ],
    }


@st.cache_data
def load_mars_aggregate_metrics() -> pd.DataFrame:
    """Load primary fixed-threshold aggregate metrics CSV."""
    path = MARS_METRICS_DIR / "fixed_threshold_aggregate.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_per_site_metrics() -> pd.DataFrame:
    """Load per-site-seed metrics table."""
    path = MARS_METRICS_DIR / "fixed_threshold_per_site_seed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_site_mean_metrics() -> pd.DataFrame:
    """Load site-mean metrics table."""
    path = MARS_METRICS_DIR / "fixed_threshold_site_mean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_paired_differences() -> pd.DataFrame:
    """Load paired site differences (QCNN minus CNN)."""
    path = MARS_METRICS_DIR / "fixed_threshold_paired_site_differences.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_validation_calibrated_metrics() -> pd.DataFrame:
    """Load validation-calibrated test aggregate metrics."""
    path = MARS_METRICS_DIR / "validation_calibrated_test_aggregate.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_validation_calibrated_paired() -> pd.DataFrame:
    """Load validation-calibrated paired site differences."""
    path = MARS_METRICS_DIR / "validation_calibrated_test_paired_site_differences.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_epoch_log(model_name: str, fold: int, seed: int) -> pd.DataFrame:
    """Load training epoch history for a specific model run."""
    dir_name = "model1_resnet18" if "resnet" in model_name.lower() or "cnn" in model_name.lower() else "model2_qcnn"
    path = MARS_RUNS_DIR / dir_name / f"fold_{fold}_seed_{seed}" / "epoch_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing epoch log at {path}")
    return pd.read_csv(path)


@st.cache_data
def load_mars_config(fold: int, seed: int) -> Dict[str, Any]:
    """Load configuration YAML snapshot for a fold/seed run."""
    path = MARS_CONFIGS_DIR / f"fold_{fold}_seed_{seed}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config YAML at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_mars_dataset_metadata() -> Dict[str, Any]:
    """Load dataset metadata JSON."""
    if not MARS_DATASET_METADATA.exists():
        raise FileNotFoundError(f"Missing dataset metadata at {MARS_DATASET_METADATA}")
    with open(MARS_DATASET_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_mars_param_counts(model_name: str, fold: int = 0, seed: int = 42) -> Dict[str, Any]:
    """Load parameter count JSON for a model run."""
    dir_name = "model1_resnet18" if "resnet" in model_name.lower() or "cnn" in model_name.lower() else "model2_qcnn"
    path = MARS_RUNS_DIR / dir_name / f"fold_{fold}_seed_{seed}" / "param_counts.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
