"""Data loaders for Moon crater classification experiments (LROCNet dataset)."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from data_loading.paths import (
    MOON_ROOT,
    MOON_LOGS_DIR,
    MOON_METRICS_CSV,
    MOON_RECOMPUTED_METRICS_CSV,
    MOON_PROVENANCE_MANIFEST,
    MOON_CONFIG_YAML,
)

CLASS_NAMES = ["ejecta", "oldcrater", "none"]
CLASS_IDX_TO_NAME = {0: "ejecta", 1: "oldcrater", 2: "none"}
CLASS_NAME_TO_IDX = {"ejecta": 0, "oldcrater": 1, "none": 2}

# Independent audit verified counts:
# During CNN training: 59,347 trainable parameters
# Post-freeze for hybrid/MLP head fitting: 195 trainable classifier parameters
CORRECTED_CNN_TRAINABLE_PARAMS = 59347


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along the last axis."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


@st.cache_data
def load_moon_results(seed: int) -> Dict[str, Any]:
    """Load results.json for a given seed and apply parameter count corrections."""
    results_path = MOON_LOGS_DIR / f"seed_{seed}" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Moon results at {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Annotate corrected parameter counts without silently mutating original source file
    param_counts = data.get("parameter_counts", {})
    if "cnn" in param_counts:
        param_counts["cnn"]["trainable_during_training"] = CORRECTED_CNN_TRAINABLE_PARAMS
        param_counts["cnn"]["trainable_recorded_post_freeze"] = param_counts["cnn"].get("trainable", 195)
        param_counts["cnn"]["correction_note"] = (
            "Independent audit confirmed 59,347 trainable parameters during initial CNN fitting. "
            "The recorded 195 represents only the trainable linear classifier after freezing the encoder."
        )
    return data


@st.cache_data
def load_moon_provenance() -> pd.DataFrame:
    """Load the Moon provenance manifest containing 5,000 images."""
    if not MOON_PROVENANCE_MANIFEST.exists():
        raise FileNotFoundError(f"Missing Moon provenance manifest at {MOON_PROVENANCE_MANIFEST}")
    return pd.read_csv(MOON_PROVENANCE_MANIFEST)


def load_moon_image(sample_id: str) -> Image.Image:
    """Load a Moon image by relative path from MOON_ROOT.
    
    Example sample_id: 'test/ejecta/M1122721870LC_pyr-1052.jpg'
    """
    img_path = MOON_ROOT / sample_id
    if not img_path.exists():
        # Fallback if sample_id has different slash separators
        normalized = sample_id.replace("\\", "/").strip("/")
        img_path = MOON_ROOT / normalized
    if not img_path.exists():
        raise FileNotFoundError(f"Moon image file not found at {img_path}")
    return Image.open(img_path).convert("RGB")


@st.cache_data
def get_moon_predictions_df(seed: int) -> pd.DataFrame:
    """Construct a unified test-predictions DataFrame for all 779 test samples."""
    results = load_moon_results(seed)
    test_data = results["test"]
    
    sample_ids = test_data["sample_ids"]
    ground_truth = np.array(test_data["ground_truth"], dtype=int)
    
    preds_dict = test_data["predictions"]
    logits_dict = test_data["logits"]
    
    cnn_logits = np.array(logits_dict["cnn"])
    hybrid_logits = np.array(logits_dict["hybrid_qcnn"])
    mlp_logits = np.array(logits_dict["frozen_embedding_mlp"])
    
    cnn_probs = softmax(cnn_logits)
    hybrid_probs = softmax(hybrid_logits)
    mlp_probs = softmax(mlp_logits)
    
    cnn_preds = np.array(preds_dict["cnn"], dtype=int)
    hybrid_preds = np.array(preds_dict["hybrid_qcnn"], dtype=int)
    mlp_preds = np.array(preds_dict["frozen_embedding_mlp"], dtype=int)
    
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "filename": [Path(s).name for s in sample_ids],
        "seed": seed,
        "true_label_idx": ground_truth,
        "true_label_name": [CLASS_IDX_TO_NAME[idx] for idx in ground_truth],
        
        # CNN
        "cnn_pred_idx": cnn_preds,
        "cnn_pred_name": [CLASS_IDX_TO_NAME[idx] for idx in cnn_preds],
        "cnn_prob_ejecta": cnn_probs[:, 0],
        "cnn_prob_oldcrater": cnn_probs[:, 1],
        "cnn_prob_none": cnn_probs[:, 2],
        "cnn_logit_ejecta": cnn_logits[:, 0],
        "cnn_logit_oldcrater": cnn_logits[:, 1],
        "cnn_logit_none": cnn_logits[:, 2],
        "cnn_confidence": np.max(cnn_probs, axis=-1),
        "cnn_correct": cnn_preds == ground_truth,
        
        # Hybrid QCNN
        "hybrid_pred_idx": hybrid_preds,
        "hybrid_pred_name": [CLASS_IDX_TO_NAME[idx] for idx in hybrid_preds],
        "hybrid_prob_ejecta": hybrid_probs[:, 0],
        "hybrid_prob_oldcrater": hybrid_probs[:, 1],
        "hybrid_prob_none": hybrid_probs[:, 2],
        "hybrid_logit_ejecta": hybrid_logits[:, 0],
        "hybrid_logit_oldcrater": hybrid_logits[:, 1],
        "hybrid_logit_none": hybrid_logits[:, 2],
        "hybrid_confidence": np.max(hybrid_probs, axis=-1),
        "hybrid_correct": hybrid_preds == ground_truth,
        
        # Frozen MLP
        "mlp_pred_idx": mlp_preds,
        "mlp_pred_name": [CLASS_IDX_TO_NAME[idx] for idx in mlp_preds],
        "mlp_prob_ejecta": mlp_probs[:, 0],
        "mlp_prob_oldcrater": mlp_probs[:, 1],
        "mlp_prob_none": mlp_probs[:, 2],
        "mlp_logit_ejecta": mlp_logits[:, 0],
        "mlp_logit_oldcrater": mlp_logits[:, 1],
        "mlp_logit_none": mlp_logits[:, 2],
        "mlp_confidence": np.max(mlp_probs, axis=-1),
        "mlp_correct": mlp_preds == ground_truth,
    })
    
    # Model agreement logic
    df["all_agree"] = (df["cnn_pred_idx"] == df["hybrid_pred_idx"]) & (df["hybrid_pred_idx"] == df["mlp_pred_idx"])
    df["has_disagreement"] = ~df["all_agree"]
    df["all_correct"] = df["cnn_correct"] & df["hybrid_correct"] & df["mlp_correct"]
    df["all_wrong"] = (~df["cnn_correct"]) & (~df["hybrid_correct"]) & (~df["mlp_correct"])
    df["hybrid_beats_cnn"] = df["hybrid_correct"] & (~df["cnn_correct"])
    df["cnn_beats_hybrid"] = df["cnn_correct"] & (~df["hybrid_correct"])
    
    return df


@st.cache_data
def load_moon_recomputed_metrics() -> pd.DataFrame:
    """Load the independently recomputed headline metrics (mean +/- SD)."""
    if not MOON_RECOMPUTED_METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing recomputed metrics at {MOON_RECOMPUTED_METRICS_CSV}")
    return pd.read_csv(MOON_RECOMPUTED_METRICS_CSV)


@st.cache_data
def load_moon_metrics() -> pd.DataFrame:
    """Load per-seed metrics CSV (12 rows: 4 models x 3 seeds)."""
    if not MOON_METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing {MOON_METRICS_CSV}")
    return pd.read_csv(MOON_METRICS_CSV)


@st.cache_data
def load_moon_history(seed: int) -> pd.DataFrame:
    """Load training history CSV for a seed."""
    path = MOON_LOGS_DIR / f"seed_{seed}" / "history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing history log at {path}")
    return pd.read_csv(path)


MOON_DEMO_PRESETS = [
    "Manual / Filter Selection",
    "All 3 Models Agree & Correct",
    "Hybrid QCNN Correct / Classical CNN Wrong",
    "Classical CNN Correct / Hybrid QCNN Wrong",
    "All 3 Models Wrong (Hard Crater Example)",
]


def get_moon_demo_presets(df_preds: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return dictionary of demo preset subsets for Moon test images.
    
    Categories:
    - All 3 Models Agree & Correct: all 3 models predict true label
    - Hybrid QCNN Correct / Classical CNN Wrong: Hybrid QCNN correct and classical CNN incorrect
    - Classical CNN Correct / Hybrid QCNN Wrong: Classical CNN correct and Hybrid QCNN incorrect
    - All 3 Models Wrong: None of the 3 models predict the true class
    """
    return {
        "Manual / Filter Selection": df_preds,
        "All 3 Models Agree & Correct": df_preds[df_preds["all_correct"]],
        "Hybrid QCNN Correct / Classical CNN Wrong": df_preds[df_preds["hybrid_beats_cnn"]],
        "Classical CNN Correct / Hybrid QCNN Wrong": df_preds[df_preds["cnn_beats_hybrid"]],
        "All 3 Models Wrong (Hard Crater Example)": df_preds[df_preds["all_wrong"]],
    }

