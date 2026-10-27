"""Metric tables formatting and rendering utilities (Mission Console styling)."""

from typing import Dict, Any
import pandas as pd
import streamlit as st


def render_mars_paired_differences_table(paired_df: pd.DataFrame) -> None:
    """Render the paired differences table with explicit CI-crosses-zero indicators.
    
    Preserved in component library for programmatic evaluation; intentionally removed from Page 5 UI.
    """
    formatted_rows = []
    
    for _, row in paired_df.iterrows():
        metric = row.get("metric", "")
        mean_diff = row.get("mean_difference", row.get("mean_site_diff", 0.0))
        t95_low = row.get("t95_low", 0.0)
        t95_high = row.get("t95_high", 0.0)
        
        crosses_zero = (t95_low <= 0.0) and (t95_high >= 0.0)
        verdict = "⚠️ Crosses 0 (No clear winner)" if crosses_zero else ("✅ Favors QCNN" if mean_diff < 0 else "✅ Favors CNN")
        
        formatted_rows.append({
            "Target / Metric": metric,
            "Mean Diff (QCNN - CNN)": f"{mean_diff:+.4f}",
            "Nominal 95% t-Interval": f"[{t95_low:+.4f}, {t95_high:+.4f}]",
            "Zero Bound Check": "Crosses Zero" if crosses_zero else "Does not cross zero",
            "Statistical Assessment": verdict,
        })
        
    st.dataframe(pd.DataFrame(formatted_rows), width="stretch", hide_index=True)


def render_parameter_count_table() -> None:
    """Render verified parameter count comparison table."""
    param_data = [
        {
            "Experiment": "Mars Terrain",
            "Model Variant": "Classical ResNet-18",
            "Total Parameters": "11,244,102",
            "Trainable Parameters": "11,242,563",
            "Training Regime": "End-to-end backpropagation from random initialization",
            "Status": "Verified in param_counts.json",
        },
        {
            "Experiment": "Mars Terrain",
            "Model Variant": "Hybrid QCNN Head",
            "Total Parameters": "4,152 (Head) + 11.2M (Frozen)",
            "Trainable Parameters": "4,152 (in QCNN head)",
            "Training Regime": "Fitted on supervised, frozen CNN representation",
            "Status": "Verified in param_counts.json",
        },
        {
            "Experiment": "Moon Crater",
            "Model Variant": "Classical CNN",
            "Total Parameters": "59,347",
            "Trainable Parameters": "59,347 (during training)",
            "Training Regime": "End-to-end backpropagation from random initialization",
            "Status": "Corrected from 195 (post-freeze) by independent audit",
        },
        {
            "Experiment": "Moon Crater",
            "Model Variant": "Hybrid CNN-QCNN",
            "Total Parameters": "568 (Head) + 59.3k (Frozen)",
            "Trainable Parameters": "568 (head only)",
            "Training Regime": "Fitted on supervised, frozen CNN embeddings",
            "Status": "Verified in results_audit.json",
        },
        {
            "Experiment": "Moon Crater",
            "Model Variant": "Frozen-Embedding MLP",
            "Total Parameters": "547 (Head) + 59.3k (Frozen)",
            "Trainable Parameters": "547 (head only)",
            "Training Regime": "Fitted on supervised, frozen CNN embeddings",
            "Status": "Matched classical head control (547 params)",
        },
    ]
    st.dataframe(pd.DataFrame(param_data), width="stretch", hide_index=True)


def render_inference_timing_table(moon_results: Dict[str, Any]) -> None:
    """Render inference latency table with mandatory non-comparability caveat."""
    inf_data = moon_results.get("test", {}).get("inference", {})
    
    rows = []
    if "cnn" in inf_data:
        rows.append({
            "Model Variant": "Classical CNN",
            "Total Inference Time": f"{inf_data['cnn'].get('total_seconds', 0):.3f} s",
            "Latency per Image": f"{inf_data['cnn'].get('milliseconds_per_image', 0):.2f} ms",
            "Scope Measured": "Full end-to-end inference (Image -> Backbone -> Prediction)",
            "Comparability Note": "Baseline full-pipeline latency",
        })
    if "hybrid_qcnn" in inf_data:
        rows.append({
            "Model Variant": "Hybrid QCNN Head",
            "Total Inference Time": f"{inf_data['hybrid_qcnn'].get('total_seconds', 0):.3f} s",
            "Latency per Image": f"{inf_data['hybrid_qcnn'].get('milliseconds_per_image', 0):.2f} ms",
            "Scope Measured": "Head-only evaluation on pre-computed cached embeddings",
            "Comparability Note": "⚠️ Excludes image encoder time. NOT apples-to-apples.",
        })
    if "frozen_embedding_mlp" in inf_data:
        rows.append({
            "Model Variant": "Frozen MLP Control",
            "Total Inference Time": f"{inf_data['frozen_embedding_mlp'].get('total_seconds', 0):.3f} s",
            "Latency per Image": f"{inf_data['frozen_embedding_mlp'].get('milliseconds_per_image', 0):.2f} ms",
            "Scope Measured": "Head-only evaluation on pre-computed cached embeddings",
            "Comparability Note": "⚠️ Excludes image encoder time. Matched with QCNN head.",
        })
        
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_mars_training_timing_table(fold: int = 0, seed: int = 42) -> pd.DataFrame:
    """Render Mars wall-clock training duration breakdown from verified epoch logs."""
    from data_loading.mars_data import load_mars_epoch_log
    
    cnn_log = load_mars_epoch_log("model1_resnet18", fold, seed)
    qcnn_log = load_mars_epoch_log("model2_qcnn", fold, seed)
    
    cnn_total = cnn_log["cumulative_wall_time_sec"].iloc[-1]
    qcnn_total = qcnn_log["cumulative_wall_time_sec"].iloc[-1]
    
    rows = [
        {
            "Model Variant": "Classical ResNet-18",
            "Epochs Trained": len(cnn_log),
            "Total Wall Time": f"{cnn_total:.1f} s ({cnn_total/60.0:.2f} min)",
            "Mean Seconds / Epoch": f"{cnn_log['epoch_wall_time_sec'].mean():.2f} s",
            "Execution Mode": "Full end-to-end PyTorch backpropagation",
        },
        {
            "Model Variant": "Hybrid QCNN Head",
            "Epochs Trained": len(qcnn_log),
            "Total Wall Time": f"{qcnn_total:.1f} s ({qcnn_total/60.0:.2f} min)",
            "Mean Seconds / Epoch": f"{qcnn_log['epoch_wall_time_sec'].mean():.2f} s",
            "Execution Mode": "PennyLane state-vector simulation on frozen embeddings",
        },
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    return df


def render_moon_training_fit_table(moon_results: Dict[str, Any]) -> pd.DataFrame:
    """Render Moon training fit duration table from results.json."""
    sec = moon_results.get("seconds", {})
    cnn_fit = sec.get("cnn_fit", 0.0)
    qcnn_fit = sec.get("qcnn_head_fit", 0.0)
    
    rows = [
        {
            "Pipeline Stage": "Classical CNN Training",
            "Wall-Clock Duration": f"{cnn_fit:.2f} s",
            "Description": "End-to-end training of convolutional backbone + classifier (59,347 params)",
        },
        {
            "Pipeline Stage": "Hybrid QCNN Head Optimization",
            "Wall-Clock Duration": f"{qcnn_fit:.2f} s",
            "Description": "Variational circuit angle parameter updates on cached 8D embeddings (568 params)",
        },
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    return df
