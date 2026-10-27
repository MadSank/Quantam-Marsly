"""Model Comparison & Aggregate Metrics: Detailed evaluation, confusion matrices, and training curves (Mission Console styling)."""

import sys
from pathlib import Path

# Add visualization_app to path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from style.theme import apply_theme, MARSQCNN_TEMPLATE
from data_loading.mars_data import (
    load_mars_aggregate_metrics,
    load_mars_per_site_metrics,
    load_mars_site_mean_metrics,
    load_mars_validation_calibrated_metrics,
    load_mars_epoch_log,
)
from data_loading.moon_data import (
    load_moon_recomputed_metrics,
    load_moon_metrics,
    load_moon_results,
    load_moon_history,
)
from components.charts import (
    build_mars_regression_chart,
    build_mars_safety_chart,
    build_moon_comparison_chart,
    build_loss_curve_chart,
)
from components.confusion_matrix import render_confusion_matrix_heatmap
from components.metric_tables import (
    render_parameter_count_table,
    render_inference_timing_table,
    render_mars_training_timing_table,
    render_moon_training_fit_table,
)
from components.layout import page_header, note_block, section_header
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=5,
    title="Model Comparison & Aggregate Metrics",
    subtitle="Rigorous scientific comparison across geographic folds, random seeds, and classical vs quantum pipelines.",
    section_name="AGGREGATE METRICS, ERROR BARS & TRADE-OFFS",
)

tab_mars_comp, tab_moon_comp, tab_curves, tab_params = st.tabs([
    "🔴 Mars Comparative Evaluation",
    "🌕 Moon Multi-Model Benchmark",
    "📉 Training Epoch Curves",
    "⚙️ Capacity & Timing Analysis",
])

# ==============================================================================
# TAB 1: MARS COMPARISON
# ==============================================================================
with tab_mars_comp:
    section_header("Mars Terrain Continuous & Safety Metrics", "Comparing classical ResNet-18 vs hybrid QCNN across 5 held-out sites.")
    
    df_mars_agg = load_mars_aggregate_metrics()
    
    # Mode toggle: Fixed Threshold vs Validation-Calibrated
    view_mode = st.radio(
        "Operating Point / Threshold Regime:",
        ["Fixed Risk Threshold (Slope < 8.0°, Hazard < 0.05)", "Validation-Calibrated Thresholds (FPR ≤ 5% Target)"],
        horizontal=True,
    )
    
    if "Fixed" in view_mode:
        note_block(
            "integrity",
            "Fixed-Threshold Performance Trade-off",
            '<ul style="margin: 0; padding-left: 20px;">'
            '<li><strong>Continuous Regression:</strong> Hybrid QCNN achieves numerically lower mean RMSE on all three continuous terrain targets across sites.</li>'
            '<li><strong>Safety Classification Trade-off:</strong> Classical CNN achieves higher safe recall (5.56% vs 2.77%) and higher safe coverage (5.67% vs 3.02%), whereas QCNN behaves more conservatively with lower hazard false positive rate (2.78% vs 5.27%).</li>'
            '</ul>'
        )
        
        st.markdown("#### 1. Continuous Terrain Regression Errors")
        st.caption("Grouped root-mean-square errors (RMSE) for Mean Slope [°], Roughness [m], and Hazard Fraction [fraction]. Physical units labeled on each target.")
        st.plotly_chart(build_mars_regression_chart(df_mars_agg, metric_type="RMSE"), width="stretch")
        
        st.markdown("#### 2. Landing-Safety Classification Proxy Rates")
        st.caption("Rates computed under fixed operational criteria (Mean Slope < 8.0° & Hazard Fraction < 0.05). Physical units in percentage [%].")
        st.plotly_chart(build_mars_safety_chart(df_mars_agg), width="stretch")
            
        with st.expander("Full Fixed-Threshold Aggregate Metrics Table"):
            st.dataframe(df_mars_agg, width="stretch", hide_index=True)
            
        with st.expander("Per-Site and Per-Seed Metric Breakdown"):
            st.dataframe(load_mars_per_site_metrics(), width="stretch", hide_index=True)
            
    else:
        note_block(
            "warning",
            "Validation-Calibrated Threshold Findings (Domain Shift)",
            """
            When decision thresholds were calibrated fold-by-fold on the validation site to enforce a strict validation FPR ≤ 5%,
            the calibrated thresholds did <strong>not</strong> transfer safely to unseen, held-out test sites:<br/>
            - Test-site CNN Mean Safe Recall / Hazard FPR: <strong>43.33% / 39.65%</strong> (Target was ≤ 5.0%)<br/>
            - Test-site QCNN Mean Safe Recall / Hazard FPR: <strong>30.62% / 33.94%</strong> (Target was ≤ 5.0%)<br/>
            <em>High held-out false positive rates confirm severe inter-site geological domain shift, proving that single-site threshold tuning fails to generalize across Martian terrain regimes.</em>
            """
        )
        df_val_cal = load_mars_validation_calibrated_metrics()
        st.dataframe(df_val_cal, width="stretch", hide_index=True)

# ==============================================================================
# TAB 2: MOON COMPARISON
# ==============================================================================
with tab_moon_comp:
    section_header("Moon Crater Classification Performance (LROCNet)", "Multi-seed benchmark comparing Classical CNN, Hybrid QCNN, and Frozen MLP Control.")
    
    df_moon_recomputed = load_moon_recomputed_metrics()
    
    st.plotly_chart(build_moon_comparison_chart(df_moon_recomputed), width="stretch")
    
    note_block(
        "warning",
        "Imbalance & Majority Baseline Notice",
        "Because <code>none</code> constitutes 68.42% of the test partition, a naive majority-class classifier attains "
        "<strong>68.4% raw accuracy</strong>, but has a <strong>balanced accuracy of only 33.3%</strong> and <strong>macro-F1 of 0.271</strong> "
        "(zero recall on crater classes). Raw accuracy alone is therefore a misleading metric; evaluators must focus on Balanced Accuracy."
    )
    
    st.markdown("#### Test Confusion Matrices")
    col_cm_ctrl1, col_cm_ctrl2 = st.columns([1, 2])
    with col_cm_ctrl1:
        cm_seed = st.selectbox("Select Evaluation Seed:", [42, 43, 44], key="moon_cm_seed")
    with col_cm_ctrl2:
        cm_view_mode = st.radio("Display Layout:", ["All Three Models Side-by-Side", "Single Model (Enlarged)"], horizontal=True, key="moon_cm_layout")
        
    moon_seed_results = load_moon_results(seed=cm_seed)
    test_block = moon_seed_results["test"]
    
    if cm_view_mode == "All Three Models Side-by-Side":
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        with col_cm1:
            cm_cnn = test_block["cnn"]["confusion_matrix"]
            st.plotly_chart(render_confusion_matrix_heatmap(cm_cnn, title=f"Classical CNN (Seed {cm_seed})"), width="stretch")
        with col_cm2:
            cm_hyb = test_block["hybrid_qcnn"]["confusion_matrix"]
            st.plotly_chart(render_confusion_matrix_heatmap(cm_hyb, title=f"Hybrid QCNN (Seed {cm_seed})"), width="stretch")
        with col_cm3:
            cm_mlp = test_block["frozen_embedding_mlp"]["confusion_matrix"]
            st.plotly_chart(render_confusion_matrix_heatmap(cm_mlp, title=f"Frozen MLP (Seed {cm_seed})"), width="stretch")
    else:
        selected_mod = st.selectbox("Select Model to Inspect:", ["Classical CNN", "Hybrid QCNN", "Frozen MLP Control"], key="moon_single_cm_mod")
        mod_key_map = {"Classical CNN": "cnn", "Hybrid QCNN": "hybrid_qcnn", "Frozen MLP Control": "frozen_embedding_mlp"}
        cm_target = test_block[mod_key_map[selected_mod]]["confusion_matrix"]
        st.plotly_chart(render_confusion_matrix_heatmap(cm_target, title=f"{selected_mod} Confusion Matrix (Seed {cm_seed})"), width="stretch")
        
    st.markdown("#### Per-Class Precision, Recall & F1 Breakdown")
    per_class_rows = []
    for mod_key, mod_label in [("cnn", "Classical CNN"), ("hybrid_qcnn", "Hybrid QCNN"), ("frozen_embedding_mlp", "Frozen MLP")]:
        pc_dict = test_block[mod_key].get("per_class", {})
        for cls_name, vals in pc_dict.items():
            per_class_rows.append({
                "Model": mod_label,
                "Class": cls_name,
                "Precision": f"{vals.get('precision', 0):.4f}",
                "Recall": f"{vals.get('recall', 0):.4f}",
                "F1-Score": f"{vals.get('f1', 0):.4f}",
                "Support": vals.get("support", 0),
            })
    st.dataframe(pd.DataFrame(per_class_rows), width="stretch", hide_index=True)

# ==============================================================================
# TAB 3: TRAINING CURVES
# ==============================================================================
with tab_curves:
    section_header("Training Loss & Validation Progression", "Epoch-by-epoch loss convergence across models and folds.")
    
    subtab_m_curve, subtab_l_curve = st.tabs(["Mars Epoch Loss Logs", "Moon Training Progression"])
    
    with subtab_m_curve:
        col_f, col_s = st.columns(2)
        with col_f:
            curve_fold = st.selectbox("Select Mars Fold:", [0, 1, 2, 3, 4], key="curve_fold")
        with col_s:
            curve_seed = st.selectbox("Select Mars Seed:", [42, 43, 44], key="curve_seed")
            
        cnn_log = load_mars_epoch_log("resnet18", fold=curve_fold, seed=curve_seed)
        qcnn_log = load_mars_epoch_log("qcnn", fold=curve_fold, seed=curve_seed)
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.plotly_chart(
                build_loss_curve_chart(
                    epochs=cnn_log["epoch"].values,
                    train_loss=cnn_log["train_loss"].values,
                    val_loss=cnn_log["val_loss"].values if "val_loss" in cnn_log else None,
                    title=f"Classical CNN ResNet-18 (Fold {curve_fold}, Seed {curve_seed})",
                ),
                width="stretch",
            )
        with col_plot2:
            st.plotly_chart(
                build_loss_curve_chart(
                    epochs=qcnn_log["epoch"].values,
                    train_loss=qcnn_log["train_loss"].values,
                    val_loss=qcnn_log["val_loss"].values if "val_loss" in qcnn_log else None,
                    title=f"Hybrid QCNN Head (Fold {curve_fold}, Seed {curve_seed})",
                ),
                width="stretch",
            )
            
    with subtab_l_curve:
        moon_curve_seed = st.selectbox("Select Moon Seed:", [42, 43, 44], key="moon_curve_seed")
        moon_hist = load_moon_history(seed=moon_curve_seed)
        
        col_m_p1, col_m_p2 = st.columns(2)
        with col_m_p1:
            fig_hist_loss = go.Figure()
            for m_name in moon_hist["model"].unique():
                sub = moon_hist[moon_hist["model"] == m_name]
                fig_hist_loss.add_trace(go.Scatter(x=sub["epoch"], y=sub["train_loss"], mode="lines+markers", name=m_name))
            fig_hist_loss.update_layout(
                template=MARSQCNN_TEMPLATE,
                title=dict(text=f"Moon Training Loss by Epoch (Seed {moon_curve_seed})"),
                height=320,
                xaxis=dict(title="Epoch"),
                yaxis=dict(title="Train Loss"),
            )
            st.plotly_chart(fig_hist_loss, width="stretch")
            
        with col_m_p2:
            fig_hist_f1 = go.Figure()
            for m_name in moon_hist["model"].unique():
                sub = moon_hist[moon_hist["model"] == m_name]
                if "val_macro_f1" in sub:
                    fig_hist_f1.add_trace(go.Scatter(x=sub["epoch"], y=sub["val_macro_f1"], mode="lines+markers", name=m_name))
            fig_hist_f1.update_layout(
                template=MARSQCNN_TEMPLATE,
                title=dict(text=f"Moon Validation Macro-F1 by Epoch (Seed {moon_curve_seed})"),
                height=320,
                xaxis=dict(title="Epoch"),
                yaxis=dict(title="Val Macro-F1"),
            )
            st.plotly_chart(fig_hist_f1, width="stretch")

# ==============================================================================
# TAB 4: CAPACITY & TIMING ANALYSIS
# ==============================================================================
with tab_params:
    section_header("Model Capacity & Trainable Parameter Verification", "Verified parameter counts comparing classical backbones vs frozen-representation quantum and classical heads.")
    render_parameter_count_table()
    
    st.markdown("---")
    section_header("Execution Latency & Training Wall-Clock Analysis", "Audited training durations and execution latency boundaries.")
    
    note_block(
        "warning",
        "Mandatory Scientific Notice on Latency Comparability",
        "In the recorded Moon experiment logs, Classical CNN timing represents <strong>full end-to-end inference</strong> "
        "(raw 227×227 RGB image -> conv feature extraction -> classification).<br/>"
        "In contrast, recorded QCNN and MLP timings were measured <strong>only on pre-cached 8D embeddings</strong>, "
        "excluding the upstream feature extraction time.<br/>"
        "<strong>These timings are not apples-to-apples and do not substantiate a quantum latency advantage.</strong>"
    )
    
    note_block(
        "disclosure",
        "Moon Test Embedding Computation Disclosure",
        "In the completed Moon experiment run, test embeddings were computed prior to fitting the QCNN and MLP heads. "
        "<strong>Test labels were never used</strong> for gradient updates, model selection, or threshold calibration. "
        "The feature representation remained completely frozen from the classical CNN training phase."
    )
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### Moon Inference Latency Snapshot (Per Image)")
        timing_seed = st.selectbox("Select Seed for Moon Timing:", [42, 43, 44], key="timing_seed")
        moon_timing_res = load_moon_results(seed=timing_seed)
        render_inference_timing_table(moon_timing_res)
        
        st.markdown("#### Moon Training Fit Duration")
        render_moon_training_fit_table(moon_timing_res)
        
    with col_t2:
        st.markdown("#### Mars Training Wall-Clock Duration")
        col_mf, col_ms = st.columns(2)
        with col_mf:
            m_fold = st.selectbox("Fold:", [0, 1, 2, 3, 4], key="mars_timing_fold")
        with col_ms:
            m_seed = st.selectbox("Seed:", [42, 43, 44], key="mars_timing_seed")
            
        render_mars_training_timing_table(fold=m_fold, seed=m_seed)
        
        st.caption(
            "Note on Mars Inference: Batch test predictions across the 5 held-out sites were generated and serialized "
            "directly as static Parquet prediction shards without per-patch latency benchmarking."
        )

render_walkthrough_footer(
    next_page_num=6,
    next_page_title="Methodological Caveats & Provenance",
    next_page_filename="6_Methodology.py",
    next_page_icon="🔬",
    highlight_text="Highlight the 59,347 parameter correction, test embedding computation disclosure, non-comparability of cached-embedding latencies, and proxy disclaimers.",
    prev_page_num=4,
    prev_page_title="Moon Multi-Model Sample Inspection",
    prev_page_filename="4_Moon_Test_Explorer.py",
    prev_page_icon="🌕",
)
