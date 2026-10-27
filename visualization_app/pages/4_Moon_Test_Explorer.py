"""Moon Test Explorer: Multi-model evaluation and confidence inspection on 779 lunar test images (Mission Console styling)."""

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

from style.theme import (
    apply_theme,
    badge_html,
    COLOR_CNN,
    COLOR_HYBRID,
    COLOR_MLP,
    MARSQCNN_TEMPLATE,
)
from data_loading.moon_data import (
    load_moon_results,
    load_moon_image,
    get_moon_predictions_df,
    get_moon_demo_presets,
    MOON_DEMO_PRESETS,
    CLASS_NAMES,
    CLASS_IDX_TO_NAME,
)
from components.layout import page_header, note_block, section_header, kv_table, model_chip
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=4,
    title="Moon Test Explorer",
    subtitle="Inspect predictions, softmax confidence, and model disagreements across 779 held-out LROCNet test images.",
    section_name="MOON MULTI-MODEL SAMPLE INSPECTION",
)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Test Evaluation Setup")
selected_seed = st.sidebar.selectbox("Evaluation Seed", options=[42, 43, 44], index=0, key="moon_test_seed")

# Load predictions DataFrame
df_moon_preds = get_moon_predictions_df(seed=selected_seed)
results_raw = load_moon_results(seed=selected_seed)

st.sidebar.markdown("### Filter Images")
class_filter = st.sidebar.selectbox(
    "Ground Truth Class",
    options=["All Classes", "ejecta", "oldcrater", "none"],
    key="moon_test_class",
)

agreement_filter = st.sidebar.selectbox(
    "Prediction Status & Disagreement",
    options=[
        "All 779 Images",
        "Show Only Disagreements (CNN != Hybrid or Hybrid != MLP)",
        "Hybrid Beats CNN (Hybrid Correct, CNN Wrong)",
        "CNN Beats Hybrid (CNN Correct, Hybrid Wrong)",
        "All Models Correct",
        "All Models Wrong",
        "CNN Correct Only",
        "CNN Wrong Only",
        "Hybrid QCNN Correct Only",
        "Hybrid QCNN Wrong Only",
    ],
    key="moon_test_agree",
)

st.sidebar.markdown("### Confusion Pair Isolation")
confusion_model = st.sidebar.selectbox(
    "Confusion Reference Model",
    options=["Hybrid QCNN", "Classical CNN", "Frozen MLP"],
    key="moon_conf_model",
)
confusion_pair = st.sidebar.selectbox(
    "Confusion Transition (True → Pred)",
    options=[
        "All Pairs",
        "ejecta → ejecta (Correct)",
        "ejecta → oldcrater (Misclassified)",
        "ejecta → none (Misclassified)",
        "oldcrater → ejecta (Misclassified)",
        "oldcrater → oldcrater (Correct)",
        "oldcrater → none (Misclassified)",
        "none → ejecta (Misclassified)",
        "none → oldcrater (Misclassified)",
        "none → none (Correct)",
    ],
    key="moon_conf_pair",
)

sort_option = st.sidebar.selectbox(
    "Sort By",
    options=[
        "Default (Sample Order)",
        "CNN Confidence (High to Low)",
        "Hybrid Confidence (High to Low)",
        "MLP Confidence (High to Low)",
        "Hybrid vs CNN Confidence Gap (High to Low)",
    ],
    key="moon_test_sort",
)

# Apply filters
filtered_df = df_moon_preds.copy()
if class_filter != "All Classes":
    filtered_df = filtered_df[filtered_df["true_label_name"] == class_filter]

if agreement_filter == "Show Only Disagreements (CNN != Hybrid or Hybrid != MLP)":
    filtered_df = filtered_df[filtered_df["has_disagreement"]]
elif agreement_filter == "Hybrid Beats CNN (Hybrid Correct, CNN Wrong)":
    filtered_df = filtered_df[filtered_df["hybrid_beats_cnn"]]
elif agreement_filter == "CNN Beats Hybrid (CNN Correct, Hybrid Wrong)":
    filtered_df = filtered_df[filtered_df["cnn_beats_hybrid"]]
elif agreement_filter == "All Models Correct":
    filtered_df = filtered_df[filtered_df["all_correct"]]
elif agreement_filter == "All Models Wrong":
    filtered_df = filtered_df[filtered_df["all_wrong"]]
elif agreement_filter == "CNN Correct Only":
    filtered_df = filtered_df[filtered_df["cnn_correct"]]
elif agreement_filter == "CNN Wrong Only":
    filtered_df = filtered_df[~filtered_df["cnn_correct"]]
elif agreement_filter == "Hybrid QCNN Correct Only":
    filtered_df = filtered_df[filtered_df["hybrid_correct"]]
elif agreement_filter == "Hybrid QCNN Wrong Only":
    filtered_df = filtered_df[~filtered_df["hybrid_correct"]]

if confusion_pair != "All Pairs":
    true_target = confusion_pair.split(" → ")[0].strip()
    pred_target = confusion_pair.split(" → ")[1].split(" ")[0].strip()
    pred_col = (
        "hybrid_pred_name"
        if confusion_model == "Hybrid QCNN"
        else ("cnn_pred_name" if confusion_model == "Classical CNN" else "mlp_pred_name")
    )
    filtered_df = filtered_df[
        (filtered_df["true_label_name"] == true_target) & (filtered_df[pred_col] == pred_target)
    ]

# Apply sorting
if sort_option == "CNN Confidence (High to Low)":
    filtered_df = filtered_df.sort_values(by="cnn_confidence", ascending=False)
elif sort_option == "Hybrid Confidence (High to Low)":
    filtered_df = filtered_df.sort_values(by="hybrid_confidence", ascending=False)
elif sort_option == "MLP Confidence (High to Low)":
    filtered_df = filtered_df.sort_values(by="mlp_confidence", ascending=False)
elif sort_option == "Hybrid vs CNN Confidence Gap (High to Low)":
    filtered_df = filtered_df.copy()
    filtered_df["conf_gap"] = (filtered_df["hybrid_confidence"] - filtered_df["cnn_confidence"]).abs()
    filtered_df = filtered_df.sort_values(by="conf_gap", ascending=False)

# -----------------------------------------------------------------------------
# UNIFIED METRIC STRIP (Single Bordered Container)
# -----------------------------------------------------------------------------
total_samples = len(df_moon_preds)
cnn_acc = (df_moon_preds["cnn_correct"].sum() / total_samples) * 100.0
hybrid_acc = (df_moon_preds["hybrid_correct"].sum() / total_samples) * 100.0
mlp_acc = (df_moon_preds["mlp_correct"].sum() / total_samples) * 100.0
disagree_count = df_moon_preds["has_disagreement"].sum()

with st.container(border=True):
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Test Images", f"{total_samples}", f"Filtered: {len(filtered_df)}")
    col_s2.metric("CNN Test Accuracy", f"{cnn_acc:.1f}%", f"Seed {selected_seed}")
    col_s3.metric("Hybrid QCNN Accuracy", f"{hybrid_acc:.1f}%", f"Seed {selected_seed}")
    col_s4.metric("Frozen MLP Accuracy", f"{mlp_acc:.1f}%", f"Seed {selected_seed}")
    col_s5.metric("Model Disagreements", f"{disagree_count}", f"{(disagree_count/total_samples)*100:.1f}% of test")

# -----------------------------------------------------------------------------
# QUICK-SELECT DEMO CASE BAR
# -----------------------------------------------------------------------------
section_header("Quick-Select Demo Case", "Choose a significant lunar crater classification scenario or use sidebar filters.")
selected_preset = st.radio(
    "Choose a curated demo case or use custom sidebar filters:",
    options=MOON_DEMO_PRESETS,
    index=0,
    horizontal=True,
    key="moon_demo_preset",
)

demo_subsets = get_moon_demo_presets(df_moon_preds)

if selected_preset != "Manual / Filter Selection":
    preset_matches = demo_subsets[selected_preset]
    if len(preset_matches) == 0:
        st.info(
            f"ℹ️ Category **'{selected_preset}'** has 0 matching images in Seed {selected_seed}. "
            f"Falling back to active manual filter selection."
        )
        active_df = filtered_df
    else:
        st.success(
            f"🎯 Demo Preset active: Showing **{len(preset_matches):,}** images for **{selected_preset}**."
        )
        if sort_option == "CNN Confidence (High to Low)":
            active_df = preset_matches.sort_values(by="cnn_confidence", ascending=False)
        elif sort_option == "Hybrid Confidence (High to Low)":
            active_df = preset_matches.sort_values(by="hybrid_confidence", ascending=False)
        elif sort_option == "MLP Confidence (High to Low)":
            active_df = preset_matches.sort_values(by="mlp_confidence", ascending=False)
        elif sort_option == "Hybrid vs CNN Confidence Gap (High to Low)":
            preset_matches = preset_matches.copy()
            preset_matches["conf_gap"] = (preset_matches["hybrid_confidence"] - preset_matches["cnn_confidence"]).abs()
            active_df = preset_matches.sort_values(by="conf_gap", ascending=False)
        else:
            active_df = preset_matches
else:
    active_df = filtered_df

if active_df.empty:
    st.info("ℹ️ No test images match the active filter criteria. Try adjusting your filters or selecting a demo preset.")
    st.stop()

# -----------------------------------------------------------------------------
# STEPPING CONTROLS & IMAGE SELECTOR
# -----------------------------------------------------------------------------
cfg_tracker = f"cfg_{selected_seed}_{class_filter}_{agreement_filter}_{confusion_model}_{confusion_pair}_{sort_option}_{selected_preset}"
if st.session_state.get("moon_last_cfg") != cfg_tracker:
    st.session_state["moon_last_cfg"] = cfg_tracker
    st.session_state["moon_img_idx"] = 0

current_img_idx = st.session_state.get("moon_img_idx", 0)
if current_img_idx >= len(active_df):
    current_img_idx = 0
    st.session_state["moon_img_idx"] = 0

image_options = [
    f"[{i+1}/{len(active_df)}] {r['filename']} | True:{r['true_label_name'].upper()} | CNN:{r['cnn_pred_name']} ({r['cnn_confidence']*100:.0f}%) | QCNN:{r['hybrid_pred_name']} ({r['hybrid_confidence']*100:.0f}%)"
    for i, (_, r) in enumerate(active_df.iterrows())
]

with st.container(border=True):
    col_prev, col_select, col_next, col_dl = st.columns([1.2, 5.0, 1.2, 2.6])

    with col_prev:
        if st.button("◀ Prev Image", key="moon_btn_prev", disabled=(current_img_idx <= 0), width="stretch"):
            new_idx = max(0, current_img_idx - 1)
            st.session_state["moon_img_idx"] = new_idx
            st.session_state[f"moon_img_select_{cfg_tracker}"] = new_idx
            st.rerun()

    with col_select:
        selected_idx = st.selectbox(
            "Select Image to Inspect:",
            options=range(len(active_df)),
            format_func=lambda i: image_options[i],
            index=current_img_idx,
            key=f"moon_img_select_{cfg_tracker}",
            label_visibility="collapsed",
        )
        if selected_idx != current_img_idx:
            st.session_state["moon_img_idx"] = selected_idx
            current_img_idx = selected_idx

    with col_next:
        if st.button("Next Image ▶", key="moon_btn_next", disabled=(current_img_idx >= len(active_df) - 1), width="stretch"):
            new_idx = min(len(active_df) - 1, current_img_idx + 1)
            st.session_state["moon_img_idx"] = new_idx
            st.session_state[f"moon_img_select_{cfg_tracker}"] = new_idx
            st.rerun()

    with col_dl:
        st.download_button(
            "📥 Export CSV",
            data=active_df.to_csv(index=False).encode("utf-8"),
            file_name=f"moon_test_filtered_seed{selected_seed}.csv",
            mime="text/csv",
            key="moon_download_csv",
            width="stretch",
        )

row = active_df.iloc[current_img_idx]
sample_id = row["sample_id"]

img = load_moon_image(sample_id)

# -----------------------------------------------------------------------------
# SIDE-BY-SIDE INSPECTION LAYOUT (Left: Image & GT; Right: Predictions & Softmax)
# -----------------------------------------------------------------------------
col_img_view, col_preds_view = st.columns([3.8, 8.2])

with col_img_view:
    with st.container(border=True):
        st.markdown("##### Lunar Surface Image")
        st.image(img, caption=f"227×227 RGB ({row['filename']})", width="stretch")
        
        true_class = row["true_label_name"]
        badge_type = "safe" if true_class == "ejecta" else ("disagree" if true_class == "oldcrater" else "neutral")
        
        st.markdown(
            f"""
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--mq-line);">
                <div style="font-size: 0.75rem; color: var(--mq-text-3); text-transform: uppercase; font-family: var(--mq-font-mono);">Ground Truth Target:</div>
                <div style="margin-top: 4px; margin-bottom: 8px;">{badge_html(true_class.upper(), badge_type)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        moon_img_kv = [
            ("Relative Path", str(sample_id)),
            ("Evaluation Seed", str(selected_seed)),
            ("True Label Name", str(true_class).upper()),
            ("Verification", "FILE_FACTS_CONFIRMED"),
        ]
        st.markdown(kv_table(moon_img_kv), unsafe_allow_html=True)

with col_preds_view:
    # Model ledger / cards
    with st.container(border=True):
        st.markdown("##### Multi-Model Classification Ledger")
        
        c_cnn, c_hyb, c_mlp = st.columns(3)
        
        with c_cnn:
            st.markdown(
                f"""
                <div style="background: var(--mq-surface-2); border: 1px solid var(--mq-line); border-radius: var(--mq-r); padding: 12px; text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--mq-cnn);">{model_chip('cnn')}Classical CNN</div>
                    <div style="margin: 8px 0;">{badge_html(row['cnn_pred_name'].upper(), 'safe' if row['cnn_pred_name']=='ejecta' else ('disagree' if row['cnn_pred_name']=='oldcrater' else 'neutral'))}</div>
                    <div style="font-family: var(--mq-font-mono); font-size: 1.25rem; font-weight: 600; color: var(--mq-text);">{row['cnn_confidence']*100:.1f}%</div>
                    <div style="margin-top: 6px;">{badge_html('CORRECT' if row['cnn_correct'] else 'WRONG', 'correct' if row['cnn_correct'] else 'incorrect')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        with c_hyb:
            st.markdown(
                f"""
                <div style="background: var(--mq-surface-2); border: 1px solid var(--mq-line); border-radius: var(--mq-r); padding: 12px; text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--mq-qcnn);">{model_chip('qcnn')}Hybrid QCNN</div>
                    <div style="margin: 8px 0;">{badge_html(row['hybrid_pred_name'].upper(), 'safe' if row['hybrid_pred_name']=='ejecta' else ('disagree' if row['hybrid_pred_name']=='oldcrater' else 'neutral'))}</div>
                    <div style="font-family: var(--mq-font-mono); font-size: 1.25rem; font-weight: 600; color: var(--mq-text);">{row['hybrid_confidence']*100:.1f}%</div>
                    <div style="margin-top: 6px;">{badge_html('CORRECT' if row['hybrid_correct'] else 'WRONG', 'correct' if row['hybrid_correct'] else 'incorrect')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        with c_mlp:
            st.markdown(
                f"""
                <div style="background: var(--mq-surface-2); border: 1px solid var(--mq-line); border-radius: var(--mq-r); padding: 12px; text-align: center;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--mq-mlp);">{model_chip('mlp')}Frozen MLP Control</div>
                    <div style="margin: 8px 0;">{badge_html(row['mlp_pred_name'].upper(), 'safe' if row['mlp_pred_name']=='ejecta' else ('disagree' if row['mlp_pred_name']=='oldcrater' else 'neutral'))}</div>
                    <div style="font-family: var(--mq-font-mono); font-size: 1.25rem; font-weight: 600; color: var(--mq-text);">{row['mlp_confidence']*100:.1f}%</div>
                    <div style="margin-top: 6px;">{badge_html('CORRECT' if row['mlp_correct'] else 'WRONG', 'correct' if row['mlp_correct'] else 'incorrect')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        # Softmax probability distribution chart
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Softmax Predicted Probability Distribution")
        fig_prob = go.Figure()
        
        classes = ["ejecta", "oldcrater", "none"]
        cnn_probs = [row["cnn_prob_ejecta"], row["cnn_prob_oldcrater"], row["cnn_prob_none"]]
        hyb_probs = [row["hybrid_prob_ejecta"], row["hybrid_prob_oldcrater"], row["hybrid_prob_none"]]
        mlp_probs = [row["mlp_prob_ejecta"], row["mlp_prob_oldcrater"], row["mlp_prob_none"]]
        
        fig_prob.add_trace(go.Bar(name="Classical CNN", x=classes, y=cnn_probs, marker_color=COLOR_CNN))
        fig_prob.add_trace(go.Bar(name="Hybrid QCNN", x=classes, y=hyb_probs, marker_color=COLOR_HYBRID))
        fig_prob.add_trace(go.Bar(name="Frozen MLP", x=classes, y=mlp_probs, marker_color=COLOR_MLP))
        
        fig_prob.update_layout(
            template=MARSQCNN_TEMPLATE,
            barmode="group",
            height=230,
            margin=dict(l=40, r=20, t=30, b=30),
            yaxis=dict(title="Probability", range=[0.0, 1.0]),
        )
        st.plotly_chart(fig_prob, width="stretch")

# -----------------------------------------------------------------------------
# DETAILED PROBABILITY & RAW LOGIT TABLE
# -----------------------------------------------------------------------------
section_header("Raw Logits & Derived Probabilities", "Standard softmax applied to saved unscaled test activations.")

prob_table = [
    {
        "Class": "ejecta",
        "CNN Logit": f"{row['cnn_logit_ejecta']:.4f}",
        "CNN Prob (Softmax)": f"{row['cnn_prob_ejecta']*100:.2f}%",
        "Hybrid Logit": f"{row['hybrid_logit_ejecta']:.4f}",
        "Hybrid Prob (Softmax)": f"{row['hybrid_prob_ejecta']*100:.2f}%",
        "MLP Logit": f"{row['mlp_logit_ejecta']:.4f}",
        "MLP Prob (Softmax)": f"{row['mlp_prob_ejecta']*100:.2f}%",
    },
    {
        "Class": "oldcrater",
        "CNN Logit": f"{row['cnn_logit_oldcrater']:.4f}",
        "CNN Prob (Softmax)": f"{row['cnn_prob_oldcrater']*100:.2f}%",
        "Hybrid Logit": f"{row['hybrid_logit_oldcrater']:.4f}",
        "Hybrid Prob (Softmax)": f"{row['hybrid_prob_oldcrater']*100:.2f}%",
        "MLP Logit": f"{row['mlp_logit_oldcrater']:.4f}",
        "MLP Prob (Softmax)": f"{row['mlp_prob_oldcrater']*100:.2f}%",
    },
    {
        "Class": "none",
        "CNN Logit": f"{row['cnn_logit_none']:.4f}",
        "CNN Prob (Softmax)": f"{row['cnn_prob_none']*100:.2f}%",
        "Hybrid Logit": f"{row['hybrid_logit_none']:.4f}",
        "Hybrid Prob (Softmax)": f"{row['hybrid_prob_none']*100:.2f}%",
        "MLP Logit": f"{row['mlp_logit_none']:.4f}",
        "MLP Prob (Softmax)": f"{row['mlp_prob_none']*100:.2f}%",
    },
]
st.dataframe(pd.DataFrame(prob_table), hide_index=True, width="stretch")

note_block(
    "disclosure",
    "Explainability Notice (Softmax Confidence vs Causal Explanation)",
    "The probabilities above are derived directly via standard softmax: "
    "$$P(y = c \\mid x) = \\frac{e^{z_c}}{\\sum_{k} e^{z_k}}$$"
    "These indicate predictive confidence across the 3 classes. They are <strong>not causal or pixel-level explanations</strong> "
    "and do not imply physical or geological features are causal factors."
)

render_walkthrough_footer(
    current_page_num=4,
    current_page_name="Moon Multi-Model Sample Inspection",
    next_page_path="pages/5_Model_Comparison.py",
    next_page_name="Aggregate Metrics, Error Bars & Trade-offs",
    next_page_takeaway="Analyze aggregate metrics across all 15 Mars runs and 3 Moon seeds, paired t-intervals, and capacity/timing analysis.",
    prev_page_path="pages/3_Mars_Test_Explorer.py",
    prev_page_name="Mars Sample Inspection & Low-Risk Proxy",
)
