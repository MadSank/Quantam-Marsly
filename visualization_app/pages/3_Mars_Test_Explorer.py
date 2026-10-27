"""Mars Test Explorer: Interactive paired CNN vs QCNN evaluation on held-out test sites (Mission Console styling)."""

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
    COLOR_QCNN,
    COLOR_SAFE,
    COLOR_UNSAFE,
    COLOR_DISAGREE,
    MARSQCNN_TEMPLATE,
)
from data_loading.mars_data import (
    load_mars_metadata,
    load_mars_predictions,
    load_mars_patch_image,
    get_mars_demo_presets,
    MARS_DEMO_PRESETS,
    FOLD_SITE_MAPPING,
    SLOPE_SAFE_THRESHOLD,
    HAZARD_SAFE_THRESHOLD,
)
from components.charts import build_mars_grid_scatter
from components.layout import page_header, note_block, section_header, kv_table, model_chip
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=3,
    title="Mars Held-Out Test Explorer",
    subtitle="Inspect paired classical CNN vs hybrid QCNN predictions on unseen, held-out geographic test sites.",
    section_name="MARS SAMPLE INSPECTION & LOW-RISK PROXY",
)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Test Run Configuration")

selected_fold = st.sidebar.selectbox(
    "Leave-One-Site-Out Fold",
    options=[0, 1, 2, 3, 4],
    format_func=lambda f: f"Fold {f} (Held-out Test Site: {FOLD_SITE_MAPPING[f]['test']})",
    key="mars_test_fold",
)

selected_seed = st.sidebar.selectbox(
    "Random Seed",
    options=[42, 43, 44],
    key="mars_test_seed",
)

test_site_name = FOLD_SITE_MAPPING[selected_fold]["test"]
st.sidebar.info(
    f"**Fold {selected_fold} Configuration:**\n"
    f"- Test Site: `{test_site_name}`\n"
    f"- Validation Site: `{FOLD_SITE_MAPPING[selected_fold]['val']}`\n"
    f"- Training Sites: `{', '.join(FOLD_SITE_MAPPING[selected_fold]['train'])}`"
)

# Load predictions
df_preds = load_mars_predictions(fold=selected_fold, seed=selected_seed)
df_meta = load_mars_metadata()

# Filters
st.sidebar.markdown("### Filter Patches")
filter_mode = st.sidebar.selectbox(
    "Filter by Status / Agreement",
    options=[
        "All Patches",
        "Models Disagree on Safe (CNN != QCNN)",
        "Both Models Correct",
        "Both Models Wrong",
        "Ground Truth Safe (True Safe = 1)",
        "Ground Truth Unsafe (True Safe = 0)",
        "CNN False Positive (CNN Safe, True Unsafe)",
        "QCNN False Positive (QCNN Safe, True Unsafe)",
        "CNN False Negative (CNN Unsafe, True Safe)",
        "QCNN False Negative (QCNN Unsafe, True Safe)",
    ],
    key="mars_test_filter",
)

sort_mode = st.sidebar.selectbox(
    "Sort By",
    options=[
        "Default (Tile ID)",
        "CNN Absolute Slope Error (High to Low)",
        "QCNN Absolute Slope Error (High to Low)",
        "Slope Difference |CNN - QCNN| (High to Low)",
        "CNN Absolute Hazard Error (High to Low)",
        "QCNN Absolute Hazard Error (High to Low)",
    ],
    key="mars_test_sort",
)

# Apply filters
filtered_df = df_preds.copy()
if filter_mode == "Models Disagree on Safe (CNN != QCNN)":
    filtered_df = filtered_df[~filtered_df["agree_safe"]]
elif filter_mode == "Both Models Correct":
    filtered_df = filtered_df[filtered_df["both_correct_safe"]]
elif filter_mode == "Both Models Wrong":
    filtered_df = filtered_df[filtered_df["both_wrong_safe"]]
elif filter_mode == "Ground Truth Safe (True Safe = 1)":
    filtered_df = filtered_df[filtered_df["true_safe"] == 1]
elif filter_mode == "Ground Truth Unsafe (True Safe = 0)":
    filtered_df = filtered_df[filtered_df["true_safe"] == 0]
elif filter_mode == "CNN False Positive (CNN Safe, True Unsafe)":
    filtered_df = filtered_df[(filtered_df["cnn_pred_safe"] == 1) & (filtered_df["true_safe"] == 0)]
elif filter_mode == "QCNN False Positive (QCNN Safe, True Unsafe)":
    filtered_df = filtered_df[(filtered_df["qcnn_pred_safe"] == 1) & (filtered_df["true_safe"] == 0)]
elif filter_mode == "CNN False Negative (CNN Unsafe, True Safe)":
    filtered_df = filtered_df[(filtered_df["cnn_pred_safe"] == 0) & (filtered_df["true_safe"] == 1)]
elif filter_mode == "QCNN False Negative (QCNN Unsafe, True Safe)":
    filtered_df = filtered_df[(filtered_df["qcnn_pred_safe"] == 0) & (filtered_df["true_safe"] == 1)]

# Apply sorting helper
def apply_mars_sorting(df_in: pd.DataFrame, sort_key: str) -> pd.DataFrame:
    df_sorted = df_in.copy()
    if sort_key == "CNN Absolute Slope Error (High to Low)":
        return df_sorted.sort_values(by="cnn_abs_err_slope", ascending=False)
    elif sort_key == "QCNN Absolute Slope Error (High to Low)":
        return df_sorted.sort_values(by="qcnn_abs_err_slope", ascending=False)
    elif sort_key == "Slope Difference |CNN - QCNN| (High to Low)":
        df_sorted["slope_diff"] = (df_sorted["cnn_pred_mean_slope_5m"] - df_sorted["qcnn_pred_mean_slope_5m"]).abs()
        return df_sorted.sort_values(by="slope_diff", ascending=False)
    elif sort_key == "CNN Absolute Hazard Error (High to Low)":
        return df_sorted.sort_values(by="cnn_abs_err_hazard", ascending=False)
    elif sort_key == "QCNN Absolute Hazard Error (High to Low)":
        return df_sorted.sort_values(by="qcnn_abs_err_hazard", ascending=False)
    return df_sorted

filtered_df = apply_mars_sorting(filtered_df, sort_mode)

# -----------------------------------------------------------------------------
# UNIFIED METRIC STRIP (Single Bordered Container)
# -----------------------------------------------------------------------------
total_test = len(df_preds)
agree_count = df_preds["agree_safe"].sum()
disagree_count = total_test - agree_count
cnn_acc = (df_preds["cnn_correct_safe"].sum() / total_test) * 100.0
qcnn_acc = (df_preds["qcnn_correct_safe"].sum() / total_test) * 100.0

with st.container(border=True):
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Held-Out Test Patches", f"{total_test:,}", f"Site: {test_site_name}")
    col_m2.metric("Filtered Patches", f"{len(filtered_df):,}", f"{(len(filtered_df)/total_test)*100:.1f}%")
    col_m3.metric("Model Agreement", f"{(agree_count/total_test)*100:.1f}%", f"{agree_count} agreed")
    col_m4.metric("CNN Safe Accuracy", f"{cnn_acc:.1f}%", f"Seed {selected_seed}")
    col_m5.metric("QCNN Safe Accuracy", f"{qcnn_acc:.1f}%", f"Seed {selected_seed}")

# -----------------------------------------------------------------------------
# QUICK-SELECT DEMO PRESET BAR
# -----------------------------------------------------------------------------
section_header("Quick-Select Demo Case", "Jump directly to significant model comparison scenarios or use custom filters.")
selected_preset = st.radio(
    "Choose a curated demo case or use custom sidebar filters:",
    options=MARS_DEMO_PRESETS,
    index=0,
    horizontal=True,
    key="mars_demo_preset",
)

demo_subsets = get_mars_demo_presets(df_preds)

if selected_preset != "Manual / Filter Selection":
    preset_matches = demo_subsets[selected_preset]
    if len(preset_matches) == 0:
        st.info(
            f"ℹ️ Category **'{selected_preset}'** has 0 matching patches in Fold {selected_fold}, Seed {selected_seed} "
            f"(e.g. QCNN made 0 positive safe predictions on safe patches in this site). "
            f"Falling back to active manual filter selection."
        )
        active_df = filtered_df
    else:
        st.success(
            f"🎯 Demo Preset active: Showing **{len(preset_matches):,}** patches for **{selected_preset}**."
        )
        active_df = apply_mars_sorting(preset_matches, sort_mode)
else:
    active_df = filtered_df

if active_df.empty:
    st.info("ℹ️ No patches match the active filter criteria. Try adjusting your filters or selecting a demo preset.")
    st.stop()

# -----------------------------------------------------------------------------
# STEPPING CONTROLS & PATCH SELECTOR
# -----------------------------------------------------------------------------
cfg_tracker = f"cfg_{selected_fold}_{selected_seed}_{filter_mode}_{sort_mode}_{selected_preset}"
if st.session_state.get("mars_last_cfg") != cfg_tracker:
    st.session_state["mars_last_cfg"] = cfg_tracker
    st.session_state["mars_patch_idx"] = 0

current_patch_idx = st.session_state.get("mars_patch_idx", 0)
if current_patch_idx >= len(active_df):
    current_patch_idx = 0
    st.session_state["mars_patch_idx"] = 0

patch_options = [
    f"[{i+1}/{len(active_df)}] {r['tile_id']} | True:{'SAFE' if r['true_safe']==1 else 'UNSAFE'} | CNN:{'SAFE' if r['cnn_pred_safe']==1 else 'UNSAFE'} | QCNN:{'SAFE' if r['qcnn_pred_safe']==1 else 'UNSAFE'}"
    for i, (_, r) in enumerate(active_df.iterrows())
]

with st.container(border=True):
    col_prev, col_select, col_next, col_dl = st.columns([1.2, 5.0, 1.2, 2.6])

    with col_prev:
        if st.button("◀ Prev Patch", key="mars_btn_prev", disabled=(current_patch_idx <= 0), width="stretch"):
            new_idx = max(0, current_patch_idx - 1)
            st.session_state["mars_patch_idx"] = new_idx
            st.session_state[f"mars_patch_select_{cfg_tracker}"] = new_idx
            st.rerun()

    with col_select:
        selected_idx = st.selectbox(
            "Select Patch to Inspect:",
            options=range(len(active_df)),
            format_func=lambda i: patch_options[i],
            index=current_patch_idx,
            key=f"mars_patch_select_{cfg_tracker}",
            label_visibility="collapsed",
        )
        if selected_idx != current_patch_idx:
            st.session_state["mars_patch_idx"] = selected_idx
            current_patch_idx = selected_idx

    with col_next:
        if st.button("Next Patch ▶", key="mars_btn_next", disabled=(current_patch_idx >= len(active_df) - 1), width="stretch"):
            new_idx = min(len(active_df) - 1, current_patch_idx + 1)
            st.session_state["mars_patch_idx"] = new_idx
            st.session_state[f"mars_patch_select_{cfg_tracker}"] = new_idx
            st.rerun()

    with col_dl:
        st.download_button(
            "📥 Export CSV",
            data=active_df.to_csv(index=False).encode("utf-8"),
            file_name=f"mars_test_fold{selected_fold}_seed{selected_seed}_filtered.csv",
            mime="text/csv",
            key="mars_download_csv",
            width="stretch",
        )

row = active_df.iloc[current_patch_idx]
tile_id = row["tile_id"]

# Look up shard information in metadata
meta_match = df_meta[df_meta["tile_id"] == tile_id]
if meta_match.empty:
    st.error(f"Metadata for tile {tile_id} could not be resolved.")
    st.stop()
meta_row = meta_match.iloc[0]

patch_img = load_mars_patch_image(meta_row["shard_path"], int(meta_row["shard_idx"]))

# -----------------------------------------------------------------------------
# SELECTED PATCH HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background: var(--mq-surface); border: 1px solid var(--mq-line); border-radius: var(--mq-r); padding: 12px 16px; margin-top: 8px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.95rem; font-weight: 600; color: var(--mq-text);">Selected Patch: <code style="color: var(--mq-rust-hi);">{tile_id}</code></span>
                <span style="font-size: 0.8rem; color: var(--mq-text-2); margin-left: 8px;">(Site: <code>{test_site_name}</code> | Image: <code>{row['image_id']}</code>)</span>
            </div>
            <div>
                <span class="badge badge-model-cnn">Fold {selected_fold}</span>
                <span class="badge badge-model-qcnn">Seed {selected_seed}</span>
                <span class="badge badge-neutral">Grid: R{meta_row['row']} / C{meta_row['col']}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDE-BY-SIDE INSPECTION LAYOUT (Left: Image & GT; Right: Tabs for Predictions, Scatter, Attribution)
# -----------------------------------------------------------------------------
col_img, col_tabs = st.columns([3.8, 8.2])

with col_img:
    with st.container(border=True):
        st.markdown("##### Test Patch Orthoimage")
        st.image(patch_img, caption=f"64×64 @ 2m/px ({tile_id})", width="stretch", clamp=True)
        
        true_safe_str = "SAFE PROXY" if row["true_safe"] == 1 else "UNSAFE PROXY"
        true_badge = badge_html(true_safe_str, "safe" if row["true_safe"] == 1 else "unsafe")
        
        st.markdown(
            f"""
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--mq-line);">
                <div style="font-size: 0.75rem; color: var(--mq-text-3); text-transform: uppercase; font-family: var(--mq-font-mono);">Ground Truth Status:</div>
                <div style="margin-top: 4px; margin-bottom: 8px;">{true_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        gt_kv = [
            ("Mean Slope (5m)", f"{row['true_mean_slope_5m']:.2f}°"),
            ("Roughness", f"{row['true_roughness_m']:.3f} m"),
            ("Hazard Fraction", f"{row['true_hazard_fraction']:.4f}"),
            ("Intra-Site Grid", f"R{meta_row['row']}, C{meta_row['col']}"),
            ("Valid Coverage", f"{meta_row['valid_coverage']*100:.1f}%"),
        ]
        st.markdown(kv_table(gt_kv), unsafe_allow_html=True)
        
        note_block(
            "warning",
            "Research Proxy Notice",
            "The rule <code>slope &lt; 8.0° & hazard &lt; 0.05</code> is a simplified terrain feasibility proxy, not a validated landing certification."
        )

with col_tabs:
    t_preds, t_grid, t_rule = st.tabs([
        "📊 Paired Predictions & Errors",
        "🗺️ Intra-Site Mosaic Grid (2D Coordinates)",
        "🎯 Deterministic Decision Rule Attribution",
    ])
    
    with t_preds:
        c_cnn, c_qcnn = st.columns(2)
        
        cnn_safe_badge = badge_html("SAFE", "safe") if row["cnn_pred_safe"] == 1 else badge_html("UNSAFE", "unsafe")
        cnn_correct_badge = badge_html("CORRECT", "correct") if row["cnn_correct_safe"] else badge_html("WRONG", "incorrect")
        
        qcnn_safe_badge = badge_html("SAFE", "safe") if row["qcnn_pred_safe"] == 1 else badge_html("UNSAFE", "unsafe")
        qcnn_correct_badge = badge_html("CORRECT", "correct") if row["qcnn_correct_safe"] else badge_html("WRONG", "incorrect")
        
        with c_cnn:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>{model_chip('cnn')}<strong style="color: var(--mq-cnn); font-size: 0.95rem;">Classical ResNet-18</strong></div>
                        <div>{cnn_safe_badge} {cnn_correct_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cnn_kv = [
                    ("Mean Slope (5m)", f"{row['cnn_pred_mean_slope_5m']:.2f}° (Δ: {row['cnn_signed_err_slope']:+.2f}°)"),
                    ("Roughness", f"{row['cnn_pred_roughness_m']:.3f} m (Δ: {row['cnn_signed_err_roughness']:+.3f})"),
                    ("Hazard Fraction", f"{row['cnn_pred_hazard_fraction']:.4f} (Δ: {row['cnn_signed_err_hazard']:+.4f})"),
                    ("Safe Proxy", "SAFE (1)" if row["cnn_pred_safe"] == 1 else "UNSAFE (0)"),
                ]
                st.markdown(kv_table(cnn_kv), unsafe_allow_html=True)
                
        with c_qcnn:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>{model_chip('qcnn')}<strong style="color: var(--mq-qcnn); font-size: 0.95rem;">Hybrid QCNN Head</strong></div>
                        <div>{qcnn_safe_badge} {qcnn_correct_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                qcnn_kv = [
                    ("Mean Slope (5m)", f"{row['qcnn_pred_mean_slope_5m']:.2f}° (Δ: {row['qcnn_signed_err_slope']:+.2f}°)"),
                    ("Roughness", f"{row['qcnn_pred_roughness_m']:.3f} m (Δ: {row['qcnn_signed_err_roughness']:+.3f})"),
                    ("Hazard Fraction", f"{row['qcnn_pred_hazard_fraction']:.4f} (Δ: {row['qcnn_signed_err_hazard']:+.4f})"),
                    ("Safe Proxy", "SAFE (1)" if row["qcnn_pred_safe"] == 1 else "UNSAFE (0)"),
                ]
                st.markdown(kv_table(qcnn_kv), unsafe_allow_html=True)
                
        # Error comparison bar chart
        fig_err = go.Figure()
        targets = ["Mean Slope (5m) [°]", "Roughness [m]", "Hazard Fraction [fraction]"]
        cnn_errs = [row["cnn_abs_err_slope"], row["cnn_abs_err_roughness"], row["cnn_abs_err_hazard"]]
        qcnn_errs = [row["qcnn_abs_err_slope"], row["qcnn_abs_err_roughness"], row["qcnn_abs_err_hazard"]]
        
        fig_err.add_trace(go.Bar(name="CNN Absolute Error (|Δ|)", x=targets, y=cnn_errs, marker_color=COLOR_CNN))
        fig_err.add_trace(go.Bar(name="QCNN Absolute Error (|Δ|)", x=targets, y=qcnn_errs, marker_color=COLOR_QCNN))
        fig_err.update_layout(
            template=MARSQCNN_TEMPLATE,
            title=dict(text="Patch Absolute Errors (|Δ| = |pred - true|) vs Ground Truth"),
            barmode="group",
            height=230,
            margin=dict(l=40, r=20, t=40, b=30),
            yaxis=dict(title="Absolute Error |Δ|"),
        )
        st.plotly_chart(fig_err, width="stretch")
        
    with t_grid:
        col_grid_ctrl, col_grid_stat = st.columns([1.2, 1.8])
        with col_grid_ctrl:
            grid_color_mode = st.selectbox(
                "Color Grid Tiles By:",
                ["Ground Truth Safe Proxy", "Model Agreement", "CNN vs QCNN Absolute Slope Error"],
                key="mars_grid_color_mode",
            )
        with col_grid_stat:
            st.caption(
                f"**Site:** `{test_site_name}` | **Total Tiles:** {len(df_preds):,} | "
                f"**Selected Tile:** `{tile_id}` (Col {meta_row['col']}, Row {meta_row['row']})"
            )
            
        fig_grid = build_mars_grid_scatter(
            df_preds,
            color_by=grid_color_mode,
            selected_tile_id=tile_id,
        )
        st.plotly_chart(fig_grid, width="stretch")
        st.caption("Unprojected local HiRISE mosaic grid coordinates (`row`, `col` from `metadata.parquet`), not global Martian latitude/longitude.")
        
    with t_rule:
        st.markdown(
            """
            The landing safety proxy is determined via a deterministic bivariate rule:
            $$\\text{Safe Proxy} = (\\text{Mean Slope} < 8.0^\\circ) \\land (\\text{Hazard Fraction} < 0.05)$$
            """
        )
        
        cnn_slope_pass = row["cnn_pred_mean_slope_5m"] < SLOPE_SAFE_THRESHOLD
        cnn_haz_pass = row["cnn_pred_hazard_fraction"] < HAZARD_SAFE_THRESHOLD
        qcnn_slope_pass = row["qcnn_pred_mean_slope_5m"] < SLOPE_SAFE_THRESHOLD
        qcnn_haz_pass = row["qcnn_pred_hazard_fraction"] < HAZARD_SAFE_THRESHOLD
        true_slope_pass = row["true_mean_slope_5m"] < SLOPE_SAFE_THRESHOLD
        true_haz_pass = row["true_hazard_fraction"] < HAZARD_SAFE_THRESHOLD
        
        decision_table = [
            {
                "Decision Criterion": "Mean Slope (5m)",
                "Safety Threshold": f"< {SLOPE_SAFE_THRESHOLD}°",
                "CNN Prediction": f"{row['cnn_pred_mean_slope_5m']:.2f}°",
                "CNN Pass?": "✅ PASS" if cnn_slope_pass else "❌ FAIL",
                "QCNN Prediction": f"{row['qcnn_pred_mean_slope_5m']:.2f}°",
                "QCNN Pass?": "✅ PASS" if qcnn_slope_pass else "❌ FAIL",
                "Ground Truth Target": f"{row['true_mean_slope_5m']:.2f}°",
                "Ground Truth Pass?": "✅ PASS" if true_slope_pass else "❌ FAIL",
            },
            {
                "Decision Criterion": "Hazard Fraction",
                "Safety Threshold": f"< {HAZARD_SAFE_THRESHOLD}",
                "CNN Prediction": f"{row['cnn_pred_hazard_fraction']:.4f}",
                "CNN Pass?": "✅ PASS" if cnn_haz_pass else "❌ FAIL",
                "QCNN Prediction": f"{row['qcnn_pred_hazard_fraction']:.4f}",
                "QCNN Pass?": "✅ PASS" if qcnn_haz_pass else "❌ FAIL",
                "Ground Truth Target": f"{row['true_hazard_fraction']:.4f}",
                "Ground Truth Pass?": "✅ PASS" if true_haz_pass else "❌ FAIL",
            },
            {
                "Decision Criterion": "Combined Safety Outcome",
                "Safety Threshold": "Both Pass",
                "CNN Prediction": "SAFE" if row["cnn_pred_safe"] == 1 else "UNSAFE",
                "CNN Pass?": "🟢 SAFE" if row["cnn_pred_safe"] == 1 else "🔴 UNSAFE",
                "QCNN Prediction": "SAFE" if row["qcnn_pred_safe"] == 1 else "UNSAFE",
                "QCNN Pass?": "🟢 SAFE" if row["qcnn_pred_safe"] == 1 else "🔴 UNSAFE",
                "Ground Truth Target": "SAFE" if row["true_safe"] == 1 else "UNSAFE",
                "Ground Truth Pass?": "🟢 SAFE" if row["true_safe"] == 1 else "🔴 UNSAFE",
            },
        ]
        st.dataframe(pd.DataFrame(decision_table), hide_index=True, width="stretch")
        
        note_block(
            "disclosure",
            "Explainability Notice (Numerical Rule vs Pixel Attribution)",
            "The decision breakdown above provides full mathematical attribution for the final classification outcome by showing "
            "how the continuous regressed physical parameters directly trigger the decision threshold. "
            "Neither the variational quantum circuit nor the frozen representation includes pixel-level gradient maps in saved artifacts; "
            "we do not synthesize artificial gradient maps for the QCNN to maintain research integrity."
        )

render_walkthrough_footer(
    current_page_num=3,
    current_page_name="Mars Sample Inspection & Low-Risk Proxy",
    next_page_path="pages/4_Moon_Test_Explorer.py",
    next_page_name="Moon Multi-Model Sample Inspection",
    next_page_takeaway="Inspect multi-model crater classification on the Moon dataset, confidence distributions, and confusion pairs across seeds.",
    prev_page_path="pages/2_Training_Data_Explorer.py",
    prev_page_name="Dataset & Ground Truth",
)
