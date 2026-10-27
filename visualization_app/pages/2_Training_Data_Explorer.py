"""Training Data Explorer: Browse Mars HiRISE terrain patches and Moon crater images with verified metadata (Mission Console styling)."""

import sys
from pathlib import Path

# Add visualization_app to path
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from style.theme import apply_theme, badge_html, COLOR_SAFE, COLOR_UNSAFE, COLOR_RUST, COLOR_BORDER
from data_loading.mars_data import (
    load_mars_metadata,
    load_mars_patch_image,
    FOLD_SITE_MAPPING,
    SLOPE_SAFE_THRESHOLD,
    HAZARD_SAFE_THRESHOLD,
)
from data_loading.moon_data import (
    load_moon_provenance,
    load_moon_image,
    CLASS_NAMES,
)
from components.filters import render_pagination_controls
from components.layout import page_header, note_block, section_header, kv_table
from components.navigation import render_walkthrough_footer

apply_theme()

# Standardized Mission Console Page Header
page_header(
    step=2,
    title="Training & Dataset Explorer",
    subtitle="Inspect genuine training and evaluation images, verified terrain targets, and data provenance.",
    section_name="DATASET & GROUND TRUTH",
)

tab_mars, tab_moon = st.tabs(["🔴 Mars HiRISE Patches (15,233)", "🌕 Moon LROCNet Images (5,000)"])

# ==============================================================================
# TAB 1: MARS HIRISE PATCHES
# ==============================================================================
with tab_mars:
    df_mars = load_mars_metadata()
    
    st.sidebar.markdown("### Mars Explorer Filters")
    selected_site = st.sidebar.selectbox("Site Filter", ["All"] + sorted(df_mars["site_id"].unique().tolist()), key="mars_site")
    
    fold_options = ["None (All Data)"] + [f"Fold {f}" for f in range(5)]
    selected_fold_str = st.sidebar.selectbox("Perspective Fold", fold_options, index=1, key="mars_fold")
    
    selected_safe = st.sidebar.selectbox("Safe Proxy Label", ["All", "Safe Only (1)", "Unsafe Only (0)"], key="mars_safe")
    
    # Filter dataset
    filtered_mars = df_mars.copy()
    if selected_site != "All":
        filtered_mars = filtered_mars[filtered_mars["site_id"] == selected_site]
        
    if selected_safe == "Safe Only (1)":
        filtered_mars = filtered_mars[filtered_mars["safe_label"] == 1]
    elif selected_safe == "Unsafe Only (0)":
        filtered_mars = filtered_mars[filtered_mars["safe_label"] == 0]
        
    # Perspective Fold logic
    if selected_fold_str != "None (All Data)":
        fold_idx = int(selected_fold_str.split()[1])
        mapping = FOLD_SITE_MAPPING[fold_idx]
        test_site = mapping["test"]
        val_site = mapping["val"]
        train_sites = mapping["train"]
        
        st.info(
            f"**Perspective Fold {fold_idx}:** "
            f"Test Site = **{test_site}** | "
            f"Validation Site = **{val_site}** | "
            f"Training Sites = **{', '.join(train_sites)}**"
        )
        
        split_role_filter = st.sidebar.selectbox("Split Role in Fold", ["All", "Train", "Validation", "Test"], key="mars_role")
        if split_role_filter == "Train":
            filtered_mars = filtered_mars[filtered_mars["site_id"].isin(train_sites)]
        elif split_role_filter == "Validation":
            filtered_mars = filtered_mars[filtered_mars["site_id"] == val_site]
        elif split_role_filter == "Test":
            filtered_mars = filtered_mars[filtered_mars["site_id"] == test_site]
            
    st.markdown(f"<span style='font-family: var(--mq-font-mono); font-size: 0.85rem; color: var(--mq-text-2);'>Matched <strong>{len(filtered_mars):,}</strong> patches out of 15,233.</span>", unsafe_allow_html=True)
    
    if filtered_mars.empty:
        st.warning("No patches match the selected criteria.")
    else:
        # State tracking for selected patch
        if "selected_mars_idx" not in st.session_state:
            st.session_state["selected_mars_idx"] = 0
            
        # Pagination
        page_mars = st.session_state.get("mars_page_num", 0)
        page_mars = render_pagination_controls(len(filtered_mars), items_per_page=12, current_page=page_mars, key_prefix="mars")
        st.session_state["mars_page_num"] = page_mars
        
        start_idx = page_mars * 12
        page_rows = filtered_mars.iloc[start_idx : start_idx + 12]
        
        col_grid, col_detail = st.columns([7, 5])
        
        with col_grid:
            section_header("Patch Grid (Click to Inspect)", "Click any tile to load full sensor and DTM metadata.")
            # 3x4 grid
            grid_cols = st.columns(3)
            inspected_row = st.session_state.get("inspected_mars_row", page_rows.iloc[0].to_dict())
            active_tile_id = inspected_row.get("tile_id", "")
            
            for i, (_, row) in enumerate(page_rows.iterrows()):
                c_idx = i % 3
                tile_id = row["tile_id"]
                shard_path = row["shard_path"]
                shard_idx = int(row["shard_idx"])
                
                img = load_mars_patch_image(shard_path, shard_idx)
                is_safe = row["safe_label"] == 1
                slope = row["mean_slope_5m"]
                hazard = row["hazard_fraction"]
                is_selected = (tile_id == active_tile_id)
                border_color = "var(--mq-rust)" if is_selected else "var(--mq-line)"
                
                with grid_cols[c_idx]:
                    st.markdown(
                        f"""
                        <div style="background: var(--mq-surface); border: 1px solid {border_color}; border-radius: var(--mq-r); padding: 8px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-size: 0.72rem; font-family: var(--mq-font-mono); color: var(--mq-text-3);">{row['site_id']}</span>
                                {badge_html("SAFE" if is_safe else "UNSAFE", "safe" if is_safe else "unsafe")}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(img, width="stretch", clamp=True)
                    st.caption(f"Slope: **{slope:.1f}°** | Haz: **{hazard:.3f}**")
                    if st.button("Inspect Patch", key=f"btn_mars_{tile_id}", width="stretch"):
                        st.session_state["inspected_mars_row"] = row.to_dict()
                        st.rerun()
                        
        with col_detail:
            section_header("Patch Inspection Details", "Continuous physical features and ground truth labels.")
            with st.container(border=True):
                img_detail = load_mars_patch_image(inspected_row["shard_path"], int(inspected_row["shard_idx"]))
                col_img_sub, col_meta_sub = st.columns([1, 1])
                with col_img_sub:
                    st.image(img_detail, caption=f"Tile: {inspected_row['tile_id'][:14]}...", width="stretch", clamp=True)
                with col_meta_sub:
                    is_safe = inspected_row["safe_label"] == 1
                    status_badge = badge_html("SAFE PROXY", "safe") if is_safe else badge_html("UNSAFE PROXY", "unsafe")
                    st.markdown(
                        f"""
                        <div style="font-size: 0.8rem; color: var(--mq-text-3); margin-top: 4px;">GROUND TRUTH STATUS:</div>
                        <div style="margin-top: 4px; margin-bottom: 12px;">{status_badge}</div>
                        <div style="font-size: 0.8rem; font-family: var(--mq-font-mono); color: var(--mq-text-2);">
                            Site: <strong>{inspected_row['site_id']}</strong><br/>
                            Grid: <strong>R{inspected_row['row']} / C{inspected_row['col']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # Criteria check table
                slope_val = inspected_row["mean_slope_5m"]
                hazard_val = inspected_row["hazard_fraction"]
                roughness_val = inspected_row["roughness_m"]
                
                slope_pass = slope_val < SLOPE_SAFE_THRESHOLD
                hazard_pass = hazard_val < HAZARD_SAFE_THRESHOLD
                
                criteria_df = pd.DataFrame([
                    {
                        "Criterion": "Mean Slope (5m)",
                        "Threshold": f"< {SLOPE_SAFE_THRESHOLD}°",
                        "Target Value": f"{slope_val:.2f}°",
                        "Pass Rule?": "✅ PASS" if slope_pass else "❌ FAIL",
                    },
                    {
                        "Criterion": "Hazard Fraction",
                        "Threshold": f"< {HAZARD_SAFE_THRESHOLD}",
                        "Target Value": f"{hazard_val:.4f}",
                        "Pass Rule?": "✅ PASS" if hazard_pass else "❌ FAIL",
                    },
                    {
                        "Criterion": "Roughness",
                        "Threshold": "Continuous regressor",
                        "Target Value": f"{roughness_val:.3f} m",
                        "Pass Rule?": "N/A (Continuous)",
                    },
                ])
                st.dataframe(criteria_df, hide_index=True, width="stretch")
                
                with st.expander("HiRISE Sensor & Geospatial Metadata", expanded=True):
                    sensor_meta = [
                        ("Site ID", str(inspected_row["site_id"])),
                        ("HiRISE Image ID", str(inspected_row["image_id"])),
                        ("Grid Row / Col", f"Row {inspected_row['row']}, Col {inspected_row['col']}"),
                        ("Valid Coverage", f"{inspected_row['valid_coverage'] * 100:.1f}%"),
                        ("DTM Valid Coverage", f"{inspected_row['dtm_valid_coverage'] * 100:.1f}%"),
                        ("Ortho Product ID", str(inspected_row.get("ortho_product_id", "N/A"))),
                        ("DTM Product ID", str(inspected_row.get("dtm_product_id", "N/A"))),
                    ]
                    st.markdown(kv_table(sensor_meta), unsafe_allow_html=True)

# ==============================================================================
# TAB 2: MOON CRATER IMAGES
# ==============================================================================
with tab_moon:
    df_prov = load_moon_provenance()
    
    st.sidebar.markdown("### Moon Explorer Filters")
    selected_split = st.sidebar.selectbox("Dataset Split", ["All", "train", "val", "test"], index=0, key="moon_split")
    selected_class = st.sidebar.selectbox("Class Filter", ["All"] + CLASS_NAMES, index=0, key="moon_class")
    
    filtered_moon = df_prov.copy()
    if selected_split != "All":
        filtered_moon = filtered_moon[filtered_moon["existing_split"] == selected_split]
    if selected_class != "All":
        filtered_moon = filtered_moon[filtered_moon["class_label"] == selected_class]
        
    st.markdown(f"<span style='font-family: var(--mq-font-mono); font-size: 0.85rem; color: var(--mq-text-2);'>Matched <strong>{len(filtered_moon):,}</strong> images out of 5,000.</span>", unsafe_allow_html=True)
    
    if filtered_moon.empty:
        st.warning("No images match the selected criteria.")
    else:
        page_moon = st.session_state.get("moon_page_num", 0)
        page_moon = render_pagination_controls(len(filtered_moon), items_per_page=12, current_page=page_moon, key_prefix="moon")
        st.session_state["moon_page_num"] = page_moon
        
        start_idx = page_moon * 12
        page_moon_rows = filtered_moon.iloc[start_idx : start_idx + 12]
        
        col_grid_m, col_detail_m = st.columns([7, 5])
        
        inspected_m_row = st.session_state.get("inspected_moon_row", page_moon_rows.iloc[0].to_dict())
        active_sample_path = inspected_m_row.get("relative_image_path", "")
        
        with col_grid_m:
            section_header("Crater Image Grid", "Click any lunar surface image to inspect metadata.")
            grid_cols_m = st.columns(3)
            for i, (_, row) in enumerate(page_moon_rows.iterrows()):
                c_idx = i % 3
                sample_rel_path = row["relative_image_path"]
                class_label = row["class_label"]
                split_val = row["existing_split"]
                
                img_moon = load_moon_image(sample_rel_path)
                badge_type = "safe" if class_label == "ejecta" else ("disagree" if class_label == "oldcrater" else "neutral")
                is_selected = (sample_rel_path == active_sample_path)
                border_color = "var(--mq-rust)" if is_selected else "var(--mq-line)"
                
                with grid_cols_m[c_idx]:
                    st.markdown(
                        f"""
                        <div style="background: var(--mq-surface); border: 1px solid {border_color}; border-radius: var(--mq-r); padding: 8px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-size: 0.72rem; font-family: var(--mq-font-mono); color: var(--mq-text-3);">{row['relative_image_path'].split('/')[-1][:12]}..</span>
                                {badge_html(class_label.upper(), badge_type)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(img_moon, width="stretch")
                    st.caption(f"Split: **{split_val}** | Mode: **{row['color_mode']}**")
                    if st.button("Inspect Image", key=f"btn_moon_{row['relative_image_path']}", width="stretch"):
                        st.session_state["inspected_moon_row"] = row.to_dict()
                        st.rerun()
                        
        with col_detail_m:
            section_header("Image Provenance Details", "LROC surface properties and SHA-256 integrity.")
            with st.container(border=True):
                img_detail_m = load_moon_image(inspected_m_row["relative_image_path"])
                st.image(img_detail_m, caption=inspected_m_row["relative_image_path"], width="stretch")
                
                class_badge = badge_html(inspected_m_row['class_label'].upper(), "safe" if inspected_m_row['class_label'] == "ejecta" else "disagree")
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; margin-bottom: 8px;">
                        <div>Class: {class_badge}</div>
                        <div style="font-family: var(--mq-font-mono); font-size: 0.8rem; color: var(--mq-text-2);">Split: <strong>{inspected_m_row['existing_split']}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                note_block(
                    "warning",
                    "Provenance Status: FILE_FACTS_CONFIRMED_SOURCE_ID_UNVERIFIED",
                    "Source observation IDs and geographic sites are unverified in this dataset release. "
                    "The filename prefix is candidate only and cannot guarantee geographic independence."
                )
                
                moon_kv = [
                    ("Relative Path", str(inspected_m_row["relative_image_path"])),
                    ("Dimensions", f"{inspected_m_row['width_px']} × {inspected_m_row['height_px']} px"),
                    ("Color Mode", str(inspected_m_row["color_mode"])),
                    ("SHA-256 Hash", f"{inspected_m_row['sha256'][:24]}..."),
                    ("Source Obs ID", "Unverified (None recorded)"),
                    ("Site ID", "Unverified (None recorded)"),
                    ("Candidate Prefix", str(inspected_m_row.get("filename_prefix_candidate_unverified", "N/A"))),
                ]
                st.markdown(kv_table(moon_kv), unsafe_allow_html=True)

render_walkthrough_footer(
    current_page_num=2,
    current_page_name="Dataset & Ground Truth",
    next_page_path="pages/3_Mars_Test_Explorer.py",
    next_page_name="Mars Sample Inspection & Low-Risk Proxy",
    next_page_takeaway="Analyze held-out Mars terrain test patches, continuous regression predictions, model disagreements, and safe proxy classification.",
    prev_page_path="pages/1_Overview.py",
    prev_page_name="Task & Architecture Overview",
)
