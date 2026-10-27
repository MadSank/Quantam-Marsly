"""Milestone 4 Acceptance Tests: Mars Test Explorer Logic and Filtering."""

import sys
from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.mars_data import (
    load_mars_predictions,
    SLOPE_SAFE_THRESHOLD,
    HAZARD_SAFE_THRESHOLD,
    get_mars_demo_presets,
    MARS_DEMO_PRESETS,
)


def test_mars_fold0_seed42_predictions():
    df = load_mars_predictions(fold=0, seed=42)
    assert len(df) == 2195
    assert df["test_site"].iloc[0] == "site_01"


def test_mars_disagreement_filtering():
    df = load_mars_predictions(fold=0, seed=42)
    disagree_df = df[~df["agree_safe"]]
    assert len(disagree_df) > 0
    # Every row in disagree_df must have different cnn_pred_safe and qcnn_pred_safe
    assert (disagree_df["cnn_pred_safe"] != disagree_df["qcnn_pred_safe"]).all()


def test_mars_both_wrong_filtering():
    df = load_mars_predictions(fold=0, seed=42)
    both_wrong = df[df["both_wrong_safe"]]
    assert len(both_wrong) > 0
    assert (both_wrong["cnn_correct_safe"] == False).all()
    assert (both_wrong["qcnn_correct_safe"] == False).all()


def test_mars_threshold_consistency():
    df = load_mars_predictions(fold=0, seed=42)
    
    # Check that pred_safe accurately reflects (pred_slope < 8.0) & (pred_hazard < 0.05)
    expected_cnn_safe = (df["cnn_pred_mean_slope_5m"] < SLOPE_SAFE_THRESHOLD) & (
        df["cnn_pred_hazard_fraction"] < HAZARD_SAFE_THRESHOLD
    )
    assert (df["cnn_pred_safe"] == expected_cnn_safe.astype(int)).all()
    
    expected_qcnn_safe = (df["qcnn_pred_mean_slope_5m"] < SLOPE_SAFE_THRESHOLD) & (
        df["qcnn_pred_hazard_fraction"] < HAZARD_SAFE_THRESHOLD
    )
    assert (df["qcnn_pred_safe"] == expected_qcnn_safe.astype(int)).all()


def test_mars_error_sorting():
    df = load_mars_predictions(fold=0, seed=42)
    sorted_df = df.sort_values(by="cnn_abs_err_slope", ascending=False)
    assert sorted_df["cnn_abs_err_slope"].iloc[0] >= sorted_df["cnn_abs_err_slope"].iloc[-1]


def test_mars_demo_presets():
    """Verify that Mars demo presets return valid dataframes and obey their filtering rules."""
    df_f0 = load_mars_predictions(fold=0, seed=42)
    presets_f0 = get_mars_demo_presets(df_f0)

    # Check all presets present
    for preset_name in MARS_DEMO_PRESETS:
        assert preset_name in presets_f0
        assert isinstance(presets_f0[preset_name], pd.DataFrame)

    # In Fold 0, seed 42:
    # 1. Model Disagreement
    disagree = presets_f0["Model Disagreement (CNN != QCNN)"]
    assert len(disagree) > 0
    assert (disagree["cnn_pred_safe"] != disagree["qcnn_pred_safe"]).all()

    # 2. CNN False Positive (Unsafe predicted Safe)
    cnn_fp = presets_f0["CNN False Positive (Unsafe predicted Safe)"]
    assert len(cnn_fp) > 0
    assert ((cnn_fp["cnn_pred_safe"] == 1) & (cnn_fp["true_safe"] == 0)).all()

    # 3. QCNN False Negative (Safe predicted Unsafe)
    qcnn_fn = presets_f0["QCNN False Negative (Safe predicted Unsafe)"]
    assert len(qcnn_fn) > 0
    assert ((qcnn_fn["qcnn_pred_safe"] == 0) & (qcnn_fn["true_safe"] == 1)).all()

    # 4. Both models agree safe: handles 0-match across all runs gracefully
    safe_tp_f0 = presets_f0["Both Models Agree Safe (True Positive)"]
    assert isinstance(safe_tp_f0, pd.DataFrame)
    assert len(safe_tp_f0) == 0  # In real experiment runs, QCNN and CNN never simultaneously predicted true safe at fixed threshold

    # Verify preset behavior on another fold/seed (Fold 1, seed 43)
    df_f1 = load_mars_predictions(fold=1, seed=43)
    presets_f1 = get_mars_demo_presets(df_f1)
    
    # Model Disagreement is non-empty in Fold 1
    assert len(presets_f1["Model Disagreement (CNN != QCNN)"]) > 0
    # QCNN False Negative is non-empty in Fold 1
    assert len(presets_f1["QCNN False Negative (Safe predicted Unsafe)"]) > 0


def test_mars_test_explorer_page_execution():
    """Verify that pages/3_Mars_Test_Explorer.py compiles and runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_Mars_Test_Explorer.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"


def test_mars_grid_scatter_builder():
    """Verify intra-site grid scatter plot builds valid figure with accurate row/col bounds and honest axis labels."""
    from components.charts import build_mars_grid_scatter
    import plotly.graph_objects as go
    
    df_f0 = load_mars_predictions(fold=0, seed=42)
    selected_tile = df_f0.iloc[0]["tile_id"]
    
    fig = build_mars_grid_scatter(df_f0, color_by="Ground Truth Safe Proxy", selected_tile_id=selected_tile)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Traces for data + star marker for selected tile
    
    # Check honest axis labels
    assert fig.layout.xaxis.title.text == "Intra-Site Grid Col (px/tile offset)"
    assert fig.layout.yaxis.title.text == "Intra-Site Grid Row (px/tile offset)"
    assert fig.layout.yaxis.autorange == "reversed"
    
    # Check coordinate bounds
    min_col, max_col = df_f0["col"].min(), df_f0["col"].max()
    min_row, max_row = df_f0["row"].min(), df_f0["row"].max()
    
    for trace in fig.data:
        if trace.x is not None and len(trace.x) > 0:
            assert min_col <= min(trace.x) and max(trace.x) <= max_col
        if trace.y is not None and len(trace.y) > 0:
            assert min_row <= min(trace.y) and max(trace.y) <= max_row
            
    # Test alternative color modes
    fig_agree = build_mars_grid_scatter(df_f0, color_by="Model Agreement", selected_tile_id=selected_tile)
    assert len(fig_agree.data) >= 2
    
    fig_err = build_mars_grid_scatter(df_f0, color_by="CNN vs QCNN Absolute Slope Error", selected_tile_id=selected_tile)
    assert len(fig_err.data) >= 1


