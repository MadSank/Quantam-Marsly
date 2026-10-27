"""Milestone 2 Acceptance Tests: Overview Page Data Consistency."""

import sys
from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.mars_data import load_mars_aggregate_metrics
from data_loading.moon_data import load_moon_recomputed_metrics


def test_mars_headline_metrics_match():
    df_mars = load_mars_aggregate_metrics()
    
    # Check CNN slope RMSE
    cnn_slope = df_mars[(df_mars["model"] == "CNN") & (df_mars["metric"] == "RMSE_mean_slope_5m")]
    assert len(cnn_slope) == 1
    assert abs(cnn_slope["mean_site_metric"].values[0] - 10.250) < 0.01
    
    # Check QCNN slope RMSE
    qcnn_slope = df_mars[(df_mars["model"] == "CNN_QCNN") & (df_mars["metric"] == "RMSE_mean_slope_5m")]
    assert len(qcnn_slope) == 1
    assert abs(qcnn_slope["mean_site_metric"].values[0] - 8.190) < 0.01
    
    # Check safe recall values
    cnn_recall = df_mars[(df_mars["model"] == "CNN") & (df_mars["metric"] == "safe_recall")]
    assert abs(cnn_recall["mean_site_metric"].values[0] - 0.0556) < 0.001
    
    qcnn_recall = df_mars[(df_mars["model"] == "CNN_QCNN") & (df_mars["metric"] == "safe_recall")]
    assert abs(qcnn_recall["mean_site_metric"].values[0] - 0.0277) < 0.001


def test_moon_headline_metrics_match():
    df_moon = load_moon_recomputed_metrics()
    
    cnn_acc = df_moon[(df_moon["model"] == "cnn") & (df_moon["metric"] == "accuracy")]["mean"].values[0]
    assert abs(cnn_acc - 0.5297) < 0.001
    
    hybrid_bal_acc = df_moon[(df_moon["model"] == "hybrid_qcnn") & (df_moon["metric"] == "balanced_accuracy")]["mean"].values[0]
    assert abs(hybrid_bal_acc - 0.6080) < 0.001
    
    maj_acc = df_moon[(df_moon["model"] == "majority_train_baseline") & (df_moon["metric"] == "accuracy")]["mean"].values[0]
    assert abs(maj_acc - 0.6842) < 0.001


def test_overview_walkthrough_and_page_execution():
    """Verify that pages/1_Overview.py contains the 6-step walkthrough and executes cleanly."""
    overview_path = APP_DIR / "pages" / "1_Overview.py"
    content = overview_path.read_text(encoding="utf-8")
    assert "Coursework Demonstration Walkthrough (6-Step Presentation Guide)" in content
    assert "Step 1: Task Framing & Headline Results" in content
    assert "render_walkthrough_footer" in content
    
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(overview_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"
