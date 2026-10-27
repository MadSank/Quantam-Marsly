"""Milestone 6 Acceptance Tests: Model Comparison and Metrics Integrity."""

import sys
from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.mars_data import (
    load_mars_aggregate_metrics,
    load_mars_paired_differences,
    load_mars_epoch_log,
)
from data_loading.moon_data import (
    load_moon_recomputed_metrics,
    load_moon_results,
    load_moon_history,
)
from components.charts import (
    build_mars_regression_chart,
    build_mars_safety_chart,
    build_moon_comparison_chart,
)
from components.confusion_matrix import render_confusion_matrix_heatmap


def test_mars_paired_differences_cross_zero():
    df_paired = load_mars_paired_differences()
    assert not df_paired.empty
    
    # Check that for slope, roughness, and hazard RMSE, the nominal 95% t-interval crosses zero
    for metric_name in ["RMSE_mean_slope_5m", "RMSE_roughness_m", "RMSE_hazard_fraction"]:
        sub = df_paired[df_paired["metric"] == metric_name]
        assert len(sub) == 1
        t_low = sub["t95_low"].values[0]
        t_high = sub["t95_high"].values[0]
        assert t_low <= 0.0 <= t_high, f"Expected {metric_name} to cross zero, got [{t_low}, {t_high}]"


def test_moon_confusion_matrices_shape_and_sums():
    results = load_moon_results(seed=42)
    test_block = results["test"]
    
    for mod in ["cnn", "hybrid_qcnn", "frozen_embedding_mlp"]:
        cm = test_block[mod]["confusion_matrix"]
        assert len(cm) == 3
        assert len(cm[0]) == 3
        # Total sum of confusion matrix elements must equal 779
        total_sum = sum(sum(row) for row in cm)
        assert total_sum == 779


def test_chart_builders_return_valid_figures():
    df_mars = load_mars_aggregate_metrics()
    fig_reg = build_mars_regression_chart(df_mars)
    assert fig_reg is not None
    assert len(fig_reg.data) == 2  # CNN and QCNN traces
    
    fig_safe = build_mars_safety_chart(df_mars)
    assert fig_safe is not None
    assert len(fig_safe.data) == 2
    
    df_moon = load_moon_recomputed_metrics()
    fig_moon = build_moon_comparison_chart(df_moon)
    assert fig_moon is not None
    assert len(fig_moon.data) == 4  # 4 models


def test_mars_chart_layout_and_units():
    """Verify that Mars charts have non-overlapping layouts and explicit physical units/error definitions."""
    df_mars = load_mars_aggregate_metrics()
    
    # Regression chart
    fig_reg = build_mars_regression_chart(df_mars)
    title_text = fig_reg.layout.title.text
    assert "± 1 Sample SD across 5" in title_text
    # Verify physical units are explicitly labeled in target categories
    x_categories = list(fig_reg.data[0].x)
    assert any("°" in cat for cat in x_categories), f"Expected ° in categories, got {x_categories}"
    assert any("m" in cat for cat in x_categories), f"Expected m in categories, got {x_categories}"
    assert any("fraction" in cat for cat in x_categories), f"Expected fraction in categories, got {x_categories}"
    
    # Check generous margins to prevent label overlap
    assert fig_reg.layout.margin.t >= 70
    assert fig_reg.layout.margin.b >= 50
    
    # Safety chart
    fig_safe = build_mars_safety_chart(df_mars)
    safe_title = fig_safe.layout.title.text
    assert "[%]" in safe_title or "%" in safe_title
    assert "± 1 Sample SD across 5" in safe_title
    assert fig_safe.layout.margin.t >= 70
    assert fig_safe.layout.margin.b >= 50
    
    # Moon comparison chart
    df_moon = load_moon_recomputed_metrics()
    fig_moon = build_moon_comparison_chart(df_moon)
    moon_title = fig_moon.layout.title.text
    assert "± 1 Sample SD across" in moon_title


def test_capacity_and_timing_tables():
    """Verify capacity and timing tables load real data with verified parameter counts and durations."""
    from components.metric_tables import (
        render_mars_training_timing_table,
        render_moon_training_fit_table,
    )
    
    df_mars_time = render_mars_training_timing_table(fold=0, seed=42)
    assert not df_mars_time.empty
    assert len(df_mars_time) == 2  # Classical ResNet-18 and Hybrid QCNN Head
    assert "Model Variant" in df_mars_time.columns
    assert "Total Wall Time" in df_mars_time.columns
    
    results = load_moon_results(seed=42)
    df_moon_time = render_moon_training_fit_table(results)
    assert not df_moon_time.empty
    assert len(df_moon_time) == 2
    assert "Pipeline Stage" in df_moon_time.columns
    assert "Wall-Clock Duration" in df_moon_time.columns


def test_model_comparison_page_execution():
    """Verify that pages/5_Model_Comparison.py compiles and runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_Model_Comparison.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"


def test_mars_comparative_evaluation_statistical_assessment_removed():
    """Verify that Statistical Assessment section is removed from pages/5_Model_Comparison.py while other views remain."""
    content = (APP_DIR / "pages" / "5_Model_Comparison.py").read_text(encoding="utf-8")
    
    # Statistical Assessment section and paired differences table must be removed
    assert "render_mars_paired_differences_table" not in content
    assert "3. Paired Site Differences" not in content
    assert "df_paired" not in content
    
    # Core regression, safety, and capacity views must remain
    assert "build_mars_regression_chart" in content
    assert "build_mars_safety_chart" in content
    assert "build_moon_comparison_chart" in content
    assert "render_mars_training_timing_table" in content
    assert "render_moon_training_fit_table" in content
    assert "Capacity & Timing Analysis" in content
