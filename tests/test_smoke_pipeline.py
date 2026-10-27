"""End-to-end smoke verification pipeline testing all models and controls."""

import time
from typing import Any

import pandas as pd

from src.evaluate import evaluate_all
from src.train import train_single_run


def test_smoke_trains_all_models_under_10_minutes(synthetic_workspace: dict[str, Any]) -> None:
    """Verifies that all 4 models and their matched controls train and evaluate under 10 minutes on CPU."""
    cfg = synthetic_workspace["config"]
    t0 = time.perf_counter()

    model_variants = [
        ("model1", "resnet18"),
        ("model2", "qcnn"),
        ("model2", "linear_control"),
        ("model2", "mlp_control"),
        ("model2", "no_entanglement_control"),
        ("model2", "permuted_qubits_control"),
        ("model3", "quanv"),
        ("model3", "classical_filter_control"),
        ("model3", "plain_cnn_control"),
        ("model4", "pure_qcnn"),
        ("model4", "tiny_cnn_control"),
        ("model4", "mlp_control"),
    ]

    for model_name, variant in model_variants:
        res = train_single_run(cfg, model_name=model_name, variant=variant, fold=0, seed=0)
        assert res["status"] in {"complete", "already_complete"}

    # Run evaluation pipeline
    summary_df, paired_df = evaluate_all(cfg, smoke=True)

    assert len(summary_df) == len(model_variants), (
        f"Expected {len(model_variants)} models in summary, got {len(summary_df)}"
    )

    # Verify zero NaNs in primary evaluation metrics
    primary_metrics = [
        "RMSE_mean_slope_5m_mean",
        "MAE_mean_slope_5m_mean",
        "safe_recall_mean",
        "false_safe_rate_mean",
    ]
    for m in primary_metrics:
        assert m in summary_df.columns, f"Missing metric column {m} in summary_df"
        assert not summary_df[m].isna().any(), f"Metric {m} contains NaNs in summary_df"

    # Verify paired comparison table exists and contains expected columns
    assert isinstance(paired_df, pd.DataFrame)
    if len(paired_df) > 0:
        assert "mean_diff" in paired_df.columns
        assert "ci_excludes_zero" in paired_df.columns

    elapsed = time.perf_counter() - t0
    assert elapsed < 600.0, f"Smoke pipeline took {elapsed:.1f}s, exceeding 10 minute limit (600s)"
