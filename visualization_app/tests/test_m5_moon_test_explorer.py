"""Milestone 5 Acceptance Tests: Moon Test Explorer Logic and Data Validation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data_loading.moon_data import (
    load_moon_results,
    get_moon_predictions_df,
    load_moon_image,
    get_moon_demo_presets,
    MOON_DEMO_PRESETS,
)


def test_moon_seed42_count():
    df = get_moon_predictions_df(seed=42)
    assert len(df) == 779


def test_moon_softmax_sum_and_argmax():
    df = get_moon_predictions_df(seed=42)
    results = load_moon_results(seed=42)
    
    # Check probabilities sum to 1.0
    for prefix in ["cnn", "hybrid", "mlp"]:
        prob_sum = df[f"{prefix}_prob_ejecta"] + df[f"{prefix}_prob_oldcrater"] + df[f"{prefix}_prob_none"]
        assert np.allclose(prob_sum, 1.0, atol=1e-5)
        
    # Check predictions match argmax of logits
    cnn_logits = np.array(results["test"]["logits"]["cnn"])
    cnn_expected_argmax = np.argmax(cnn_logits, axis=-1)
    assert np.array_equal(df["cnn_pred_idx"].values, cnn_expected_argmax)
    
    hybrid_logits = np.array(results["test"]["logits"]["hybrid_qcnn"])
    hybrid_expected_argmax = np.argmax(hybrid_logits, axis=-1)
    assert np.array_equal(df["hybrid_pred_idx"].values, hybrid_expected_argmax)


def test_moon_correctness_and_disagreements():
    df = get_moon_predictions_df(seed=42)
    
    # Check correctness flags
    assert (df["cnn_correct"] == (df["cnn_pred_idx"] == df["true_label_idx"])).all()
    assert (df["hybrid_correct"] == (df["hybrid_pred_idx"] == df["true_label_idx"])).all()
    
    # Check disagreements
    disagree_df = df[df["has_disagreement"]]
    assert len(disagree_df) > 0
    for _, row in disagree_df.iterrows():
        assert not (row["cnn_pred_idx"] == row["hybrid_pred_idx"] == row["mlp_pred_idx"])


def test_moon_ejecta_class_count():
    df = get_moon_predictions_df(seed=42)
    ejecta_df = df[df["true_label_name"] == "ejecta"]
    assert len(ejecta_df) == 89
    
    oldcrater_df = df[df["true_label_name"] == "oldcrater"]
    assert len(oldcrater_df) == 157
    
    none_df = df[df["true_label_name"] == "none"]
    assert len(none_df) == 533


def test_moon_actual_jpeg_loads():
    df = get_moon_predictions_df(seed=42)
    first_sample = df.iloc[0]["sample_id"]
    img = load_moon_image(first_sample)
    assert img.size == (227, 227)


def test_moon_demo_presets():
    """Verify Moon demo presets return populated valid dataframes obeying exact logical rules."""
    for seed in [42, 43, 44]:
        df = get_moon_predictions_df(seed=seed)
        presets = get_moon_demo_presets(df)

        for name in MOON_DEMO_PRESETS:
            assert name in presets
            assert isinstance(presets[name], pd.DataFrame)

        # 1. All 3 Models Agree & Correct
        all_correct = presets["All 3 Models Agree & Correct"]
        assert len(all_correct) > 0
        assert all_correct["all_correct"].all()
        assert (all_correct["cnn_correct"] & all_correct["hybrid_correct"] & all_correct["mlp_correct"]).all()

        # 2. Hybrid QCNN Correct / Classical CNN Wrong
        hyb_beats = presets["Hybrid QCNN Correct / Classical CNN Wrong"]
        assert len(hyb_beats) > 0
        assert hyb_beats["hybrid_beats_cnn"].all()
        assert (hyb_beats["hybrid_correct"] & (~hyb_beats["cnn_correct"])).all()

        # 3. Classical CNN Correct / Hybrid QCNN Wrong
        cnn_beats = presets["Classical CNN Correct / Hybrid QCNN Wrong"]
        assert len(cnn_beats) > 0
        assert cnn_beats["cnn_beats_hybrid"].all()
        assert (cnn_beats["cnn_correct"] & (~cnn_beats["hybrid_correct"])).all()

        # 4. All 3 Models Wrong
        all_wrong = presets["All 3 Models Wrong (Hard Crater Example)"]
        assert len(all_wrong) > 0
        assert all_wrong["all_wrong"].all()
        assert ((~all_wrong["cnn_correct"]) & (~all_wrong["hybrid_correct"]) & (~all_wrong["mlp_correct"])).all()


def test_moon_confusion_pair_filtering():
    """Verify confusion transition filtering logic for Moon predictions."""
    df = get_moon_predictions_df(seed=42)

    # Test Hybrid QCNN oldcrater -> ejecta
    oldcrater_to_ejecta = df[(df["true_label_name"] == "oldcrater") & (df["hybrid_pred_name"] == "ejecta")]
    assert len(oldcrater_to_ejecta) > 0
    assert (oldcrater_to_ejecta["true_label_name"] == "oldcrater").all()
    assert (oldcrater_to_ejecta["hybrid_pred_name"] == "ejecta").all()

    # Test Classical CNN none -> oldcrater
    none_to_oldcrater = df[(df["true_label_name"] == "none") & (df["cnn_pred_name"] == "oldcrater")]
    assert len(none_to_oldcrater) > 0
    assert (none_to_oldcrater["true_label_name"] == "none").all()
    assert (none_to_oldcrater["cnn_pred_name"] == "oldcrater").all()


def test_moon_test_explorer_page_execution():
    """Verify that pages/4_Moon_Test_Explorer.py compiles and runs without unhandled exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_Moon_Test_Explorer.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"

