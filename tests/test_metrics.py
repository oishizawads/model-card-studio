from __future__ import annotations

import numpy as np

from src.metrics import (
    apply_threshold,
    brier_score,
    compute_calibration,
    compute_metrics,
    confusion_counts,
    expected_calibration_error,
    threshold_recall_swing,
)


def test_apply_threshold_basic():
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    np.testing.assert_array_equal(apply_threshold(y_prob, 0.5), np.array([0, 0, 1, 1]))


def test_apply_threshold_boundary_inclusive():
    # 境界値は陽性側に含める（>=）
    np.testing.assert_array_equal(apply_threshold(np.array([0.5]), 0.5), np.array([1]))


def test_apply_threshold_empty():
    out = apply_threshold(np.array([]), 0.5)
    assert out.shape == (0,)
    assert out.dtype == int


def test_apply_threshold_invalid():
    import pytest

    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), 1.5)
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), -0.1)


def test_confusion_counts_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    cm = confusion_counts(y_true, y_pred)
    assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_confusion_counts_empty():
    assert confusion_counts(np.array([]), np.array([])) == {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }


def test_compute_metrics_known():
    # tp=3, fp=1, tn=4, fn=2 -> acc=7/10, p=3/4, r=3/5, f1=2*.75*.6/(.75+.6)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["support"] == 10
    assert np.isclose(m["accuracy"], 0.7)
    assert np.isclose(m["precision"], 0.75)
    assert np.isclose(m["recall"], 0.6)
    assert np.isclose(m["f1"], 2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_compute_metrics_all_negative_pred():
    # 陽性を一つも予測しない場合、precision は 0（分母 0 を潰す）
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.5


def test_compute_metrics_empty():
    m = compute_metrics(np.array([]), np.array([]))
    assert m["support"] == 0
    assert m["accuracy"] == 0.0


def test_brier_score_perfect_and_worst():
    assert brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])) == 0.0
    assert np.isclose(brier_score(np.array([1, 0]), np.array([0.0, 1.0])), 1.0)


def test_brier_score_empty():
    assert brier_score(np.array([]), np.array([])) == 0.0


def test_brier_score_shape_mismatch():
    import pytest

    with pytest.raises(ValueError):
        brier_score(np.array([1, 0]), np.array([0.5]))


def test_ece_perfect_calibration():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_prob, n_bins=5) == 0.0


def test_ece_decreasing_with_better_calibration():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) > 0.5).astype(int)
    bad = np.where(y_true == 1, 0.9, 0.1) + rng.normal(0, 0.05, 500)
    good = y_true.astype(float) * 0.0 + np.clip(y_true + rng.normal(0, 0.02, 500), 0, 1)
    bad = np.clip(bad, 0, 1)
    assert expected_calibration_error(y_true, good, n_bins=10) < expected_calibration_error(
        y_true, bad, n_bins=10
    )


def test_compute_calibration_structure():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.95])
    cal = compute_calibration(y_true, y_prob, n_bins=5)
    assert cal["n_bins"] == 5
    assert len(cal["centers"]) == len(cal["empirical"]) == len(cal["counts"])
    assert 0.0 <= cal["brier"] <= 1.0
    assert 0.0 <= cal["ece"] <= 1.0


def test_compute_calibration_empty():
    cal = compute_calibration(np.array([]), np.array([]), n_bins=5)
    assert cal["centers"] == []
    assert cal["brier"] == 0.0


def test_threshold_recall_swing_zero_when_confident():
    # 確率が極端に離れていれば ±0.1 で再現率は変わらない
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.99, 0.98, 0.01, 0.02])
    assert threshold_recall_swing(y_true, y_prob, 0.5) == 0.0


def test_threshold_recall_swing_positive_near_boundary():
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.45, 0.55, 0.5])
    swing = threshold_recall_swing(y_true, y_prob, 0.5)
    assert swing > 0.0
