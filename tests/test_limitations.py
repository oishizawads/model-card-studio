from __future__ import annotations

from src.limitations import generate_limitations
from src.metrics import compute_calibration, compute_metrics, confusion_counts


def _bundle(pos_count=300, neg_count=300, source="sklearn.datasets"):
    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "source": source,
    }


def test_always_includes_generic_caveats():
    metrics = {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 500}
    confusion = {"tp": 100, "fp": 5, "tn": 100, "fn": 5}
    cal = compute_calibration([1, 0, 1, 0], [0.9, 0.1, 0.9, 0.1], n_bins=5)
    items = generate_limitations(metrics, confusion, cal, _bundle(), 0.5, 0.0)
    assert any("分布シフト" in i for i in items)
    assert any("実運用の母集団" in i for i in items)


def test_imbalance_triggered():
    metrics = {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 500}
    confusion = {"tp": 10, "fp": 1, "tn": 200, "fn": 1}
    cal = {"brier": 0.05, "ece": 0.02, "centers": [], "empirical": [], "counts": [], "n_bins": 10}
    items = generate_limitations(metrics, confusion, cal, _bundle(pos_count=50, neg_count=950), 0.5, 0.0)
    assert any("クラス不均衡" in i for i in items)


def test_low_recall_and_precision_triggered():
    metrics = {"accuracy": 0.5, "precision": 0.4, "recall": 0.3, "f1": 0.34, "support": 1000}
    confusion = {"tp": 30, "fp": 45, "tn": 20, "fn": 70}
    cal = {"brier": 0.05, "ece": 0.02, "centers": [], "empirical": [], "counts": [], "n_bins": 10}
    items = generate_limitations(metrics, confusion, cal, _bundle(), 0.5, 0.0)
    assert any("再現率" in i and "取りこぼし" in i for i in items)
    assert any("適合率" in i and "誤報" in i for i in items)


def test_calibration_triggered():
    metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 1000}
    confusion = {"tp": 90, "fp": 10, "tn": 90, "fn": 10}
    cal = {"brier": 0.3, "ece": 0.2, "centers": [], "empirical": [], "counts": [], "n_bins": 10}
    items = generate_limitations(metrics, confusion, cal, _bundle(), 0.5, 0.0)
    assert any("Brier" in i for i in items)
    assert any("ECE" in i for i in items)


def test_threshold_swing_triggered():
    metrics = {"accuracy": 0.8, "precision": 0.8, "recall": 0.6, "f1": 0.69, "support": 1000}
    confusion = {"tp": 60, "fp": 15, "tn": 85, "fn": 40}
    cal = {"brier": 0.05, "ece": 0.02, "centers": [], "empirical": [], "counts": [], "n_bins": 10}
    items = generate_limitations(metrics, confusion, cal, _bundle(), 0.5, 0.4)
    assert any("しきい値" in i and "再現率が" in i for i in items)


def test_small_sample_triggered():
    metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 50}
    confusion = {"tp": 5, "fp": 1, "tn": 5, "fn": 1}
    cal = {"brier": 0.05, "ece": 0.02, "centers": [], "empirical": [], "counts": [], "n_bins": 10}
    items = generate_limitations(metrics, confusion, cal, _bundle(), 0.5, 0.0)
    assert any("評価サンプル数" in i for i in items)
