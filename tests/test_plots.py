from __future__ import annotations

import altair as alt

from src.plots import plot_calibration, plot_confusion_matrix


def _spec(chart) -> dict:
    return chart.to_dict()


def test_plot_confusion_matrix_returns_altair_chart():
    confusion = {"tp": 30, "fp": 5, "tn": 50, "fn": 15}
    chart = plot_confusion_matrix(confusion, "陽性", "陰性")
    assert isinstance(chart, alt.LayerChart)
    spec = _spec(chart)
    assert spec["title"] == "混同行列"
    # ヒートマップ + テキストの2レイヤー
    assert len(spec["layer"]) == 2


def test_plot_confusion_matrix_values_in_data():
    confusion = {"tp": 3, "fp": 1, "tn": 4, "fn": 2}
    chart = plot_confusion_matrix(confusion, "pos", "neg")
    values = list(chart.data["count"])
    assert sorted(values) == [1, 2, 3, 4]


def test_plot_calibration_returns_altair_chart():
    calibration = {
        "centers": [0.1, 0.5, 0.9],
        "empirical": [0.08, 0.55, 0.92],
        "counts": [40, 30, 20],
    }
    chart = plot_calibration(calibration)
    assert isinstance(chart, alt.LayerChart)
    spec = _spec(chart)
    assert spec["title"] == "較正（信頼性）曲線"
    # 対角線 + 観測折れ線 + 観測点の3レイヤー
    assert len(spec["layer"]) == 3


def test_plot_calibration_empty_bins_safe():
    chart = plot_calibration({"centers": [], "empirical": [], "counts": []})
    # 空でも落ちず、対角線のみのチャートを返す
    spec = _spec(chart)
    assert spec["title"] == "較正（信頼性）曲線"
