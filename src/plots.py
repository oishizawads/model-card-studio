"""混同行列と較正曲線の描画。Altair を使用し、WASM 環境での日本語豆腐化を回避する。"""
from __future__ import annotations

import pandas as pd

import altair as alt

from src.brand import PALETTE


def plot_confusion_matrix(
    confusion: dict[str, int], pos_label: str, neg_label: str
) -> alt.LayerChart:
    """混同行列をヒートマップで描画。行=正解、列=予測。"""
    tn, fp = confusion["tn"], confusion["fp"]
    fn, tp = confusion["fn"], confusion["tp"]
    total = tn + fp + fn + tp

    data = pd.DataFrame([
        {"正解": neg_label, "予測": neg_label, "count": tn, "label": str(tn)},
        {"正解": neg_label, "予測": pos_label, "count": fp, "label": str(fp)},
        {"正解": pos_label, "予測": neg_label, "count": fn, "label": str(fn)},
        {"正解": pos_label, "予測": pos_label, "count": tp, "label": str(tp)},
    ])

    order = [neg_label, pos_label]

    base = alt.Chart(data).encode(
        x=alt.X("予測:N", sort=order, title="予測"),
        y=alt.Y("正解:N", sort=order, title="正解"),
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "count:Q",
            scale=alt.Scale(scheme="blues"),
            legend=alt.Legend(title="件数"),
        )
    )

    threshold = total * 0.4 if total > 0 else 1
    text = base.mark_text(baseline="middle", fontSize=14).encode(
        text="label:N",
        color=alt.condition(
            alt.datum["count"] > threshold,
            alt.value("white"),
            alt.value("#0d1526"),
        ),
    )

    return (heatmap + text).properties(
        title="混同行列",
        width=280,
        height=240,
    )


def plot_calibration(calibration: dict) -> alt.LayerChart:
    """信頼性曲線（reliability curve）と対角線を描画。"""
    centers = calibration.get("centers", [])
    empirical = calibration.get("empirical", [])
    counts = calibration.get("counts", [])

    diag_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    diag = (
        alt.Chart(diag_df)
        .mark_line(color="#5b6474", strokeDash=[4, 4])
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), title="予測確率（bin 平均）"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]), title="実測の陽性率"),
        )
    )

    if not centers:
        return diag.properties(title="較正（信頼性）曲線", width=280, height=240)

    obs_df = pd.DataFrame({"center": centers, "empirical": empirical, "count": counts})

    obs_line = (
        alt.Chart(obs_df)
        .mark_line(color=PALETTE[0], opacity=0.5)
        .encode(
            x=alt.X("center:Q"),
            y=alt.Y("empirical:Q"),
        )
    )

    obs_scatter = (
        alt.Chart(obs_df)
        .mark_point(filled=True, color=PALETTE[0])
        .encode(
            x=alt.X("center:Q"),
            y=alt.Y("empirical:Q"),
            size=alt.Size(
                "count:Q",
                scale=alt.Scale(range=[40, 240]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("center:Q", format=".2f", title="予測確率"),
                alt.Tooltip("empirical:Q", format=".2f", title="実測陽性率"),
                alt.Tooltip("count:Q", title="件数"),
            ],
        )
    )

    return (diag + obs_line + obs_scatter).properties(
        title="較正（信頼性）曲線",
        width=280,
        height=240,
    )
