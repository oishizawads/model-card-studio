"""混同行列と較正曲線の描画。matplotlib を使い、日本語フォントを安全に設定する。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def _setup_font() -> None:
    """利用可能な CJK フォントを探して設定。無ければ既定フォントのまま（クラッシュしない）。"""
    candidates = [
        "Hiragino Sans",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "IPAGothic",
        "Meiryo",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_font()


def plot_confusion_matrix(
    confusion: dict[str, int], pos_label: str, neg_label: str
) -> plt.Figure:
    """混同行列をヒートマップで描画。行=正解、列=予測。"""
    tn, fp = confusion["tn"], confusion["fp"]
    fn, tp = confusion["fn"], confusion["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [neg_label, pos_label]

    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("予測")
    ax.set_ylabel("正解")
    ax.set_title("混同行列")
    total = int(matrix.sum())
    for i in range(2):
        for j in range(2):
            value = int(matrix[i, j])
            color = "white" if value > total * 0.4 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="件数")
    return fig


def plot_calibration(calibration: dict) -> plt.Figure:
    """信頼性曲線（reliability curve）と対角線を描画。"""
    centers = calibration.get("centers", [])
    empirical = calibration.get("empirical", [])
    counts = calibration.get("counts", [])

    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="完全較正")
    if centers:
        sizes = [max(8, min(120, c * 2)) for c in counts]
        ax.scatter(centers, empirical, s=sizes, color="#1f77b4", alpha=0.8, label="観測")
        ax.plot(centers, empirical, color="#1f77b4", alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("予測確率（bin 平均）")
    ax.set_ylabel("実測の陽性率")
    ax.set_title("較正（信頼性）曲線")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig
