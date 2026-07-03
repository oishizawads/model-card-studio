"""制限事項の自動文章化。指標とデータ特性から根拠付きの注意点を生成する。"""
from __future__ import annotations

from typing import Any


def generate_limitations(
    metrics: dict[str, Any],
    confusion: dict[str, int],
    calibration: dict[str, Any],
    dataset_info: dict[str, Any],
    threshold: float,
    recall_swing: float,
) -> list[str]:
    """評価結果から制限事項を自動生成する。常に末尾に汎用的な注意を2件追加する。"""
    items: list[str] = []
    pos = int(dataset_info.get("pos_count", 0))
    neg = int(dataset_info.get("neg_count", 0))
    minority = max(1, min(pos, neg))
    ratio = max(pos, neg) / minority
    if ratio >= 4.0:
        items.append(
            f"クラス不均衡が大きい（陽性:陰性 = {pos}:{neg}、比 {ratio:.1f}:1）。"
            "全体の正解率が高く見えても少数クラスの再現率は低くなりやすく、少数クラス単独で評価すること。"
        )
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    if recall < 0.7:
        items.append(
            f"再現率（Recall）が {recall:.2f} と低く、陽性の取りこぼしが多い。"
            "しきい値を下げるか、クラス重み付け / 再サンプリングを検討すること。"
        )
    if precision < 0.7:
        items.append(
            f"適合率（Precision）が {precision:.2f} と低く、誤報が多い。"
            "しきい値を上げて陽性判定を厳しくすることを検討すること。"
        )
    if abs(recall - precision) >= 0.2:
        items.append(
            f"再現率 {recall:.2f} と適合率 {precision:.2f} に大きく差がある。"
            "正解率だけではこの乖離が見えないため、用途に応じた指標を選ぶこと。"
        )
    brier = float(calibration.get("brier", 0.0))
    ece = float(calibration.get("ece", 0.0))
    if brier > 0.2:
        items.append(
            f"Brierスコア {brier:.3f} が高く、予測確率の較正が悪い。"
            "Platt scaling / isotonic 回帰による較正を検討すること。"
        )
    if ece > 0.1:
        items.append(
            f"期待較正誤差（ECE）{ece:.3f} が大きく、予測確率と実測頻度にずれがある。"
            "確率をそのまま意思決定に使わないこと。"
        )
    if recall_swing > 0.15:
        items.append(
            f"しきい値 ±0.1 で再現率が {recall_swing:.2f} 変動する。"
            "運用時のしきい値選定が結果に強く影響するため、根拠を明示して固定すること。"
        )
    support = int(metrics.get("support", 0))
    if support < 200:
        items.append(
            f"評価サンプル数が {support} と少なく、指標のばらつきが大きい。"
            "追加データで再評価すること。"
        )
    # 汎用注意（常に表示）
    items.append(
        "本モデルは学習データの分布のみで評価しており、運用環境での分布シフト・ドリフトには未対応。"
        "本番投入前に定期評価と人間のレビューを設けること。"
    )
    items.append(
        f"データは {dataset_info.get('source', 'サンプル/合成データ')} であり、"
        "実運用の母集団を代表しない。実データでの再検証が必須。"
    )
    return items
