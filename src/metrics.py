"""評価指標・混同行列・較正の計算。UI に依存しない純粋関数。"""
from __future__ import annotations

import numpy as np


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """予測確率をしきい値で二値ラベルに変換する。"""
    y_prob = np.asarray(y_prob, dtype=float)
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("予測確率に非有限値が含まれます")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("予測確率は 0〜1 の範囲である必要があります")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"しきい値は 0〜1 の範囲で指定してください（指定値: {threshold}）")
    if y_prob.size == 0:
        return np.array([], dtype=int)
    return (y_prob >= threshold).astype(int)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """混同行列の各成分を返す。空入力は全 0。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """accuracy / precision / recall / f1 / support を返す。分母 0 は 0.0 に潰す。"""
    cm = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    support = tp + fp + tn + fn
    if support == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    accuracy = (tp + tn) / support
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brierスコア（低いほど較正が良い）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true と y_prob の形状が一致しません")
    return float(np.mean((y_prob - y_true) ** 2))


def _bin_indices(y_prob: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if n_bins < 1:
        raise ValueError(f"n_bins は 1 以上で指定してください（指定値: {n_bins}）")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # 境界値の扱い: 各 bin は [left, right)。最後の bin だけ [left, right]。
    idx = np.digitize(y_prob, edges[1:-1])
    idx = np.clip(idx, 0, n_bins - 1)
    return idx, edges


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE（期待較正誤差）。各 bin の (信頼度 - 実測頻度) を重み付き平均。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true と y_prob の形状が一致しません")
    idx, _ = _bin_indices(y_prob, n_bins)
    n = y_true.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(y_prob[mask]))
        emp = float(np.mean(y_true[mask]))
        ece += (count / n) * abs(conf - emp)
    return float(ece)


def compute_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """較正情報をまとめて返す。reliability 曲線描画用の bin データを含む。"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {
            "brier": 0.0,
            "ece": 0.0,
            "centers": [],
            "empirical": [],
            "counts": [],
            "n_bins": n_bins,
        }
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true と y_prob の形状が一致しません")
    idx, edges = _bin_indices(y_prob, n_bins)
    centers, empirical, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        centers.append(float(np.mean(y_prob[mask])))
        empirical.append(float(np.mean(y_true[mask])))
        counts.append(count)
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "centers": centers,
        "empirical": empirical,
        "counts": counts,
        "n_bins": n_bins,
    }


def threshold_recall_swing(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, delta: float = 0.1
) -> float:
    """しきい値 ±delta での再現率の差（感度指標）。大きい=しきい値選定が結果に強く影響。"""
    lo = max(0.0, threshold - delta)
    hi = min(1.0, threshold + delta)
    rec_lo = compute_metrics(y_true, apply_threshold(y_prob, lo))["recall"]
    rec_hi = compute_metrics(y_true, apply_threshold(y_prob, hi))["recall"]
    return float(abs(rec_lo - rec_hi))
