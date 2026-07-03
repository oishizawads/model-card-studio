"""二値分類モデルの学習と予測確率の取得。"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """ロジスティック回帰を学習する。特徴量は StandardScaler で標準化する（収束と較正の安定化）。
    クラス重みは既定（不均衡の様子を見せるため補正しない）。戻り値は sklearn Pipeline。"""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"サンプル数が不一致: X_train={X_train.shape[0]}, y_train={y_train.shape[0]}"
        )
    if X_train.shape[0] == 0:
        raise ValueError("学習データが空です")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )
    model.fit(X_train, y_train)
    return model


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    """陽性クラス（ラベル 1）の確率を返す。モデルがクラス [0,1] を持つことを前提とする。"""
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        return np.array([], dtype=float)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        # 陽性クラスが学習データに無い場合は確率 0 とする
        return np.zeros(X.shape[0], dtype=float)
    idx = classes.index(1)
    return np.asarray(proba[:, idx], dtype=float)
