"""データセット読み込み。sklearn サンプル / 合成データのみを使用する。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    """学習/評価に使うデータとメタ情報。"""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    pos_label: str
    neg_label: str
    description: str
    source: str

    @property
    def pos_count(self) -> int:
        return int(np.sum(self.y_train == 1) + np.sum(self.y_test == 1))

    @property
    def neg_count(self) -> int:
        return int(np.sum(self.y_train == 0) + np.sum(self.y_test == 0))

    @property
    def total_count(self) -> int:
        return int(self.X_train.shape[0] + self.X_test.shape[0])


_DATASET_NAMES = ("breast_cancer", "imbalanced_synthetic")


def available_datasets() -> tuple[str, ...]:
    return _DATASET_NAMES


def _load_breast_cancer(test_size: float, random_state: int) -> Dataset:
    bunch = datasets.load_breast_cancer()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)
    # sklearn の定義: 0=malignant(悪性), 1=benign(良性)。陽性を benign とする。
    return Dataset(
        *_split(X, y, test_size, random_state),
        feature_names=list(bunch.feature_names),
        pos_label="benign(良性)",
        neg_label="malignant(悪性)",
        description="sklearn の Breast Cancer Wisconsin データセット。細胞核の形状特徴量から悪性/良性を分類する。",
        source="sklearn.datasets.load_breast_cancer",
    )


def _load_imbalanced_synthetic(test_size: float, random_state: int) -> Dataset:
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        flip_y=0.02,
        class_sep=0.9,
        random_state=random_state,
    )
    y = np.asarray(y, dtype=int)
    return Dataset(
        *_split(X, y, test_size, random_state),
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        pos_label="陽性(少数派)",
        neg_label="陰性(多数派)",
        description="sklearn make_classification による合成データ。意図的に 95:5 のクラス不均衡を与えている。",
        source="sklearn.datasets.make_classification (weights=[0.95, 0.05])",
    )


def _split(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int):
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size は 0<test_size<1 の範囲で指定してください（指定値: {test_size}）")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return Xtr, Xte, ytr, yte


def load_dataset(name: str, test_size: float = 0.3, random_state: int = 42) -> Dataset:
    """名前指定でデータセットを読み込む。未知の名前は ValueError。"""
    if name not in _DATASET_NAMES:
        raise ValueError(
            f"未知のデータセット名: {name!r}。利用可能: {', '.join(_DATASET_NAMES)}"
        )
    if name == "breast_cancer":
        return _load_breast_cancer(test_size, random_state)
    return _load_imbalanced_synthetic(test_size, random_state)
