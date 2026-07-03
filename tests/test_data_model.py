from __future__ import annotations

import numpy as np
import pytest

from src.data import available_datasets, load_dataset
from src.model import predict_proba_positive, train_model


def test_available_datasets_contains_expected():
    names = available_datasets()
    assert "breast_cancer" in names
    assert "imbalanced_synthetic" in names


def test_load_breast_cancer_shapes():
    ds = load_dataset("breast_cancer", test_size=0.3, random_state=0)
    assert ds.X_train.shape[0] + ds.X_test.shape[0] == ds.total_count
    assert ds.X_train.shape[1] == ds.X_test.shape[1]
    assert len(ds.feature_names) == ds.X_train.shape[1]
    assert ds.pos_count + ds.neg_count == ds.total_count


def test_load_imbalanced_has_minority():
    ds = load_dataset("imbalanced_synthetic", test_size=0.3, random_state=0)
    ratio = max(ds.pos_count, ds.neg_count) / max(1, min(ds.pos_count, ds.neg_count))
    assert ratio >= 4.0  # 意図的に不均衡


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("does_not_exist")


def test_load_invalid_test_size_raises():
    with pytest.raises(ValueError):
        load_dataset("breast_cancer", test_size=0.0)
    with pytest.raises(ValueError):
        load_dataset("breast_cancer", test_size=1.0)


def test_train_model_and_predict_proba():
    ds = load_dataset("breast_cancer", test_size=0.3, random_state=0)
    model = train_model(ds.X_train, ds.y_train, random_state=0)
    proba = predict_proba_positive(model, ds.X_test)
    assert proba.shape == (ds.X_test.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_proba_empty():
    ds = load_dataset("breast_cancer", test_size=0.3, random_state=0)
    model = train_model(ds.X_train, ds.y_train, random_state=0)
    out = predict_proba_positive(model, np.empty((0, ds.X_train.shape[1])))
    assert out.shape == (0,)


def test_train_model_mismatched_samples_raises():
    with pytest.raises(ValueError):
        train_model(np.zeros((5, 2)), np.zeros(3))
