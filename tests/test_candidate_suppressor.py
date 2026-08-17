import numpy as np
import pytest

from electrotrace.candidate_suppressor import (
    CandidateSuppressor,
    _candidate_features,
    label_candidates,
    select_threshold_for_recall,
)


def test_candidate_features_have_stable_shape_and_finite_values():
    fs = 360.0
    t = np.arange(4000) / fs
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.15 * np.sin(2 * np.pi * 12 * t)
    candidates = np.array([500, 1000, 1500])
    prominences = np.array([1.0, 1.2, 0.9])
    features, names = _candidate_features(signal, fs, candidates, prominences)
    assert features.shape == (3, len(names))
    assert features.shape[1] >= 80
    assert np.isfinite(features).all()


def test_label_candidates_is_one_to_one():
    candidates = np.array([100, 200, 300])
    references = np.array([102, 300])
    labels = label_candidates(candidates, references, fs_hz=1000, tolerance_s=0.005)
    assert labels.tolist() == [1, 0, 1]


def test_threshold_selection_hits_target_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([.99, .8, .7, .6, .95, .3, .2, .1])
    threshold = select_threshold_for_recall(y, p, target_recall=.75)
    assert threshold == pytest.approx(.7)


def test_suppressor_trains_and_filters_candidates():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(2, 0.2, size=(40, 8)),
        rng.normal(-2, 0.2, size=(40, 8)),
    ])
    y = np.array([1] * 40 + [0] * 40)
    model = CandidateSuppressor().fit(X, y, target_recall=.95, n_estimators=25)
    retained, probabilities = model.filter_candidates(np.arange(80), X)
    assert model.fitted
    assert retained.ndim == 1
    assert probabilities.shape == (80,)
    assert model.metadata.n_training_candidates == 80
    assert model.metadata.n_positive_candidates == 40


def test_suppressor_rejects_single_class_training():
    X = np.ones((6, 4))
    y = np.ones(6, dtype=int)
    with pytest.raises(ValueError):
        CandidateSuppressor().fit(X, y)
