import numpy as np
import pytest

from electrotrace.benchmark import benchmark_models
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.ml import rank_uncertain, train_classifier
from electrotrace.phenotype import beat_phenotypes


def _ecg(fs=200.0, n=4000):
    t = np.arange(n, dtype=float) / fs
    y = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    peaks = np.array([400, 800, 1200, 1600, 2000, 2400, 2800, 3200])
    for p in peaks:
        y[p - 2:p + 3] += np.array([0.5, 1.0, 1.5, 1.0, 0.5])
    return t, y, peaks


def test_active_learning_supports_pipeline_classes():
    t, signal, peaks = _ecg()
    annotations = [
        {"type": "point", "status": "accepted", "time": float(t[p]), "label": "normal"}
        for p in peaks[:4]
    ]
    annotations[1]["label"] = "abnormal"
    model, _ = train_classifier(signal, 200.0, peaks, annotations, time=t)
    suggestions = rank_uncertain(signal, 200.0, peaks, model, top_n=2, time=t, annotations=annotations)
    assert isinstance(suggestions, list)
    assert all("predicted_label" in item for item in suggestions)


def test_benchmark_supports_logistic_pipeline_classes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(16, 3))
    y = np.array([0, 0, 1, 1] * 4)
    groups = np.repeat(np.arange(8), 2)
    result = benchmark_models(X, y, groups, folds=2)
    assert result["models"]["logistic_regression"]["summary"]["accuracy"]["mean"] is not None
    assert result["models"]["random_forest"]["summary"]["balanced_accuracy"]["mean"] is not None


def test_suppressor_uses_held_out_threshold_calibration():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = np.array([0, 1] * 20)
    model = CandidateSuppressor().fit(X, y, n_estimators=20)
    assert model.metadata.calibration_method == "held_out_stratified"
    assert model.metadata.calibration_candidates > 0
    assert 0.0 <= model.metadata.threshold <= 1.0


def test_suppressor_can_disable_calibration_explicitly():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 5))
    y = np.array([0, 1] * 4)
    model = CandidateSuppressor().fit(X, y, n_estimators=10, calibration_fraction=0)
    assert model.metadata.calibration_method == "training_resubstitution"


def test_phenotype_rejects_partial_nonfinite_signal():
    t = np.arange(100, dtype=float) / 100.0
    signal = np.sin(2 * np.pi * t)
    signal[20] = np.nan
    with pytest.raises(ValueError, match="signal contains NaN"):
        beat_phenotypes(t, signal, np.array([10, 50]))
