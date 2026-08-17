import numpy as np
import pytest

from electrotrace.beats import segment_beats
from electrotrace.ml import _beat_features, build_training_set, rank_uncertain, train_classifier


def test_segment_beats_builds_windows_and_rr():
    fs = 500
    time = np.arange(3000) / fs
    peaks = np.array([250, 750, 1250])
    beats = segment_beats(time, peaks, pre_s=0.2, post_s=0.3)
    assert len(beats) == 3
    assert beats[1].rr_prev_s == pytest.approx(1.0)
    assert beats[1].heart_rate_bpm == pytest.approx(60.0)
    assert beats[0].start == pytest.approx(0.3)


def test_features_are_finite():
    fs = 500
    x = np.sin(2 * np.pi * 2 * np.arange(1000) / fs)
    features = _beat_features(x, fs, 500)
    assert features.ndim == 1
    assert np.isfinite(features).all()


def test_training_requires_two_labels():
    fs = 500
    x = np.sin(2 * np.pi * 2 * np.arange(1500) / fs)
    peaks = np.array([250, 750, 1250])
    anns = [{"type": "point", "status": "accepted", "label": "R Peak", "time": 0.5}]
    with pytest.raises(ValueError, match="two or more labels"):
        build_training_set(x, fs, peaks, anns)


def test_training_requires_minimum_examples():
    fs = 500
    x = np.sin(2 * np.pi * 2 * np.arange(1500) / fs)
    peaks = np.array([250, 750, 1250])
    anns = [
        {"type": "point", "status": "accepted", "label": "R Peak", "time": 0.5},
        {"type": "point", "status": "accepted", "label": "Abnormal Beat", "time": 1.5},
    ]
    with pytest.raises(ValueError, match="at least 4"):
        build_training_set(x, fs, peaks, anns)


def test_active_learning_excludes_training_beats_and_spaces_candidates():
    fs = 500
    time = np.arange(2600) / fs
    x = np.sin(2 * np.pi * 2 * time)
    peaks = np.array([250, 750, 1250, 1750, 2000, 2050, 2100, 2150, 2200])
    anns = [
        {"type": "point", "status": "accepted", "label": "R Peak", "time": 0.5},
        {"type": "point", "status": "accepted", "label": "Abnormal Beat", "time": 1.5},
        {"type": "point", "status": "accepted", "label": "R Peak", "time": 2.5},
        {"type": "point", "status": "accepted", "label": "Abnormal Beat", "time": 3.5},
    ]
    model, metrics = train_classifier(x, fs, peaks, anns, time=time)
    assert metrics["n_training_examples"] == 4
    suggestions = rank_uncertain(x, fs, peaks, model, time=time, annotations=anns, top_n=10, min_spacing_s=0.25)
    suggested_times = sorted(item["time_s"] for item in suggestions)
    assert {round(t, 3) for t in suggested_times}.isdisjoint({0.5, 1.5, 2.5, 3.5})
    assert all((b - a) >= 0.25 for a, b in zip(suggested_times, suggested_times[1:]))
