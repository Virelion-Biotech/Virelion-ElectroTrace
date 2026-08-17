import numpy as np
import pytest

from electrotrace.beats import segment_beats
from electrotrace.ml import _beat_features, build_training_set


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
