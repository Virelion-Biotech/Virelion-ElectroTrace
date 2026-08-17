import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.validation_detectors import detect_r_peaks, detect_r_peaks_two_stage, select_signal_polarity


def _synthetic_peaks(invert: bool = False) -> np.ndarray:
    fs = 360.0
    t = np.arange(7200) / fs
    signal = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    for peak in [900, 1800, 2700, 3600, 4500, 5400, 6300]:
        lo = max(0, peak - 4)
        hi = min(len(signal), peak + 5)
        pulse = np.hanning(hi - lo) * 2.0
        signal[lo:hi] += -pulse if invert else pulse
    return signal


def test_two_stage_detector_returns_retained_peaks_and_probabilities():
    fs = 360.0
    signal = _synthetic_peaks()
    candidates = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300])
    X, names = _candidate_features(signal, fs, candidates, np.ones(len(candidates)))
    X_train = np.vstack([X, X + 0.01])
    y_train = np.array([1] * len(candidates) + [0] * len(candidates))
    model = CandidateSuppressor().fit(X_train, y_train, target_recall=0.9, n_estimators=20)
    model.feature_names = names

    retained, probabilities = detect_r_peaks_two_stage(signal, fs, model)
    stage1_candidates = detect_r_peaks(signal, fs)
    assert retained.ndim == 1
    assert probabilities.shape[0] == len(stage1_candidates)
    assert np.isfinite(probabilities).all()


def test_adaptive_polarity_returns_supported_conservative_decision():
    signal = _synthetic_peaks(invert=True)
    decision = select_signal_polarity(signal, 360.0)
    assert decision.polarity in {"positive", "negative"}
    assert decision.positive_score >= 0.0
    assert decision.negative_score >= 0.0
    assert decision.confidence >= 0.0


def test_adaptive_polarity_matches_its_single_polarity_decision():
    signal = _synthetic_peaks(invert=True)
    decision = select_signal_polarity(signal, 360.0)
    adaptive = detect_r_peaks(signal, 360.0, polarity="adaptive")
    selected = detect_r_peaks(signal, 360.0, polarity=decision.polarity)
    assert np.array_equal(adaptive, selected)


def test_adaptive_polarity_never_merges_both_polarities():
    signal = _synthetic_peaks(invert=True)
    adaptive = detect_r_peaks(signal, 360.0, polarity="adaptive")
    positive = detect_r_peaks(signal, 360.0, polarity="positive")
    negative = detect_r_peaks(signal, 360.0, polarity="negative")
    merged = np.sort(np.unique(np.concatenate([positive, negative])))
    assert not np.array_equal(adaptive, merged)
