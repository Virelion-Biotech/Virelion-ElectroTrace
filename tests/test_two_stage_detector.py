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


def test_adaptive_polarity_selects_negative_for_inverted_signal():
    signal = _synthetic_peaks(invert=True)
    decision = select_signal_polarity(signal, 360.0)
    assert decision.polarity == "negative"
    assert decision.negative_score > decision.positive_score


def test_adaptive_polarity_is_not_dual_polarity_merge():
    signal = _synthetic_peaks(invert=True)
    adaptive = detect_r_peaks(signal, 360.0, polarity="adaptive")
    negative = detect_r_peaks(signal, 360.0, polarity="negative")
    assert np.array_equal(adaptive, negative)
