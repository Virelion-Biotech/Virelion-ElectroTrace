import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from electrotrace.validation_detectors import detect_r_peaks_two_stage


def test_two_stage_detector_returns_retained_peaks_and_probabilities():
    fs = 360.0
    t = np.arange(7200) / fs
    signal = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    for peak in [900, 1800, 2700, 3600, 4500, 5400, 6300]:
        lo = max(0, peak - 4)
        hi = min(len(signal), peak + 5)
        signal[lo:hi] += np.hanning(hi - lo) * 2.0

    candidates = np.array([900, 1800, 2700, 3600, 4500, 5400, 6300])
    X, names = _candidate_features(signal, fs, candidates, np.ones(len(candidates)))
    X_train = np.vstack([X, X + 0.01])
    y_train = np.array([1] * len(candidates) + [0] * len(candidates))
    model = CandidateSuppressor().fit(X_train, y_train, target_recall=0.9, n_estimators=20)
    model.feature_names = names

    retained, probabilities = detect_r_peaks_two_stage(signal, fs, model)
    assert retained.ndim == 1
    assert probabilities.shape[0] == len(candidates)
    assert np.isfinite(probabilities).all()
