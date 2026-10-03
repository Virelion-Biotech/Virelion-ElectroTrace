"""Stage-2 finding (2026-09-13): I03 remained broken even after the Stage-1
windowed_std fix, despite Stage-1 generating 2x the reference beat count in
candidates for it. Root cause: candidate_suppressor.py's `_candidate_features`
normalized prominences/z-signal by its own independent
`global_scale = std(signal - median(signal))` -- the same fragile-global-
statistic problem Stage-1 had, just in feature normalization instead of
threshold-setting. I03 has ~8mV of near-linear DC drift across its 30-minute
recording (~20x its ~0.3-0.5mV true QRS amplitude), so global_scale there is
dominated by the drift, and true-beat candidates normalize to near-zero --
indistinguishable from noise to the RF. Fixed by defaulting
`_candidate_features` to the same windowed_std estimator as Stage 1 (see
scale_estimation.py for the full history).

This test reproduces that mechanism directly on synthetic data (no PhysioNet
data needed): a record with a large linear drift dwarfing the QRS amplitude.
"""
import numpy as np
from scipy.signal import find_peaks

from electrotrace.candidate_suppressor import _candidate_features


def _drifting_signal_with_true_beat(seed=0):
    """120s @ 257Hz, regular QRS-like pulses (true amplitude ~0.4) riding on
    an 8-unit linear drift across the whole record -- ~20x the QRS amplitude,
    matching the scale of I03's real drift (mapped onto a much shorter
    synthetic record purely so the test runs fast)."""
    fs = 257.0
    n = int(120 * fs)
    rng = np.random.default_rng(seed)
    drift = np.linspace(-4.0, 4.0, n)
    signal = drift + rng.normal(0, 0.01, size=n)

    rr = int(round(0.8 * fs))
    half_w = max(3, int(round(0.02 * fs)))
    beats = np.arange(rr, n - rr, rr)
    for p in beats:
        lo, hi = p - half_w, p + half_w + 1
        signal[lo:hi] += 0.4 * np.hanning(hi - lo)

    mid_beat = beats[len(beats) // 2]
    window = signal[mid_beat - half_w - 5: mid_beat + half_w + 5]
    peaks, props = find_peaks(window - np.median(window), prominence=0.05)
    local_idx = peaks[np.argmin(np.abs(peaks - (len(window) // 2)))]
    candidate_idx = mid_beat - half_w - 5 + local_idx
    prom = props["prominences"][np.argmin(np.abs(peaks - (len(window) // 2)))]
    return fs, signal, int(candidate_idx), float(prom)


def test_global_std_normalizes_true_beat_prominence_to_near_noise_level():
    """Reproduces the bug: under global std, a true beat's prominence
    normalizes to a tiny value once drift dominates the scale estimate."""
    fs, signal, candidate_idx, prom = _drifting_signal_with_true_beat()
    X, names = _candidate_features(signal, fs, [candidate_idx], [prom], scale_method="std")
    prominence_z = X[0, names.index("prominence_z")]
    assert prominence_z < 0.3


def test_windowed_std_keeps_true_beat_prominence_signal_like():
    """The fix: windowed_std isn't dominated by slow multi-minute drift
    (each window is short relative to it), so the true beat's normalized
    prominence stays clearly signal-like."""
    fs, signal, candidate_idx, prom = _drifting_signal_with_true_beat()
    X, names = _candidate_features(signal, fs, [candidate_idx], [prom], scale_method="windowed_std")
    prominence_z = X[0, names.index("prominence_z")]
    assert prominence_z > 1.0


def test_candidate_features_default_scale_method_is_windowed_std():
    from electrotrace.scale_estimation import DEFAULT_SCALE_METHOD
    assert DEFAULT_SCALE_METHOD == "windowed_std"
    # and _candidate_features actually uses it as its own default (not just
    # importing the name) -- confirm by comparing against an explicit call
    fs, signal, candidate_idx, prom = _drifting_signal_with_true_beat()
    X_default, _ = _candidate_features(signal, fs, [candidate_idx], [prom])
    X_explicit, _ = _candidate_features(signal, fs, [candidate_idx], [prom], scale_method="windowed_std")
    assert np.allclose(X_default, X_explicit)
