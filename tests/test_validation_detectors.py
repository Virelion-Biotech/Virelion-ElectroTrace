import numpy as np
import pytest

from electrotrace.validation_detectors import detect_r_peaks


def _signal_with_peaks(sign: float) -> np.ndarray:
    fs = 360.0
    signal = 0.02 * np.sin(2 * np.pi * 1.0 * np.arange(7200) / fs)
    for peak in [900, 1800, 2700, 3600, 4500, 5400, 6300]:
        lo, hi = peak - 4, peak + 5
        signal[lo:hi] += sign * np.hanning(hi - lo) * 2.0
    return signal


def test_positive_polarity_preserves_positive_peaks():
    peaks = detect_r_peaks(_signal_with_peaks(1.0), 360.0, polarity="positive")
    assert len(peaks) >= 7


def test_negative_polarity_detects_inverted_peaks():
    peaks = detect_r_peaks(_signal_with_peaks(-1.0), 360.0, polarity="negative")
    assert len(peaks) >= 7


def test_adaptive_polarity_detects_inverted_and_positive():
    positive = detect_r_peaks(_signal_with_peaks(1.0), 360.0, polarity="adaptive")
    negative = detect_r_peaks(_signal_with_peaks(-1.0), 360.0, polarity="adaptive")
    assert len(positive) >= 7
    assert len(negative) >= 7


def test_invalid_polarity_rejected():
    with pytest.raises(ValueError):
        detect_r_peaks(np.ones(100), 360.0, polarity="sideways")
