import numpy as np
import pytest

from electrotrace.validation_detectors import detect_r_peaks


def test_reference_detector_returns_expected_peaks():
    fs = 500.0
    signal = np.zeros(2000, dtype=float)
    signal[[250, 750, 1250, 1750]] = 5.0
    peaks = detect_r_peaks(signal, fs)
    assert peaks.tolist() == [250, 750, 1250, 1750]


def test_reference_detector_rejects_nonfinite_signal():
    signal = np.zeros(100, dtype=float)
    signal[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        detect_r_peaks(signal, 500.0)
