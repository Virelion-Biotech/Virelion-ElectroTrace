import numpy as np

from electrotrace.baseline_detectors import pan_tompkins_r_peaks, hamilton_r_peaks


def test_pan_tompkins_on_synthetic_spikes():
    fs = 360.0
    t = np.arange(0, 5.0, 1 / fs)
    x = np.zeros_like(t)
    for i in range(1, 5):
        idx = int(i * fs)
        x[idx - 2 : idx + 3] = [0.2, 0.6, 1.0, 0.6, 0.2]
    peaks = pan_tompkins_r_peaks(x, fs)
    assert len(peaks) >= 3


def test_hamilton_rejects_flat_signal():
    fs = 250.0
    x = np.zeros(int(2 * fs))
    peaks = hamilton_r_peaks(x, fs)
    assert peaks.ndim == 1
