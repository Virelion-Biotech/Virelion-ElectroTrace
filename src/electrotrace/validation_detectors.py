"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def detect_r_peaks(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Run ElectroTrace's current heuristic R-peak candidate detector.

    This deliberately mirrors the server's default detector settings so external
    validation measures the detector users actually receive from the application.
    It is a research candidate generator, not a validated clinical algorithm.
    """
    signal = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if signal.ndim != 1 or signal.size < 8:
        raise ValueError("signal must be one-dimensional with at least eight samples")
    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return np.asarray([], dtype=int)
    peaks, _ = sps.find_peaks(
        z,
        distance=max(1, int(round(fs_hz * 0.25))),
        prominence=scale * 0.5,
    )
    return peaks.astype(int)
