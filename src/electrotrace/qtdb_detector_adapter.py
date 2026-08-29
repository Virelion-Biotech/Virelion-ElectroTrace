"""Adaptive-polarity R-peak detector adapter for QTDB confirmatory harness."""
from __future__ import annotations
import numpy as np
from electrotrace.validation_detectors import detect_r_peaks

def detect_r_peaks_adaptive(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    return detect_r_peaks(signal, float(fs_hz), polarity="adaptive")
