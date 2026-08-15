"""
Optional, non-destructive signal processing.

Filtering always produces a *separate* display array -- the original raw
signal loaded from disk is never overwritten -- since annotations may later
be used for scientific validation against the raw trace.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _butter_filter(data: np.ndarray, fs: float, cutoff, btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    b, a = sps.butter(order, normal_cutoff, btype=btype)
    return sps.filtfilt(b, a, data)


def remove_baseline_wander(data: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """High-pass filter to remove slow baseline drift."""
    if fs is None or fs <= cutoff_hz * 2:
        return data
    return _butter_filter(data, fs, cutoff_hz, "highpass")


def low_pass(data: np.ndarray, fs: float, cutoff_hz: float = 40.0) -> np.ndarray:
    if fs is None or fs <= cutoff_hz * 2:
        return data
    return _butter_filter(data, fs, cutoff_hz, "lowpass")


def high_pass(data: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    if fs is None or fs <= cutoff_hz * 2:
        return data
    return _butter_filter(data, fs, cutoff_hz, "highpass")


def notch_filter(data: np.ndarray, fs: float, freq_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    if fs is None or fs <= freq_hz * 2:
        return data
    b, a = sps.iirnotch(freq_hz, q, fs)
    return sps.filtfilt(b, a, data)


def apply_pipeline(
    data: np.ndarray,
    fs: float,
    baseline: bool = False,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    notch_hz: float | None = None,
) -> np.ndarray:
    """Apply the selected filters in sequence, returning a new array."""
    out = np.asarray(data, dtype=float).copy()
    if fs is None:
        return out
    if baseline:
        out = remove_baseline_wander(out, fs)
    if highpass_hz:
        out = high_pass(out, fs, highpass_hz)
    if lowpass_hz:
        out = low_pass(out, fs, lowpass_hz)
    if notch_hz:
        out = notch_filter(out, fs, notch_hz)
    return out
