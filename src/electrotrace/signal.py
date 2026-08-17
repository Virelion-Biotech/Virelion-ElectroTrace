"""Validated, non-destructive ECG filtering utilities."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

class FilterConfigurationError(ValueError):
    pass

def _validate_fs(fs: float) -> float:
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise FilterConfigurationError("sampling rate must be a positive finite number")
    return float(fs)

def _validate_cutoff(cutoff: float, fs: float, name: str) -> float:
    if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0:
        raise FilterConfigurationError(f"{name} must be positive and finite")
    nyq = fs / 2
    if cutoff >= nyq:
        raise FilterConfigurationError(f"{name} must be below Nyquist frequency ({nyq:g} Hz)")
    return float(cutoff)

def _butter_filter(data: np.ndarray, fs: float, cutoff: float, btype: str, order: int = 4) -> np.ndarray:
    fs = _validate_fs(fs)
    cutoff = _validate_cutoff(cutoff, fs, "cutoff")
    if data.size < max(3 * order, 15):
        raise FilterConfigurationError("signal is too short for stable zero-phase filtering")
    b, a = sps.butter(order, cutoff / (fs / 2), btype=btype)
    return sps.filtfilt(b, a, np.asarray(data, dtype=float))

def remove_baseline_wander(data: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    return _butter_filter(data, fs, cutoff_hz, "highpass")

def low_pass(data: np.ndarray, fs: float, cutoff_hz: float = 40.0) -> np.ndarray:
    return _butter_filter(data, fs, cutoff_hz, "lowpass")

def high_pass(data: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    return _butter_filter(data, fs, cutoff_hz, "highpass")

def notch_filter(data: np.ndarray, fs: float, freq_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    fs = _validate_fs(fs)
    freq_hz = _validate_cutoff(freq_hz, fs, "notch frequency")
    if q <= 0 or not np.isfinite(q):
        raise FilterConfigurationError("notch Q must be positive")
    if data.size < 15:
        raise FilterConfigurationError("signal is too short for stable zero-phase filtering")
    b, a = sps.iirnotch(freq_hz, q, fs)
    return sps.filtfilt(b, a, np.asarray(data, dtype=float))

def validate_filter_order(fs: float, highpass_hz: float | None, lowpass_hz: float | None, notch_hz: float | None) -> None:
    fs = _validate_fs(fs)
    nyq = fs / 2
    if highpass_hz is not None:
        _validate_cutoff(highpass_hz, fs, "high-pass cutoff")
    if lowpass_hz is not None:
        _validate_cutoff(lowpass_hz, fs, "low-pass cutoff")
    if notch_hz is not None:
        _validate_cutoff(notch_hz, fs, "notch frequency")
    if highpass_hz is not None and lowpass_hz is not None and highpass_hz >= lowpass_hz:
        raise FilterConfigurationError("high-pass cutoff must be lower than low-pass cutoff")
    if nyq <= 0:
        raise FilterConfigurationError("invalid Nyquist frequency")

def apply_pipeline(
    data: np.ndarray,
    fs: float,
    baseline: bool = False,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    notch_hz: float | None = None,
) -> np.ndarray:
    """Return a filtered copy; never mutates the raw input array."""
    fs = _validate_fs(fs)
    validate_filter_order(fs, highpass_hz, lowpass_hz, notch_hz)
    out = np.asarray(data, dtype=float).copy()
    if baseline:
        out = remove_baseline_wander(out, fs)
    if highpass_hz is not None:
        out = high_pass(out, fs, highpass_hz)
    if lowpass_hz is not None:
        out = low_pass(out, fs, lowpass_hz)
    if notch_hz is not None:
        out = notch_filter(out, fs, notch_hz)
    return out
