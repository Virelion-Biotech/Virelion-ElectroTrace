"""Classical open R-peak detectors for locked baseline comparison.

Implementations follow the classic literature at a research level:
- Pan & Tompkins (1985): bandpass → derivative → square → moving integrate → adaptive threshold
- Hamilton & Tompkins (1986): similar pipeline with Hamilton-style peak decision rules

These are retrospective full-record detectors (same evaluation mode as ElectroTrace
Stage-1) and are intended only for protocol-matched comparison on MIT-BIH.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _validate(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, float]:
    x = np.asarray(signal, dtype=float)
    fs = float(fs_hz)
    if x.ndim != 1 or x.size < 32:
        raise ValueError("signal must be one-dimensional with at least 32 samples")
    if not np.isfinite(x).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs_hz must be positive and finite")
    return x, fs


def _bandpass(x: np.ndarray, fs_hz: float, low: float = 5.0, high: float = 15.0) -> np.ndarray:
    nyq = fs_hz / 2.0
    low = min(low, nyq * 0.45)
    high = min(high, nyq * 0.9)
    if high <= low or low <= 0:
        return x - np.median(x)
    sos = sps.butter(2, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    try:
        return sps.sosfiltfilt(sos, x - np.median(x))
    except ValueError:
        return sps.sosfilt(sos, x - np.median(x))


def pan_tompkins_r_peaks(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Pan–Tompkins style R-peak detector (full-record retrospective)."""
    x, fs = _validate(signal, fs_hz)
    band = _bandpass(x, fs, 5.0, 15.0)
    deriv = np.convolve(band, np.array([1, 2, 0, -2, -1], dtype=float) / 8.0, mode="same") * fs
    squared = deriv ** 2
    win = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win, dtype=float) / win, mode="same")

    init_n = min(len(integrated), int(2.0 * fs))
    peak_level = float(np.max(integrated[:init_n])) if init_n else 0.0
    noise_level = float(np.median(integrated[:init_n])) if init_n else 0.0
    threshold = noise_level + 0.25 * (peak_level - noise_level)

    min_distance = max(1, int(round(0.2 * fs)))
    candidates, props = sps.find_peaks(integrated, distance=min_distance, height=threshold * 0.5)
    if len(candidates) == 0:
        return np.asarray([], dtype=int)

    peaks: list[int] = []
    last = -min_distance
    for idx in candidates:
        val = float(integrated[idx])
        if idx - last < min_distance:
            continue
        if val >= threshold:
            half = max(1, int(round(0.075 * fs)))
            lo = max(0, int(idx) - half)
            hi = min(len(band), int(idx) + half + 1)
            local = band[lo:hi]
            if local.size == 0:
                continue
            local_peak = lo + int(np.argmax(np.abs(local)))
            peaks.append(local_peak)
            last = idx
            peak_level = 0.125 * val + 0.875 * peak_level
            threshold = noise_level + 0.25 * (peak_level - noise_level)
        else:
            noise_level = 0.125 * val + 0.875 * noise_level
            threshold = noise_level + 0.25 * (peak_level - noise_level)

    return np.asarray(peaks, dtype=int)


def hamilton_r_peaks(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Hamilton–Tompkins style R-peak detector (full-record retrospective)."""
    x, fs = _validate(signal, fs_hz)
    band = _bandpass(x, fs, 8.0, 16.0)
    deriv = np.diff(band, prepend=band[0]) * fs
    squared = deriv ** 2
    win = max(1, int(round(0.080 * fs)))
    integrated = np.convolve(squared, np.ones(win, dtype=float) / win, mode="same")

    min_distance = max(1, int(round(0.2 * fs)))
    med = float(np.median(integrated))
    mad = float(np.median(np.abs(integrated - med)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(integrated) + 1e-12)
    height = med + 2.5 * scale

    candidates, _ = sps.find_peaks(integrated, distance=min_distance, height=height)
    if len(candidates) == 0:
        candidates, _ = sps.find_peaks(integrated, distance=min_distance, height=med + 1.0 * scale)

    peaks: list[int] = []
    half = max(1, int(round(0.06 * fs)))
    for idx in candidates:
        lo = max(0, int(idx) - half)
        hi = min(len(band), int(idx) + half + 1)
        local = band[lo:hi]
        if local.size == 0:
            continue
        local_peak = lo + int(np.argmax(np.abs(local)))
        peaks.append(local_peak)
    if not peaks:
        return np.asarray([], dtype=int)
    return np.asarray(sorted(set(peaks)), dtype=int)


BASELINE_DETECTORS = {
    "pan_tompkins": pan_tompkins_r_peaks,
    "hamilton": hamilton_r_peaks,
}
