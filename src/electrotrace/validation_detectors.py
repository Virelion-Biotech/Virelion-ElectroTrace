"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features


def detect_r_peaks(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Run ElectroTrace's current heuristic R-peak candidate detector."""
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


def detect_r_peaks_two_stage(
    signal: np.ndarray,
    fs_hz: float,
    suppressor: CandidateSuppressor,
    *,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage 1 candidate detection followed by a trained suppressor.

    Returns ``(retained_peak_indices, candidate_probabilities)``. The probability
    array corresponds to the complete Stage-1 candidate list, which is useful for
    QC and audit trails.
    """
    signal = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if signal.ndim != 1 or signal.size < 8 or not np.isfinite(signal).all():
        raise ValueError("signal must be one-dimensional, contain at least eight finite samples")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted before two-stage detection")

    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    peaks, properties = sps.find_peaks(
        z,
        distance=max(1, int(round(fs_hz * 0.25))),
        prominence=scale * 0.5,
    )
    features, _ = _candidate_features(signal, fs_hz, peaks, properties.get("prominences"))
    retained, probabilities = suppressor.filter_candidates(peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
