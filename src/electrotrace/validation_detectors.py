"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features


def detect_r_peaks(signal: np.ndarray, fs_hz: float, *, polarity: str = "positive") -> np.ndarray:
    """Run ElectroTrace's heuristic Stage-1 R-peak candidate detector.

    ``polarity="positive"`` preserves the historical behavior. ``polarity="adaptive"``
    detects both positive and negative deflections and merges nearby candidates,
    allowing the second stage to decide which morphology is a true beat.
    """
    signal = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if signal.ndim != 1 or signal.size < 8:
        raise ValueError("signal must be one-dimensional with at least eight samples")
    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    polarity = str(polarity).lower()
    if polarity not in {"positive", "negative", "adaptive"}:
        raise ValueError("polarity must be positive, negative, or adaptive")
    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return np.asarray([], dtype=int)
    distance = max(1, int(round(fs_hz * 0.25)))

    def _find(x: np.ndarray):
        return sps.find_peaks(x, distance=distance, prominence=scale * 0.5)

    if polarity == "positive":
        peaks, _ = _find(z)
        return peaks.astype(int)
    if polarity == "negative":
        peaks, _ = _find(-z)
        return peaks.astype(int)

    pos, pos_props = _find(z)
    neg, neg_props = _find(-z)
    all_peaks = np.concatenate([pos, neg]).astype(int)
    all_prom = np.concatenate([pos_props.get("prominences", np.zeros(len(pos))),
                               neg_props.get("prominences", np.zeros(len(neg)))])
    if len(all_peaks) == 0:
        return np.asarray([], dtype=int)
    order = np.argsort(all_peaks)
    all_peaks = all_peaks[order]
    all_prom = all_prom[order]
    selected: list[int] = []
    selected_prom: list[float] = []
    for peak, prom in zip(all_peaks, all_prom):
        if not selected or int(peak) - selected[-1] >= distance:
            selected.append(int(peak))
            selected_prom.append(float(prom))
        elif prom > selected_prom[-1]:
            selected[-1] = int(peak)
            selected_prom[-1] = float(prom)
    return np.asarray(selected, dtype=int)


def detect_r_peaks_two_stage(
    signal: np.ndarray,
    fs_hz: float,
    suppressor: CandidateSuppressor,
    *,
    threshold: float | None = None,
    polarity: str = "positive",
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
    peaks = detect_r_peaks(signal, fs_hz, polarity=polarity)
    if len(peaks) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    _, properties = sps.find_peaks(
        signal - np.median(signal),
        distance=max(1, int(round(fs_hz * 0.25))),
        prominence=float(np.std(signal - np.median(signal))) * 0.5,
    )
    # For adaptive/negative candidates, recompute candidate-specific prominence
    # directly rather than relying on positive-only peak properties.
    z = signal - np.median(signal)
    prom = np.zeros(len(peaks), dtype=float)
    for i, peak in enumerate(peaks):
        prom[i] = float(abs(z[peak]))
    features, _ = _candidate_features(signal, fs_hz, peaks, prom)
    retained, probabilities = suppressor.filter_candidates(peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
