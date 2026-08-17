"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features

DEFAULT_NEGATIVE_COUNT_RATIO = 0.70


@dataclass(frozen=True)
class PolarityDecision:
    polarity: str
    confidence: float
    positive_score: float
    negative_score: float
    positive_candidates: int
    negative_candidates: int


def _validate_signal(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=float)
    fs_hz = float(fs_hz)
    if signal.ndim != 1 or signal.size < 8:
        raise ValueError("signal must be one-dimensional with at least eight samples")
    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    return signal, fs_hz


def _candidate_set(z: np.ndarray, fs_hz: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    distance = max(1, int(round(fs_hz * 0.25)))
    peaks, properties = sps.find_peaks(z, distance=distance, prominence=scale * 0.5)
    prominences = properties.get("prominences", np.zeros(len(peaks), dtype=float))
    return peaks.astype(int), np.asarray(prominences, dtype=float)


def select_signal_polarity(signal: np.ndarray, fs_hz: float) -> PolarityDecision:
    """Select one polarity per recording without merging positive/negative peaks.

    The negative-polarity path is deliberately conservative: it is selected only
    when it produces substantially fewer Stage-1 candidates than the positive path.
    This is a signal-level heuristic, not a clinical ECG-quality classifier.
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return PolarityDecision("positive", 0.0, 0.0, 0.0, 0, 0)

    pos, pos_prom = _candidate_set(z, fs_hz, scale)
    neg, neg_prom = _candidate_set(-z, fs_hz, scale)

    def morphology_score(prominences: np.ndarray) -> float:
        if len(prominences) == 0:
            return 0.0
        return float(np.median(prominences) / max(scale, 1e-8))

    pos_score = morphology_score(pos_prom)
    neg_score = morphology_score(neg_prom)
    pos_count = len(pos)
    neg_count = len(neg)
    ratio = neg_count / max(pos_count, 1)
    polarity = "negative" if pos_count > 0 and neg_count > 0 and ratio < DEFAULT_NEGATIVE_COUNT_RATIO else "positive"
    confidence = float(abs(pos_count - neg_count) / max(pos_count, neg_count, 1))
    return PolarityDecision(polarity, confidence, pos_score, neg_score, pos_count, neg_count)


def detect_r_peaks(signal: np.ndarray, fs_hz: float, *, polarity: str = "positive") -> np.ndarray:
    """Run ElectroTrace's heuristic Stage-1 R-peak candidate detector.

    ``polarity="positive"`` preserves historical behavior. ``negative`` detects
    inverted deflections. ``adaptive`` selects one polarity per recording using
    :func:`select_signal_polarity`; it does not merge both polarities.
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    polarity = str(polarity).lower()
    if polarity not in {"positive", "negative", "adaptive"}:
        raise ValueError("polarity must be positive, negative, or adaptive")
    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return np.asarray([], dtype=int)
    if polarity == "adaptive":
        polarity = select_signal_polarity(signal, fs_hz).polarity
    peaks, _ = _candidate_set(z if polarity == "positive" else -z, fs_hz, scale)
    return peaks


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
    array corresponds to the complete Stage-1 candidate list, useful for QC and
    audit trails.
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted before two-stage detection")
    chosen_polarity = polarity
    if polarity == "adaptive":
        chosen_polarity = select_signal_polarity(signal, fs_hz).polarity
    peaks = detect_r_peaks(signal, fs_hz, polarity=chosen_polarity)
    if len(peaks) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    z = signal - np.median(signal)
    scale = float(np.std(z))
    candidate_signal = z if chosen_polarity == "positive" else -z
    _, prominences = _candidate_set(candidate_signal, fs_hz, scale)
    if len(prominences) != len(peaks):
        prominences = np.abs(candidate_signal[peaks])
    features, _ = _candidate_features(signal, fs_hz, peaks, prominences)
    retained, probabilities = suppressor.filter_candidates(peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
