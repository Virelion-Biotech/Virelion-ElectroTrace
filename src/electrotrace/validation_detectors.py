"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features


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
    """Choose the stronger ECG polarity without merging positive/negative candidates.

    The score combines robust prominence, plausible beat-count support, and RR
    regularity. Low-confidence decisions fall back to positive polarity so the
    historical detector behavior remains the safe default.
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    z = signal - np.median(signal)
    scale = float(np.std(z))
    if not np.isfinite(scale) or scale == 0:
        return PolarityDecision("positive", 0.0, 0.0, 0.0, 0, 0)

    pos, pos_prom = _candidate_set(z, fs_hz, scale)
    neg, neg_prom = _candidate_set(-z, fs_hz, scale)
    duration_s = signal.size / fs_hz

    def score(peaks: np.ndarray, prominences: np.ndarray) -> float:
        if len(peaks) == 0:
            return 0.0
        prominence_term = float(np.median(prominences) / max(scale, 1e-8))
        bpm = len(peaks) / max(duration_s, 1e-8) * 60.0
        if 35.0 <= bpm <= 180.0:
            count_term = 1.0
        elif bpm < 35.0:
            count_term = max(0.0, bpm / 35.0)
        else:
            count_term = max(0.0, 180.0 / bpm)
        if len(peaks) >= 3:
            rr = np.diff(peaks) / fs_hz
            rr_median = float(np.median(rr))
            rr_cv = float(np.std(rr) / max(rr_median, 1e-8)) if rr_median > 0 else 1.0
            regularity_term = 1.0 / (1.0 + rr_cv)
        else:
            regularity_term = 0.5
        return float(0.55 * prominence_term + 0.20 * count_term + 0.25 * regularity_term)

    pos_score = score(pos, pos_prom)
    neg_score = score(neg, neg_prom)
    if pos_score == neg_score == 0:
        return PolarityDecision("positive", 0.0, pos_score, neg_score, len(pos), len(neg))
    polarity = "positive" if pos_score >= neg_score else "negative"
    best = max(pos_score, neg_score)
    second = min(pos_score, neg_score)
    confidence = float((best - second) / max(best, 1e-8))
    if confidence < 0.10:
        polarity = "positive"
    return PolarityDecision(polarity, confidence, pos_score, neg_score, len(pos), len(neg))


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
