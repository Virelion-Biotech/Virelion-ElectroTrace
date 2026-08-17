"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features

DEFAULT_NEGATIVE_COUNT_RATIO = 0.70
DEFAULT_RECOVERY_GAP_RATIO = 1.65
DEFAULT_RECOVERY_PROMINENCE = 0.25


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


def _candidate_set(z: np.ndarray, fs_hz: float, scale: float, *, prominence_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    distance = max(1, int(round(fs_hz * 0.25)))
    peaks, properties = sps.find_peaks(z, distance=distance, prominence=scale * prominence_fraction)
    prominences = properties.get("prominences", np.zeros(len(peaks), dtype=float))
    return peaks.astype(int), np.asarray(prominences, dtype=float)


def select_signal_polarity(signal: np.ndarray, fs_hz: float) -> PolarityDecision:
    """Select one polarity per recording without merging positive/negative peaks."""
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
    """Run ElectroTrace's heuristic Stage-1 R-peak candidate detector."""
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


def recover_stage1_candidates(
    signal: np.ndarray,
    fs_hz: float,
    primary_peaks: np.ndarray,
    *,
    polarity: str = "positive",
    gap_ratio: float = DEFAULT_RECOVERY_GAP_RATIO,
    prominence_fraction: float = DEFAULT_RECOVERY_PROMINENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate at most one relaxed candidate inside unusually long RR gaps."""
    signal, fs_hz = _validate_signal(signal, fs_hz)
    peaks = np.asarray(primary_peaks, dtype=int)
    if peaks.ndim != 1:
        raise ValueError("primary_peaks must be one-dimensional")
    if peaks.size < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    if np.any(peaks[1:] <= peaks[:-1]) or np.any(peaks < 0) or np.any(peaks >= signal.size):
        raise ValueError("primary_peaks must be sorted and in range")
    if not np.isfinite(gap_ratio) or gap_ratio <= 1:
        raise ValueError("gap_ratio must be greater than one")
    if not np.isfinite(prominence_fraction) or prominence_fraction <= 0:
        raise ValueError("prominence_fraction must be positive")

    typical_rr = float(np.median(np.diff(peaks)))
    if not np.isfinite(typical_rr) or typical_rr <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    z = signal - np.median(signal)
    scale = float(np.std(z))
    if scale == 0 or not np.isfinite(scale):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    candidate_signal = z if polarity != "negative" else -z
    relaxed, relaxed_prom = _candidate_set(
        candidate_signal, fs_hz, scale, prominence_fraction=prominence_fraction
    )

    gaps = [(int(left), int(right)) for left, right in zip(peaks[:-1], peaks[1:])
            if (right - left) > gap_ratio * typical_rr]
    if not gaps or len(relaxed) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    selected: list[tuple[int, float]] = []
    for left, right in gaps:
        inside = [(int(idx), float(prom)) for idx, prom in zip(relaxed, relaxed_prom)
                  if left < idx < right]
        if inside:
            selected.append(max(inside, key=lambda pair: pair[1]))
    if not selected:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    selected.sort(key=lambda pair: pair[0])
    return np.asarray([idx for idx, _ in selected], dtype=int), np.asarray([prom for _, prom in selected], dtype=float)


def detect_r_peaks_two_stage(
    signal: np.ndarray,
    fs_hz: float,
    suppressor: CandidateSuppressor,
    *,
    threshold: float | None = None,
    polarity: str = "positive",
    recovery: bool = False,
    recovery_gap_ratio: float = DEFAULT_RECOVERY_GAP_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage 1, optional long-gap recovery, then the trained suppressor."""
    signal, fs_hz = _validate_signal(signal, fs_hz)
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted before two-stage detection")
    chosen_polarity = polarity
    if polarity == "adaptive":
        chosen_polarity = select_signal_polarity(signal, fs_hz).polarity

    primary_peaks = detect_r_peaks(signal, fs_hz, polarity=chosen_polarity)
    if len(primary_peaks) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    z = signal - np.median(signal)
    scale = float(np.std(z))
    candidate_signal = z if chosen_polarity != "negative" else -z
    primary_peaks, primary_prom = _candidate_set(candidate_signal, fs_hz, scale)
    all_peaks = primary_peaks
    all_prom = primary_prom

    if recovery:
        extra_peaks, extra_prom = recover_stage1_candidates(
            signal, fs_hz, primary_peaks,
            polarity=chosen_polarity,
            gap_ratio=recovery_gap_ratio,
        )
        if len(extra_peaks):
            all_peaks = np.sort(np.concatenate([primary_peaks, extra_peaks]))
            prom_map = {int(idx): float(prom) for idx, prom in zip(primary_peaks, primary_prom)}
            prom_map.update({int(idx): float(prom) for idx, prom in zip(extra_peaks, extra_prom)})
            all_prom = np.asarray([prom_map[int(idx)] for idx in all_peaks], dtype=float)

    features, _ = _candidate_features(signal, fs_hz, all_peaks, all_prom)
    retained, probabilities = suppressor.filter_candidates(all_peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
