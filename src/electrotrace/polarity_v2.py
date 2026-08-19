"""Signal-only QRS-specific polarity selection (experimental v2).

The selector is deliberately separate from the locked primary detector.
It follows a QRS-first strategy: identify QRS-like events from steepness/
energy, then infer the dominant signed R-wave direction inside those events.
No reference annotations are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps


POLARITY_V2_VERSION = "qrs-polarity-v2"
DEFAULT_QRS_LOW_HZ = 5.0
DEFAULT_QRS_HIGH_HZ = 20.0
DEFAULT_WINDOW_S = 0.20
DEFAULT_MIN_DISTANCE_S = 0.25
DEFAULT_MIN_QRS_EVENTS = 8


@dataclass(frozen=True)
class PolarityV2Decision:
    polarity: str
    confidence: float
    positive_score: float
    negative_score: float
    qrs_events: int
    positive_events: int
    negative_events: int
    ambiguous_events: int
    rr_regularity: float
    qrs_band_energy_ratio: float


def _validate_signal(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, float]:
    x = np.asarray(signal, dtype=float)
    fs = float(fs_hz)
    if x.ndim != 1 or x.size < 32:
        raise ValueError("signal must be one-dimensional with at least 32 samples")
    if not np.isfinite(x).all():
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs_hz must be positive and finite")
    return x, fs


def _bandpass(x: np.ndarray, fs_hz: float) -> np.ndarray:
    nyq = fs_hz / 2.0
    low = min(DEFAULT_QRS_LOW_HZ, nyq * 0.4)
    high = min(DEFAULT_QRS_HIGH_HZ, nyq * 0.9)
    if high <= low or low <= 0:
        return x - np.median(x)
    sos = sps.butter(3, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    try:
        return sps.sosfiltfilt(sos, x - np.median(x))
    except ValueError:
        return sps.sosfilt(sos, x - np.median(x))


def _robust_scale(x: np.ndarray) -> float:
    mad = float(np.median(np.abs(x - np.median(x))))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(x))
    return max(scale, 1e-8)


def _qrs_events(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band = _bandpass(x, fs_hz)
    derivative = np.abs(np.gradient(band) * fs_hz)
    smooth_n = max(3, int(round(0.04 * fs_hz)))
    kernel = np.ones(smooth_n, dtype=float) / smooth_n
    envelope = np.convolve(derivative, kernel, mode="same")
    scale = _robust_scale(envelope)
    distance = max(1, int(round(DEFAULT_MIN_DISTANCE_S * fs_hz)))
    peaks, props = sps.find_peaks(envelope, distance=distance, prominence=0.25 * scale)
    prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
    return peaks.astype(int), prom, band


def select_signal_polarity_v2(signal: np.ndarray, fs_hz: float) -> PolarityV2Decision:
    """Choose polarity using QRS-like steepness events and local signed morphology."""
    x, fs = _validate_signal(signal, fs_hz)
    qrs_peaks, qrs_prom, band = _qrs_events(x, fs)
    if len(qrs_peaks) == 0:
        return PolarityV2Decision("positive", 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0)

    radius = max(8, int(round(DEFAULT_WINDOW_S * fs / 2.0)))
    signed_scores: list[float] = []
    event_signs: list[int] = []
    band_energy: list[float] = []
    for idx in qrs_peaks:
        lo = max(0, int(idx) - radius)
        hi = min(len(x), int(idx) + radius + 1)
        local = x[lo:hi]
        local_center = float(np.median(local))
        centered = local - local_center
        pos_amp = float(np.max(centered)) if centered.size else 0.0
        neg_amp = float(np.max(-centered)) if centered.size else 0.0
        pos_score = pos_amp / max(_robust_scale(local), 1e-8)
        neg_score = neg_amp / max(_robust_scale(local), 1e-8)
        denom = max(pos_score + neg_score, 1e-8)
        directional = (pos_score - neg_score) / denom
        signed_scores.append(directional)
        if directional > 0.12:
            event_signs.append(1)
        elif directional < -0.12:
            event_signs.append(-1)
        else:
            event_signs.append(0)
        band_energy.append(float(np.mean(band[lo:hi] ** 2)))

    signs = np.asarray(event_signs, dtype=int)
    pos = int(np.sum(signs == 1))
    neg = int(np.sum(signs == -1))
    ambiguous = int(np.sum(signs == 0))
    decisive = max(pos + neg, 1)
    pos_weight = float(np.mean([max(v, 0.0) for v in signed_scores]))
    neg_weight = float(np.mean([max(-v, 0.0) for v in signed_scores]))

    intervals = np.diff(qrs_peaks) / fs if len(qrs_peaks) > 1 else np.asarray([], dtype=float)
    if intervals.size:
        cv = float(np.std(intervals) / max(np.mean(intervals), 1e-8))
        rr_regularity = float(1.0 / (1.0 + cv))
    else:
        rr_regularity = 0.0

    if pos + neg < DEFAULT_MIN_QRS_EVENTS:
        return PolarityV2Decision("positive", 0.0, pos_weight, neg_weight, len(qrs_peaks), pos, neg, ambiguous, rr_regularity, 0.0)

    positive_score = (pos / decisive) * (0.5 + 0.5 * rr_regularity) + 0.5 * pos_weight
    negative_score = (neg / decisive) * (0.5 + 0.5 * rr_regularity) + 0.5 * neg_weight
    margin = abs(positive_score - negative_score) / max(positive_score + negative_score, 1e-8)

    if margin < 0.08 or pos == neg:
        polarity = "positive" if positive_score >= negative_score else "negative"
        confidence = margin * 0.5
    else:
        polarity = "positive" if positive_score > negative_score else "negative"
        confidence = margin

    positive_energy = float(np.mean([e for e, s in zip(band_energy, signs) if s == 1])) if pos else 0.0
    negative_energy = float(np.mean([e for e, s in zip(band_energy, signs) if s == -1])) if neg else 0.0
    qrs_ratio = positive_energy / max(negative_energy, 1e-8) if positive_energy and negative_energy else 0.0

    return PolarityV2Decision(
        polarity,
        float(confidence),
        float(positive_score),
        float(negative_score),
        int(len(qrs_peaks)),
        pos,
        neg,
        ambiguous,
        float(rr_regularity),
        float(qrs_ratio),
    )
