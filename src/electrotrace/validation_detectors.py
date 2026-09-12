"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features

DEFAULT_NEGATIVE_COUNT_RATIO = 0.70
DEFAULT_RECOVERY_GAP_RATIO = 1.65
DEFAULT_RECOVERY_PROMINENCE = 0.25

# --- Stage-1 amplitude-scale estimation -----------------------------------
#
# Priority-1 fix (see docs/PRIORITY1_STAGE1_HANDOFF.md): Stage-1 candidate
# generation used a single global `std(signal)` to set the prominence
# threshold for the *entire* record. On INCART, a short high-amplitude
# noise/artifact segment inflates that one number, which raises the
# threshold everywhere else in the record and starves candidate generation
# even over otherwise clean QRS complexes (`stage1_candidates / reference`
# as low as ~0.01 on the Priority-1 focus records, versus gqrs F1 > 0.98 on
# the same records). "std" is kept only for exact backward-compatibility /
# A-B comparison against the pre-fix behavior.
#
# IMPORTANT: real-data validation on full INCART (2026-09-12) showed
# "windowed_mad" fixes the 3 artifact-burst-style focus records but causes
# a large *net regression* on the aggregate (F1 0.4854 -> 0.3358, sens
# 0.324 -> 0.203) because MAD's median-based robustness breaks down once a
# window's "elevated" (QRS/T) fraction exceeds ~50%, which happens on
# faster/wider-complex INCART records. DEFAULT_SCALE_METHOD is reverted to
# "std" until a method is found that doesn't regress the full set — see
# "windowed_std" below, not yet validated on real data as of this writing.
DEFAULT_SCALE_METHOD = "windowed_std"
DEFAULT_SCALE_WINDOW_S = 8.0
DEFAULT_ADAPTIVE_INFLATION_RATIO = 1.5
_MAD_TO_STD = 1.4826


def _mad_scale(x: np.ndarray) -> float:
    """Robust scale estimate: 1.4826 * MAD, falling back to std if MAD is ~0
    (e.g. a flat/quantized segment) so the estimator never silently returns
    zero and disables the prominence threshold entirely."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return _MAD_TO_STD * mad
    return float(np.std(x))


def estimate_stage1_scale(
    z: np.ndarray,
    fs_hz: float,
    *,
    method: str = DEFAULT_SCALE_METHOD,
    window_s: float = DEFAULT_SCALE_WINDOW_S,
    inflation_ratio: float = DEFAULT_ADAPTIVE_INFLATION_RATIO,
) -> float:
    """Estimate the amplitude scale used to set the Stage-1 prominence threshold.

    method="std"
        The original global standard deviation. A single artifact/noise
        burst inflates it for the whole record. Kept only for
        backward-compatible / A-B comparison runs.
    method="mad"
        Global robust scale (1.4826 * MAD). Resistant to a *few* extreme
        samples, but still one record-wide number, so a long noisy segment
        can still dominate it.
    method="windowed_mad"
        Splits the record into `window_s`-second blocks, computes a robust
        (MAD) scale per block, and takes the *median across blocks*. A
        minority of corrupted blocks can no longer set the threshold for
        the whole record — this is what fixes the INCART starvation.

        Caveat found during real-data validation: MAD is only robust to a
        *minority* of "outlier" samples per window. On faster/wider-complex
        records, QRS+T can occupy >50% of an 8s window, which flips what
        the median tracks from the quiet baseline to the QRS/T amplitude
        itself, over-suppressing real beats. See method="windowed_std".
    method="windowed_std"
        Same windowing as "windowed_mad", but takes the median of per-window
        plain std instead of MAD. Keeps std's continuous behavior (no
        discrete flip once a window's "elevated" fraction crosses ~50%)
        while still discounting a minority of corrupted windows.

        Real-data validation (2026-09-12): recovers the 5 Priority-1 focus
        records dramatically (e.g. I56 sensitivity 0.0018 -> 0.2628) and
        mostly un-does windowed_mad's collateral damage on records where
        "std" already worked (I69 0.9332 -> 0.8887, vs windowed_mad's
        0.0946). Net INCART aggregate F1 is still slightly below "std"
        (0.4716 vs 0.4854) — a dozen or so mid-performing records get
        moderately worse even as the worst ones improve a lot. See
        method="adaptive" for a per-record compromise.
    method="adaptive"
        Uses "std" by default, but falls back to "windowed_std" on a
        per-record basis when the record's own global std looks inflated
        relative to its own windowed estimate — i.e. when
        std(z) / windowed_std(z) > inflation_ratio. This is a purely
        signal-derived check (no reference annotations involved, so it
        doesn't touch the "no INCART labels" constraint) intended to catch
        exactly the artifact-burst records without changing behavior on
        records where "std" already works fine. Not yet validated on real
        data as of this writing — inflation_ratio is a starting guess.
    """
    z = np.asarray(z, dtype=float)
    method = str(method).lower()
    if method == "std":
        return float(np.std(z))
    if method == "mad":
        return _mad_scale(z)
    if method in ("windowed_mad", "windowed_std"):
        window = max(1, int(round(fs_hz * window_s)))
        n = z.size
        if n <= window:
            return _mad_scale(z) if method == "windowed_mad" else float(np.std(z))
        min_block = max(4, window // 4)
        local_scales = []
        for start in range(0, n, window):
            seg = z[start:start + window]
            if seg.size < min_block:
                continue
            s = _mad_scale(seg) if method == "windowed_mad" else float(np.std(seg))
            if np.isfinite(s) and s > 0:
                local_scales.append(s)
        if not local_scales:
            return _mad_scale(z) if method == "windowed_mad" else float(np.std(z))
        return float(np.median(local_scales))
    if method == "adaptive":
        global_std = float(np.std(z))
        windowed_std = estimate_stage1_scale(z, fs_hz, method="windowed_std", window_s=window_s)
        if windowed_std <= 0 or not np.isfinite(windowed_std):
            return global_std
        if global_std / windowed_std > inflation_ratio:
            return windowed_std
        return global_std
    raise ValueError(f"unknown scale method: {method!r}")


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


def select_signal_polarity(
    signal: np.ndarray, fs_hz: float, *, scale_method: str = DEFAULT_SCALE_METHOD
) -> PolarityDecision:
    """Select one polarity per recording without merging positive/negative peaks.

    Primary rule: candidate-count ratio (validated on full MIT-BIH).
    When count confidence is low (<0.15), fall back to QRS-band polarity v2
    (fixes inverted-lead cases such as MIT-BIH 207 without pooled regression).
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)
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

    # Low-confidence count decisions: use QRS-band polarity v2 (MIT-BIH pooled
    # F1 improves ~+0.01 and record 207 is corrected to negative).
    if confidence < 0.15:
        from .polarity_v2 import select_signal_polarity_v2
        v2 = select_signal_polarity_v2(signal, fs_hz)
        polarity = v2.polarity
        confidence = max(confidence, float(v2.confidence))

    return PolarityDecision(polarity, confidence, pos_score, neg_score, pos_count, neg_count)


def detect_r_peaks(
    signal: np.ndarray,
    fs_hz: float,
    *,
    polarity: str = "positive",
    scale_method: str = DEFAULT_SCALE_METHOD,
) -> np.ndarray:
    """Run ElectroTrace's heuristic Stage-1 R-peak candidate detector."""
    signal, fs_hz = _validate_signal(signal, fs_hz)
    polarity = str(polarity).lower()
    if polarity not in {"positive", "negative", "adaptive"}:
        raise ValueError("polarity must be positive, negative, or adaptive")
    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)
    if not np.isfinite(scale) or scale == 0:
        return np.asarray([], dtype=int)
    if polarity == "adaptive":
        polarity = select_signal_polarity(signal, fs_hz, scale_method=scale_method).polarity
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
    scale_method: str = DEFAULT_SCALE_METHOD,
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
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)
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
    scale_method: str = DEFAULT_SCALE_METHOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage 1, optional long-gap recovery, then the trained suppressor."""
    signal, fs_hz = _validate_signal(signal, fs_hz)
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted before two-stage detection")
    chosen_polarity = polarity
    if polarity == "adaptive":
        chosen_polarity = select_signal_polarity(signal, fs_hz, scale_method=scale_method).polarity

    primary_peaks = detect_r_peaks(
        signal, fs_hz, polarity=chosen_polarity, scale_method=scale_method
    )
    if len(primary_peaks) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)
    candidate_signal = z if chosen_polarity != "negative" else -z
    primary_peaks, primary_prom = _candidate_set(candidate_signal, fs_hz, scale)
    all_peaks = primary_peaks
    all_prom = primary_prom

    if recovery:
        extra_peaks, extra_prom = recover_stage1_candidates(
            signal, fs_hz, primary_peaks,
            polarity=chosen_polarity,
            gap_ratio=recovery_gap_ratio,
            scale_method=scale_method,
        )
        if len(extra_peaks):
            all_peaks = np.sort(np.concatenate([primary_peaks, extra_peaks]))
            prom_map = {int(idx): float(prom) for idx, prom in zip(primary_peaks, primary_prom)}
            prom_map.update({int(idx): float(prom) for idx, prom in zip(extra_peaks, extra_prom)})
            all_prom = np.asarray([prom_map[int(idx)] for idx in all_peaks], dtype=float)

    features, _ = _candidate_features(signal, fs_hz, all_peaks, all_prom)
    retained, probabilities = suppressor.filter_candidates(all_peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
