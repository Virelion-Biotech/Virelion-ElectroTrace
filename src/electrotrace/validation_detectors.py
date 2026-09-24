"""Reference detector adapters used by the external validation harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .candidate_suppressor import CandidateSuppressor, _candidate_features
from .scale_estimation import (
    DEFAULT_ADAPTIVE_INFLATION_RATIO,
    DEFAULT_SCALE_METHOD,
    DEFAULT_SCALE_WINDOW_S,
    estimate_scale,
    estimate_stage1_scale,
)

DEFAULT_NEGATIVE_COUNT_RATIO = 0.70
DEFAULT_RECOVERY_GAP_RATIO = 1.65
DEFAULT_RECOVERY_PROMINENCE = 0.25
DEFAULT_DUAL_POLARITY_MERGE_WINDOW_S = 0.10
MERGE_FEATURE_SCOPES = ("per_stream", "pooled")
MERGE_SCOPES = ("all", "gaps")
# Opposite-polarity candidates must rise this many scale-units above the median
# baseline (in their own polarity). See _score_dual_polarity_streams.
DEFAULT_MINORITY_MIN_PEAK_SCALE = 0.25

# Scale estimation (DEFAULT_SCALE_METHOD, estimate_stage1_scale, etc.) now
# lives in scale_estimation.py, imported above -- Stage 2's feature
# normalization (candidate_suppressor.py) needed the same fix (see that
# module's docstring for the full history/numbers), and validation_detectors.py
# already imports FROM candidate_suppressor.py, so the shared logic had to
# move to a third module to avoid a circular import.


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


DEFAULT_WIDTH_OVERRIDE_CONFIDENCE = 0.38
DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES = 3


def _width_preferred_polarity(
    z: np.ndarray, pos: np.ndarray, neg: np.ndarray, *, min_candidates: int = DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES
) -> str | None:
    """Given z-signal and each polarity's candidate indices, return whichever
    polarity has the narrower (more QRS-like) median candidate width, or
    None if either candidate set is too small to judge. Split out from
    select_signal_polarity purely so this specific comparison can be unit
    tested directly with hand-built candidate arrays, without needing a
    full synthetic signal to naturally land in the right confidence band."""
    if len(pos) < min_candidates or len(neg) < min_candidates:
        return None
    from scipy.signal import peak_widths
    pos_median_width = float(np.median(peak_widths(z, pos, rel_height=0.5)[0]))
    neg_median_width = float(np.median(peak_widths(-z, neg, rel_height=0.5)[0]))
    return "positive" if pos_median_width <= neg_median_width else "negative"


def select_signal_polarity(
    signal: np.ndarray, fs_hz: float, *, scale_method: str = DEFAULT_SCALE_METHOD,
    width_override_confidence: float = DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
) -> PolarityDecision:
    """Select one polarity per recording without merging positive/negative peaks.

    Primary rule: candidate-count ratio (validated on full MIT-BIH).
    When count confidence is low (<0.15), fall back to QRS-band polarity v2
    (fixes inverted-lead cases such as MIT-BIH 207 without pooled regression).

    Real-data validation on full INCART (2026-09-13): the count-ratio rule
    picks correctly on 61/68 records; oracle ceiling (always picking
    whichever polarity is actually better) is only 1.4 F1 points above the
    current heuristic, and the errors go in *both* directions (6 records
    want "negative" more readily chosen, 1 record -- I19 -- wants it chosen
    much less readily), so no single global threshold change on the ratio
    rule helps both without hurting the other. I19 alone accounts for most
    of the gap (F1 0.19 vs an achievable 0.70): its wrong polarity produces
    candidates with implausible median width (~640ms, when even wide PVCs
    top out around 150-160ms) -- the wrong polarity isn't detecting
    inverted QRS complexes, it's detecting something else (T-waves, baseline
    humps) that happens to clear the prominence threshold.

    A confidence-gated width check recovers most of this without the
    ratio-threshold's directional trade-off: when confidence is already low
    (< width_override_confidence), and the two polarities' candidate widths
    differ enough to matter, prefer whichever has the narrower (more
    QRS-like) median width. Tested against all 68 INCART records: mean F1
    0.8169 (current) -> 0.8270 (confidence-gated width check) -> 0.8311
    (oracle). An UNGATED width check (always overriding, regardless of
    confidence) reaches only 0.8234 and causes a real regression on I63
    (F1 0.8992 -> 0.6353, a high-confidence record where the ratio rule was
    already correct) -- gating by confidence avoids exactly that case while
    still fixing I19, I13, and I64.
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

    # Confidence-gated width check (see docstring above for validation numbers).
    if confidence < width_override_confidence:
        width_preferred = _width_preferred_polarity(z, pos, neg)
        if width_preferred is not None and width_preferred != polarity:
            polarity = width_preferred

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


def _merge_dual_polarity_candidates(
    peaks: np.ndarray,
    probabilities: np.ndarray,
    merge_window_s: float,
    fs_hz: float,
    prominences: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NMS over RF-retained candidates from both polarity streams: within
    `merge_window_s`, keep only the highest-probability candidate. Returns
    (peaks, probabilities) sorted by sample index. Note this only removes
    Q/R/S-scale double counts (~30-100 ms); T-waves sit ~150-250 ms after the
    QRS and must be rejected by the classifier, not by this window."""
    peaks = np.asarray(peaks, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if peaks.size == 0:
        return peaks, probabilities
    window = max(1, int(round(float(merge_window_s) * float(fs_hz))))
    # RF probabilities saturate (many candidates at exactly 1.0), so break ties
    # by Stage-1 prominence instead of by sample order.
    if prominences is None:
        order = np.argsort(-probabilities, kind="stable")
    else:
        order = np.lexsort((-np.asarray(prominences, dtype=float), -probabilities))
    kept: list[int] = []
    kept_peaks = np.empty(0, dtype=int)
    for i in order:
        p = int(peaks[i])
        if kept_peaks.size and np.any(np.abs(kept_peaks - p) <= window):
            continue
        kept.append(int(i))
        kept_peaks = np.append(kept_peaks, p)
    kept_arr = np.asarray(sorted(kept, key=lambda i: int(peaks[i])), dtype=int)
    return peaks[kept_arr], probabilities[kept_arr]


def _score_dual_polarity_streams(
    signal: np.ndarray,
    fs_hz: float,
    suppressor: CandidateSuppressor,
    *,
    scale_method: str,
    feature_scope: str = "per_stream",
    minority_stream: int | None = None,
    minority_min_peak_scale: float = 0.0,
) -> dict[str, np.ndarray]:
    """Generate Stage-1 candidates under BOTH polarities and score them with the
    (unretrained) Stage-2 RF. Returns parallel arrays: peaks, probabilities,
    stream (0 = positive-going, 1 = negative-going), sorted by peak index.

    feature_scope controls how the RF's candidate-list-dependent features
    (rr_prev_s, rr_next_s, rr_prev_ratio, rr_next_ratio, and the record RR
    median they are normalised by) are computed:

    "per_stream" (default): features are extracted separately for each
        polarity's own candidate list -- exactly the distribution the RF was
        trained on (a single polarity stream, >= 250 ms apart by construction
        of _candidate_set). This is the corrected behaviour.
    "pooled": features are extracted on the sorted union of both streams.
        The RR features then describe distances to *opposite-polarity*
        neighbours (S-wave / T-wave / notch candidates 30-250 ms away) and
        rr_median collapses to roughly half the true RR, a feature
        distribution the RF never saw in training. Kept only to reproduce and
        A/B the original dual-polarity experiment.

    minority_stream / minority_min_peak_scale: scipy's prominence is measured
    down to the nearest HIGHER peak on each side. In the mirrored (-z) view of
    a positive-going record the QRS complexes are deep valleys, so the highest
    isoelectric-baseline noise maximum within a few beats inherits a huge
    "prominence" (~ the R-wave depth) despite sitting at baseline level. Such
    candidates -- like S-wave / ST / PR valleys -- never appear as labelled
    negatives in single-polarity training, so the RF has no basis for rejecting
    them. Candidates of the minority stream whose own-polarity peak value is
    below `minority_min_peak_scale * scale` are therefore dropped before
    scoring (0 disables).
    """
    if feature_scope not in MERGE_FEATURE_SCOPES:
        raise ValueError(f"feature_scope must be one of {MERGE_FEATURE_SCOPES}")
    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs_hz, method=scale_method)
    empty = {
        "peaks": np.empty(0, dtype=int), "probabilities": np.empty(0, dtype=float),
        "stream": np.empty(0, dtype=int), "prominences": np.empty(0, dtype=float),
    }
    if not np.isfinite(scale) or scale == 0:
        return empty
    streams = []
    for stream_id, sig in enumerate((z, -z)):
        peaks, prom = _candidate_set(sig, fs_hz, scale)
        if stream_id == minority_stream and minority_min_peak_scale > 0 and peaks.size:
            floor = float(minority_min_peak_scale) * scale
            keep_floor = sig[peaks] >= floor
            peaks, prom = peaks[keep_floor], prom[keep_floor]
        streams.append((stream_id, peaks, prom))

    if feature_scope == "pooled":
        peaks = np.concatenate([s[1] for s in streams])
        prom = np.concatenate([s[2] for s in streams])
        stream = np.concatenate([np.full(len(s[1]), s[0], dtype=int) for s in streams])
        if peaks.size == 0:
            return empty
        order = np.argsort(peaks, kind="stable")
        peaks, prom, stream = peaks[order], prom[order], stream[order]
        features, _ = _candidate_features(signal, fs_hz, peaks, prom, scale_method=scale_method)
        return {"peaks": peaks, "probabilities": suppressor.predict_proba(features), "stream": stream, "prominences": prom}

    out_peaks, out_prob, out_stream, out_prom = [], [], [], []
    for stream_id, peaks, prom in streams:
        if peaks.size == 0:
            continue
        features, _ = _candidate_features(signal, fs_hz, peaks, prom, scale_method=scale_method)
        out_peaks.append(peaks)
        out_prob.append(suppressor.predict_proba(features))
        out_stream.append(np.full(len(peaks), stream_id, dtype=int))
        out_prom.append(prom)
    if not out_peaks:
        return empty
    peaks = np.concatenate(out_peaks)
    order = np.argsort(peaks, kind="stable")
    return {
        "peaks": peaks[order], "probabilities": np.concatenate(out_prob)[order],
        "stream": np.concatenate(out_stream)[order], "prominences": np.concatenate(out_prom)[order],
    }


def _gap_fill_minority(
    majority_peaks: np.ndarray,
    minority_peaks: np.ndarray,
    minority_prob: np.ndarray,
    fs_hz: float,
    *,
    gap_ratio: float = DEFAULT_RECOVERY_GAP_RATIO,
    merge_window_s: float = DEFAULT_DUAL_POLARITY_MERGE_WINDOW_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept opposite-polarity candidates ONLY inside unusually long RR gaps of
    the majority-polarity beat train (e.g. the compensatory pause around a
    T-wave-discordant PVC whose QRS only exists in the other polarity).
    At most round(gap / typical_rr) - 1 candidates are taken per gap, best
    probability first, never within `merge_window_s` of a gap edge or of each
    other's window. Everything else from the minority stream is discarded, which
    is what keeps the false-positive flood of unrestricted merging out."""
    majority_peaks = np.asarray(majority_peaks, dtype=int)
    minority_peaks = np.asarray(minority_peaks, dtype=int)
    minority_prob = np.asarray(minority_prob, dtype=float)
    if majority_peaks.size < 3 or minority_peaks.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    diffs = np.diff(majority_peaks)
    typical_rr = float(np.median(diffs))
    if not np.isfinite(typical_rr) or typical_rr <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    guard = max(1, int(round(merge_window_s * fs_hz)))
    add_p: list[int] = []
    add_q: list[float] = []
    for left, right in zip(majority_peaks[:-1], majority_peaks[1:]):
        gap = int(right - left)
        if gap <= gap_ratio * typical_rr:
            continue
        budget = max(1, int(round(gap / typical_rr)) - 1)
        inside = np.flatnonzero((minority_peaks > left + guard) & (minority_peaks < right - guard))
        if inside.size == 0:
            continue
        taken: list[int] = []
        for j in inside[np.argsort(-minority_prob[inside], kind="stable")]:
            if len(taken) >= budget:
                break
            if all(abs(int(minority_peaks[j]) - t) > guard for t in taken):
                taken.append(int(minority_peaks[j]))
                add_p.append(int(minority_peaks[j]))
                add_q.append(float(minority_prob[j]))
    order = np.argsort(add_p, kind="stable") if add_p else np.empty(0, dtype=int)
    return np.asarray(add_p, dtype=int)[order], np.asarray(add_q, dtype=float)[order]


def _combine_scored_streams(
    scored: dict[str, np.ndarray],
    major_id: int,
    threshold: float,
    fs_hz: float,
    *,
    merge_scope: str = "gaps",
    merge_window_s: float = DEFAULT_DUAL_POLARITY_MERGE_WINDOW_S,
    minority_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn scored dual-polarity candidates into final (peaks, probabilities).
    Split out of detect_r_peaks_two_stage so diagnostics can sweep variants
    without re-running feature extraction / the RF."""
    if merge_scope not in MERGE_SCOPES:
        raise ValueError(f"merge_scope must be one of {MERGE_SCOPES}")
    is_major = scored["stream"] == major_id
    minor_value = threshold if minority_threshold is None else max(threshold, float(minority_threshold))
    keep = scored["probabilities"] >= np.where(is_major, threshold, minor_value)
    if merge_scope == "all":
        return _merge_dual_polarity_candidates(
            scored["peaks"][keep], scored["probabilities"][keep], merge_window_s, fs_hz,
            prominences=scored["prominences"][keep],
        )
    major_mask = keep & is_major
    minor_mask = keep & ~is_major
    extra_p, extra_q = _gap_fill_minority(
        scored["peaks"][major_mask], scored["peaks"][minor_mask], scored["probabilities"][minor_mask], fs_hz,
        merge_window_s=merge_window_s,
    )
    peaks = np.concatenate([scored["peaks"][major_mask], extra_p])
    prob = np.concatenate([scored["probabilities"][major_mask], extra_q])
    order = np.argsort(peaks, kind="stable")
    return peaks[order].astype(int), prob[order].astype(float)


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
    width_override_confidence: float = DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
    dual_polarity_merge_window_s: float = DEFAULT_DUAL_POLARITY_MERGE_WINDOW_S,
    merge_feature_scope: str = "per_stream",
    merge_scope: str = "gaps",
    minority_threshold: float | None = None,
    minority_min_peak_scale: float = DEFAULT_MINORITY_MIN_PEAK_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage 1, optional long-gap recovery, then the trained suppressor.

    polarity="merge" (opt-in) scores both polarities with the same RF, then
    keeps opposite-polarity peaks only inside long RR gaps (merge_scope="gaps").
    Defaults match the best INCART opt-in variant: per_stream features,
    gaps scope, minority_min_peak_scale=0.25. Default polarity remains
    whatever the caller passes (use "adaptive" in production).
    minority_threshold (merge only) raises the RF probability an
    opposite-polarity-to-the-record candidate must reach; the RF has no
    record-level polarity feature, so its scores on the minority stream are
    not calibrated by the Stage-2 threshold. minority_min_peak_scale is the
    Stage-1 baseline-blip guard for the minority stream (see
    _score_dual_polarity_streams).
    """
    signal, fs_hz = _validate_signal(signal, fs_hz)
    if not suppressor.fitted:
        raise ValueError("suppressor must be fitted before two-stage detection")
    if merge_scope not in MERGE_SCOPES:
        raise ValueError(f"merge_scope must be one of {MERGE_SCOPES}")
    if polarity == "merge":
        if recovery:
            raise ValueError("recovery is not supported with polarity='merge'")
        majority = select_signal_polarity(
            signal, fs_hz, scale_method=scale_method,
            width_override_confidence=width_override_confidence,
        ).polarity
        major_id = 0 if majority != "negative" else 1
        scored = _score_dual_polarity_streams(
            signal, fs_hz, suppressor, scale_method=scale_method, feature_scope=merge_feature_scope,
            minority_stream=1 - major_id, minority_min_peak_scale=minority_min_peak_scale,
        )
        threshold_value = float(suppressor.metadata.threshold if threshold is None else threshold)
        return _combine_scored_streams(
            scored, major_id, threshold_value, fs_hz, merge_scope=merge_scope,
            merge_window_s=dual_polarity_merge_window_s, minority_threshold=minority_threshold,
        )
    chosen_polarity = polarity
    if polarity == "adaptive":
        chosen_polarity = select_signal_polarity(
            signal, fs_hz, scale_method=scale_method,
            width_override_confidence=width_override_confidence,
        ).polarity

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

    features, _ = _candidate_features(signal, fs_hz, all_peaks, all_prom, scale_method=scale_method)
    retained, probabilities = suppressor.filter_candidates(all_peaks, features, threshold=threshold)
    return retained.astype(int), probabilities.astype(float)
