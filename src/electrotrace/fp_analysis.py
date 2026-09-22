"""False-positive forensics for R-peak detectors.

This is a *development diagnostic*, not a validation metric.  It answers the
question "what are the extra detections?" for a detector that has good
sensitivity but poor PPV: are they double detections of the same beat, a
deflection just after the QRS, something in the T-wave window, or unrelated to
any reference beat?  The timing bands are heuristics for triage; they do not
prove a physiological cause.  Use them to decide which experiment to run next,
and never to tune a model on data you intend to use as a test set.

Everything here is pure numpy (plus ``sklearn.metrics.roc_auc_score`` for the
candidate-ranking summary), so it is testable without WFDB or a trained model.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .validation import _validate_integer_samples, match_peaks

CATEGORIES = (
    "duplicate_within_tolerance",
    "post_qrs_near",
    "post_qrs_t_window",
    "pre_qrs_near",
    "pre_qrs_p_window",
    "mid_interval_other",
    "outside_reference_span",
)

DEFAULT_BANDS_MS = {
    "post_qrs_near_max": 150.0,  # tolerance < dt_after_prev_ref <= this
    "post_qrs_t_window_max": 450.0,  # (post_qrs_near_max, this]
    "pre_qrs_near_max": 150.0,  # tolerance < dt_before_next_ref <= this
    "pre_qrs_p_window_max": 300.0,  # (pre_qrs_near_max, this]
}

OFFSET_EDGES_MS = tuple(range(-600, 601, 25))
PHASE_EDGES = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


@dataclass(frozen=True)
class MatchDetail:
    """One-to-one matching identical to ``validation.match_peaks``, with labels."""

    detected_matched: np.ndarray  # bool, one per detection
    reference_matched: np.ndarray  # bool, one per reference beat
    true_positive: int


def match_peaks_detailed(
    detected: Sequence[int],
    reference: Sequence[int],
    fs_hz: float,
    tolerance_ms: float = 75.0,
) -> MatchDetail:
    """Same greedy two-pointer matching as ``match_peaks``, but keeps the labels."""
    fs_hz = float(fs_hz)
    tolerance_ms = float(tolerance_ms)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be positive and finite")
    if not np.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("tolerance_ms must be positive and finite")
    det = _validate_integer_samples(detected, "detected_samples")
    ref = _validate_integer_samples(reference, "reference_samples")
    tol = tolerance_ms * fs_hz / 1000.0
    det_ok = np.zeros(det.size, dtype=bool)
    ref_ok = np.zeros(ref.size, dtype=bool)
    i = j = tp = 0
    while i < det.size and j < ref.size:
        delta = int(det[i]) - int(ref[j])
        if abs(delta) <= tol:
            det_ok[i] = True
            ref_ok[j] = True
            tp += 1
            i += 1
            j += 1
        elif det[i] < ref[j]:
            i += 1
        else:
            j += 1
    return MatchDetail(det_ok, ref_ok, tp)


def classify_false_positives(
    fp_samples: Sequence[int],
    reference: Sequence[int],
    fs_hz: float,
    *,
    tolerance_ms: float = 75.0,
    bands_ms: dict | None = None,
) -> dict:
    """Locate each false positive relative to its neighbouring reference beats.

    Returns arrays (one entry per false positive): ``dt_prev_ms`` (time after the
    preceding reference beat), ``dt_next_ms`` (time before the next one),
    ``rr_ms``, ``phase`` (dt_prev / RR, 0..1), ``nearest_offset_ms`` (signed:
    detection minus nearest reference), ``nearest_ref_index`` and ``category``.
    """
    bands = {**DEFAULT_BANDS_MS, **(bands_ms or {})}
    fp = np.asarray(fp_samples, dtype=np.int64)
    ref = np.asarray(reference, dtype=np.int64)
    n = fp.size
    ms = 1000.0 / float(fs_hz)
    out = {
        "sample": fp,
        "dt_prev_ms": np.full(n, np.nan),
        "dt_next_ms": np.full(n, np.nan),
        "rr_ms": np.full(n, np.nan),
        "phase": np.full(n, np.nan),
        "nearest_offset_ms": np.full(n, np.nan),
        "nearest_ref_index": np.full(n, -1, dtype=np.int64),
        "category": np.full(n, "outside_reference_span", dtype=object),
    }
    if n == 0 or ref.size == 0:
        return out

    idx = np.searchsorted(ref, fp, side="right")
    prev_i = idx - 1
    next_i = idx
    has_prev = prev_i >= 0
    has_next = next_i < ref.size
    dt_prev = np.where(has_prev, (fp - ref[np.clip(prev_i, 0, ref.size - 1)]) * ms, np.nan)
    dt_next = np.where(has_next, (ref[np.clip(next_i, 0, ref.size - 1)] - fp) * ms, np.nan)
    both = has_prev & has_next
    rr = np.where(both, dt_prev + dt_next, np.nan)

    prev_nearer = np.where(
        both, dt_prev <= dt_next, has_prev
    )  # only-prev -> prev; only-next -> next
    nearest_dt = np.where(prev_nearer, dt_prev, dt_next)
    nearest_signed = np.where(prev_nearer, dt_prev, -dt_next)
    nearest_idx = np.where(prev_nearer, prev_i, next_i)

    tol = float(tolerance_ms)
    category = np.full(n, "mid_interval_other", dtype=object)
    # apply lowest precedence first so higher-precedence rules overwrite
    category[(dt_next > tol) & (dt_next <= bands["pre_qrs_p_window_max"])] = "pre_qrs_p_window"
    category[(dt_next > tol) & (dt_next <= bands["pre_qrs_near_max"])] = "pre_qrs_near"
    category[(dt_prev > tol) & (dt_prev <= bands["post_qrs_t_window_max"])] = "post_qrs_t_window"
    category[(dt_prev > tol) & (dt_prev <= bands["post_qrs_near_max"])] = "post_qrs_near"
    category[~both] = "outside_reference_span"
    category[nearest_dt <= tol] = "duplicate_within_tolerance"

    out.update(
        dt_prev_ms=dt_prev,
        dt_next_ms=dt_next,
        rr_ms=rr,
        phase=np.where(both & (rr > 0), dt_prev / np.where(rr > 0, rr, 1.0), np.nan),
        nearest_offset_ms=nearest_signed,
        nearest_ref_index=nearest_idx.astype(np.int64),
        category=category,
    )
    return out


def _peak_context(signal: np.ndarray, fs_hz: float, i: int, *, half_s: float, peak_s: float):
    n = signal.size
    lo = max(0, i - int(round(half_s * fs_hz)))
    hi = min(n, i + int(round(half_s * fs_hz)) + 1)
    med = float(np.median(signal[lo:hi]))
    w = int(round(peak_s * fs_hz))
    a = max(0, i - w)
    b = min(n, i + w + 1)
    seg = signal[a:b] - med
    k = int(np.argmax(np.abs(seg)))
    return abs(float(seg[k])), (1.0 if seg[k] >= 0 else -1.0)


def amplitude_context(
    signal: np.ndarray,
    fs_hz: float,
    fp_samples: Sequence[int],
    reference: Sequence[int],
    nearest_ref_index: Sequence[int],
    *,
    half_window_s: float = 1.0,
    peak_window_s: float = 0.03,
) -> dict:
    """Peak height of each false positive relative to its nearest true beat.

    ``amp_ratio`` near 1 with equal polarity suggests a second detection on a
    QRS-sized deflection; 0.2-0.6 is typical of T-wave-sized deflections; below
    about 0.1 suggests baseline wander/noise.  ``same_polarity`` is False when
    the false positive points the opposite way to the nearest true beat.
    """
    x = np.asarray(signal, dtype=float)
    fp = np.asarray(fp_samples, dtype=np.int64)
    ref = np.asarray(reference, dtype=np.int64)
    ni = np.asarray(nearest_ref_index, dtype=np.int64)
    ratio = np.full(fp.size, np.nan)
    same = np.zeros(fp.size, dtype=bool)
    for k in range(fp.size):
        if ni[k] < 0 or not (0 <= fp[k] < x.size):
            continue
        r = int(ref[ni[k]])
        if not (0 <= r < x.size):
            continue
        a_fp, s_fp = _peak_context(x, fs_hz, int(fp[k]), half_s=half_window_s, peak_s=peak_window_s)
        a_ref, s_ref = _peak_context(x, fs_hz, r, half_s=half_window_s, peak_s=peak_window_s)
        if a_ref > 0:
            ratio[k] = a_fp / a_ref
        same[k] = s_fp == s_ref
    return {"amp_ratio": ratio, "same_polarity": same}


def _quantiles(values: np.ndarray) -> dict | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), **{f"p{p:02d}": float(x) for p, x in zip((5, 25, 50, 75, 95), q)}}


def candidate_scoring_summary(
    candidates: Sequence[int],
    probabilities: Sequence[float],
    reference: Sequence[int],
    fs_hz: float,
    threshold: float,
    *,
    tolerance_ms: float = 75.0,
    oracle_grid: Sequence[float] | None = None,
) -> dict:
    """How well do the Stage-2 probabilities separate true from false candidates?

    A candidate is "true" if the one-to-one matcher pairs it with a reference beat
    when *all* candidates are submitted.  High AUC with a poor operating point
    means the ranking is fine and only the threshold is off; AUC near 0.5 means
    the features do not discriminate on this data.  ``oracle_*`` uses reference
    labels to pick the best global threshold for this record: an upper bound for
    diagnosis, never a deployable setting.
    """
    from sklearn.metrics import roc_auc_score

    cand = np.asarray(candidates, dtype=np.int64)
    prob = np.asarray(probabilities, dtype=float)
    if cand.shape != prob.shape:
        raise ValueError("candidates and probabilities must have the same length")
    detail = match_peaks_detailed(cand, reference, fs_hz, tolerance_ms)
    truth = detail.detected_matched
    n_true = int(truth.sum())
    n_false = int((~truth).sum())
    auc = float(roc_auc_score(truth, prob)) if n_true and n_false else None
    kept = prob >= float(threshold)

    grid = np.linspace(0.02, 0.98, 49) if oracle_grid is None else np.asarray(oracle_grid, dtype=float)
    best_f1, best_thr = -1.0, None
    if len(reference):
        for t in grid:
            sel = cand[prob >= t]
            m = match_peaks(sel, reference, fs_hz, tolerance_ms) if sel.size else None
            f1 = m.f1 if m else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
    return {
        "n_candidates": int(cand.size),
        "n_true_candidates": n_true,
        "n_false_candidates": n_false,
        "auc": auc,
        "prob_true": _quantiles(prob[truth]),
        "prob_false": _quantiles(prob[~truth]),
        "false_candidates_retained_at_threshold": int((kept & ~truth).sum()),
        "true_candidates_dropped_at_threshold": int((~kept & truth).sum()),
        "oracle_best_f1": float(best_f1) if best_thr is not None else None,
        "oracle_best_threshold": best_thr,
        "oracle_is_diagnostic_only": True,
    }


def analyze_record(
    signal: np.ndarray | None,
    fs_hz: float,
    candidates: Sequence[int],
    probabilities: Sequence[float],
    reference: Sequence[int],
    *,
    threshold: float,
    tolerance_ms: float = 75.0,
    bands_ms: dict | None = None,
) -> dict:
    """Full forensic summary for one record at the model's operating threshold."""
    cand = np.asarray(candidates, dtype=np.int64)
    prob = np.asarray(probabilities, dtype=float)
    ref = np.asarray(reference, dtype=np.int64)
    retained = cand[prob >= float(threshold)]
    metrics = match_peaks(retained, ref, fs_hz, tolerance_ms)
    detail = match_peaks_detailed(retained, ref, fs_hz, tolerance_ms)
    fp_samples = retained[~detail.detected_matched]
    fp_prob = prob[prob >= float(threshold)][~detail.detected_matched]
    cls = classify_false_positives(
        fp_samples, ref, fs_hz, tolerance_ms=tolerance_ms, bands_ms=bands_ms
    )
    if signal is not None and fp_samples.size:
        cls.update(
            amplitude_context(signal, fs_hz, fp_samples, ref, cls["nearest_ref_index"])
        )
    else:
        cls["amp_ratio"] = np.full(fp_samples.size, np.nan)
        cls["same_polarity"] = np.zeros(fp_samples.size, dtype=bool)
    cls["probability"] = fp_prob
    counts = {c: int(np.sum(cls["category"] == c)) for c in CATEGORIES}
    return {
        "metrics": metrics.to_dict(),
        "threshold": float(threshold),
        "detected_over_reference": (
            float(metrics.detected_count / metrics.reference_count) if metrics.reference_count else None
        ),
        "false_positive_categories": counts,
        "candidates": candidate_scoring_summary(
            cand, prob, ref, fs_hz, threshold, tolerance_ms=tolerance_ms
        ),
        "false_positive_table": cls,
    }


def histogram(values: Sequence[float], edges: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(v, bins=np.asarray(edges, dtype=float))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts], "n": int(v.size)}


def pool_false_positives(tables: Sequence[dict]) -> dict:
    """Aggregate several records' ``false_positive_table`` dicts into one summary."""
    if not tables:
        return {"n_false_positives": 0}
    cat = np.concatenate([t["category"] for t in tables]) if tables else np.array([], dtype=object)
    n = int(cat.size)
    if n == 0:
        return {"n_false_positives": 0}
    cat_counts = {c: int(np.sum(cat == c)) for c in CATEGORIES}
    off = np.concatenate([t["nearest_offset_ms"] for t in tables])
    phase = np.concatenate([t["phase"] for t in tables])
    amp = np.concatenate([t["amp_ratio"] for t in tables])
    same = np.concatenate([t["same_polarity"] for t in tables])
    prob = np.concatenate([t["probability"] for t in tables])
    finite_amp = np.isfinite(amp)
    return {
        "n_false_positives": n,
        "category_counts": cat_counts,
        "category_fractions": {c: cat_counts[c] / n for c in CATEGORIES},
        "signed_offset_to_nearest_reference_ms": histogram(off, OFFSET_EDGES_MS),
        "phase_in_rr_interval": histogram(phase, PHASE_EDGES),
        "amp_ratio_to_nearest_true_beat": _quantiles(amp),
        "fraction_amp_ratio_below_0.1": float(np.mean(amp[finite_amp] < 0.1)) if finite_amp.any() else None,
        "fraction_amp_ratio_0.1_to_0.6": (
            float(np.mean((amp[finite_amp] >= 0.1) & (amp[finite_amp] < 0.6))) if finite_amp.any() else None
        ),
        "fraction_amp_ratio_at_least_0.6": float(np.mean(amp[finite_amp] >= 0.6)) if finite_amp.any() else None,
        "fraction_opposite_polarity_to_nearest_true_beat": (
            float(np.mean(~same[finite_amp])) if finite_amp.any() else None
        ),
        "stage2_probability": _quantiles(prob),
    }
