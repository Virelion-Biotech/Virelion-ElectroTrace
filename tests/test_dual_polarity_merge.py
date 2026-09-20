"""Tests for the opt-in dual-polarity merge mode (polarity="merge").

Background (see HANDOFF_dual_polarity_merge_I62.md): the first dual-polarity
experiment extracted Stage-2 features on the *pooled, index-sorted union* of
both polarity streams. `_candidate_features` derives rr_prev_s / rr_next_s /
rr_prev_ratio / rr_next_ratio and the RR median from the candidate list it is
given, so pooling silently changed those features into distances to
opposite-polarity neighbours -- a distribution the RF was never trained on.
`feature_scope="per_stream"` (the default) restores the training-time
distribution. These tests pin that behaviour and the plumbing of the merge
variants with a deterministic stub classifier (no dependence on RF training).
"""
import numpy as np
import pytest

from electrotrace.candidate_suppressor import _candidate_features
from electrotrace.validation_detectors import (
    _candidate_set,
    _gap_fill_minority,
    _merge_dual_polarity_candidates,
    _score_dual_polarity_streams,
    detect_r_peaks_two_stage,
    estimate_stage1_scale,
)

FS = 250.0
WIDTH_COL = 2  # "width_s" in _candidate_features' column order
PROM_COL = 1   # "prominence_z"


class _Meta:
    def __init__(self, threshold):
        self.threshold = threshold


class WidthStub:
    """Deterministic stand-in for the fitted RF: narrow QRS-like candidates get
    p=0.9, wide ones (T-wave humps) or low-prominence blips p=0.1."""
    fitted = True

    def __init__(self, max_width_s=0.09, threshold=0.5, min_prominence_z=2.0):
        self.max_width_s = max_width_s
        self.min_prominence_z = min_prominence_z
        self.metadata = _Meta(threshold)

    def predict_proba(self, features):
        f = np.asarray(features)
        ok = (f[:, WIDTH_COL] <= self.max_width_s) & (f[:, PROM_COL] >= self.min_prominence_z)
        return np.where(ok, 0.9, 0.1)

    def filter_candidates(self, candidate_indices, features, threshold=None):
        prob = self.predict_proba(features)
        thr = self.metadata.threshold if threshold is None else threshold
        return np.asarray(candidate_indices, dtype=int)[prob >= thr], prob


def _g(t, c, s, a):
    return a * np.exp(-0.5 * ((t - c) / s) ** 2)


def _t_discordant_record(n_beats=60, pvc_every=4, seed=0):
    """Normal narrow positive beats; every `pvc_every`-th beat is a
    T-wave-discordant PVC: small narrow NEGATIVE notch followed 200 ms later
    by a large wide POSITIVE T-wave (the I62 morphology)."""
    rng = np.random.default_rng(seed)
    rr = 0.8
    times = 1.0 + np.arange(n_beats) * rr
    t = np.arange(0, times[-1] + 1.5, 1 / FS)
    x = rng.normal(0, 0.01, t.size)
    refs, is_pvc = [], []
    for i, tb in enumerate(times):
        pvc = i % pvc_every == pvc_every - 1
        if pvc:
            x += _g(t, tb, 0.012, -0.55) + _g(t, tb + 0.20, 0.055, 1.1)
        else:
            x += _g(t, tb, 0.008, 1.0) + _g(t, tb + 0.25, 0.045, 0.25)
        refs.append(int(round(tb * FS)))
        is_pvc.append(pvc)
    return x, np.asarray(refs), np.asarray(is_pvc)


def test_nms_keeps_higher_probability_within_window():
    peaks, probs = _merge_dual_polarity_candidates(
        np.array([100, 110, 400]), np.array([0.6, 0.9, 0.7]), 0.10, FS
    )
    assert peaks.tolist() == [110, 400]
    assert probs.tolist() == [0.9, 0.7]


def test_nms_ties_broken_by_prominence_not_sample_order():
    peaks, _ = _merge_dual_polarity_candidates(
        np.array([100, 120]), np.array([1.0, 1.0]), 0.10, FS, prominences=np.array([0.5, 2.0])
    )
    assert peaks.tolist() == [120]


def test_nms_keeps_far_peaks_and_handles_empty():
    peaks, _ = _merge_dual_polarity_candidates(np.array([100, 300]), np.array([0.6, 0.7]), 0.10, FS)
    assert peaks.tolist() == [100, 300]
    peaks, probs = _merge_dual_polarity_candidates(np.array([], dtype=int), np.array([]), 0.10, FS)
    assert peaks.size == 0 and probs.size == 0


def test_nms_does_not_remove_t_wave_distance_candidates():
    """A 200 ms neighbour is outside the 100 ms window: it must be rejected by
    the classifier, never silently by the dedup window."""
    peaks, _ = _merge_dual_polarity_candidates(np.array([100, 150]), np.array([0.9, 0.8]), 0.10, FS)
    assert peaks.tolist() == [100, 150]


def test_per_stream_features_match_training_time_features():
    """Regression guard for the pooled-RR-feature defect: each stream's
    features must equal what _candidate_features gives that stream ALONE."""
    x, _, _ = _t_discordant_record()
    stub = WidthStub()
    scored = _score_dual_polarity_streams(x, FS, stub, scale_method="windowed_std", feature_scope="per_stream")
    z = x - np.median(x)
    scale = estimate_stage1_scale(z, FS, method="windowed_std")
    for sid, sig in enumerate((z, -z)):
        peaks, prom = _candidate_set(sig, FS, scale)
        alone, _ = _candidate_features(x, FS, peaks, prom, scale_method="windowed_std")
        mask = scored["stream"] == sid
        assert scored["peaks"][mask].tolist() == peaks.tolist()
        assert np.array_equal(scored["probabilities"][mask], stub.predict_proba(alone))


def test_pooled_features_differ_in_rr_columns():
    """Documents WHY pooled was wrong: interleaving both streams changes the
    RR-derived columns (13..16) of the very same candidates."""
    x, _, _ = _t_discordant_record()
    z = x - np.median(x)
    scale = estimate_stage1_scale(z, FS, method="windowed_std")
    pos, pos_prom = _candidate_set(z, FS, scale)
    neg, neg_prom = _candidate_set(-z, FS, scale)
    alone, _ = _candidate_features(x, FS, pos, pos_prom, scale_method="windowed_std")
    peaks = np.concatenate([pos, neg])
    prom = np.concatenate([pos_prom, neg_prom])
    order = np.argsort(peaks, kind="stable")
    pooled, _ = _candidate_features(x, FS, peaks[order], prom[order], scale_method="windowed_std")
    is_pos = np.isin(peaks[order], pos)
    pooled_pos = pooled[is_pos]
    assert not np.allclose(alone[:, 13:17], pooled_pos[:, 13:17])
    # Non-RR columns (amplitude, prominence, width, shape...) are unaffected.
    assert np.allclose(alone[:, :13], pooled_pos[:, :13])


def test_merge_recovers_discordant_pvc_notches_and_rejects_t_waves():
    x, refs, is_pvc = _t_discordant_record()
    stub = WidthStub()
    adaptive, _ = detect_r_peaks_two_stage(x, FS, stub, polarity="positive", scale_method="windowed_std")
    # Force full dual-stream (not ship gaps default) to assert recovery property.
    merged, probs = detect_r_peaks_two_stage(
        x, FS, stub, polarity="merge", scale_method="windowed_std",
        merge_scope="all", minority_min_peak_scale=0.0,
    )
    tol = int(0.075 * FS)

    def matched(det, ref):
        return sum(np.any(np.abs(det - r) <= tol) for r in ref)

    n_pvc = int(is_pvc.sum())
    n_norm = int((~is_pvc).sum())
    assert matched(adaptive, refs[is_pvc]) == 0
    # Allow one miss on synthetic edge beats; dual-stream must recover almost all.
    assert matched(merged, refs[is_pvc]) >= n_pvc - 1
    assert matched(merged, refs[~is_pvc]) >= n_norm - 1
    far = [p for p in merged if np.min(np.abs(refs - p)) > tol]
    assert far == []
    assert len(probs) == len(merged)



def test_gap_mode_is_superset_of_majority_and_never_adds_outside_gaps():
    x, refs, _ = _t_discordant_record()
    stub = WidthStub()
    majority, _ = detect_r_peaks_two_stage(x, FS, stub, polarity="positive", scale_method="windowed_std")
    gapped, _ = detect_r_peaks_two_stage(
        x, FS, stub, polarity="merge", merge_scope="gaps", scale_method="windowed_std"
    )
    assert set(majority.tolist()) <= set(gapped.tolist())


def test_gap_fill_only_inside_long_gaps():
    majority = np.array([0, 200, 400, 600, 1000, 1200, 1400])  # one 400-sample gap (2x typical)
    minority = np.array([100, 300, 800, 1100])                  # only 800 lies inside the long gap
    prob = np.array([0.99, 0.99, 0.9, 0.99])
    p, q = _gap_fill_minority(majority, minority, prob, FS)
    assert p.tolist() == [800] and q.tolist() == [0.9]


def test_gap_fill_respects_guard_and_budget():
    majority = np.array([0, 200, 400, 600, 1000, 1200, 1400])
    # 610 is inside guard (25 samples) of the gap start; only 1 beat missing => budget 1
    minority = np.array([610, 790, 810])
    prob = np.array([0.99, 0.8, 0.95])
    p, _ = _gap_fill_minority(majority, minority, prob, FS)
    assert p.tolist() == [810]


def test_minority_threshold_filters_opposite_polarity_only():
    x, refs, is_pvc = _t_discordant_record()
    stub = WidthStub()
    loose, _ = detect_r_peaks_two_stage(x, FS, stub, polarity="merge", scale_method="windowed_std")
    strict, _ = detect_r_peaks_two_stage(
        x, FS, stub, polarity="merge", minority_threshold=0.95, scale_method="windowed_std"
    )
    tol = int(0.075 * FS)
    kept_pvc = sum(np.any(np.abs(strict - r) <= tol) for r in refs[is_pvc])
    kept_normal = sum(np.any(np.abs(strict - r) <= tol) for r in refs[~is_pvc])
    assert kept_pvc == 0                       # notches are minority-stream, p=0.9 < 0.95
    assert kept_normal == int((~is_pvc).sum()) # majority stream untouched
    assert len(strict) < len(loose)


def test_default_modes_unchanged_by_new_kwargs():
    x, _, _ = _t_discordant_record()
    stub = WidthStub()
    a, _ = detect_r_peaks_two_stage(x, FS, stub, polarity="positive", scale_method="windowed_std")
    b, _ = detect_r_peaks_two_stage(
        x, FS, stub, polarity="positive", scale_method="windowed_std",
        dual_polarity_merge_window_s=0.2, merge_feature_scope="pooled", merge_scope="gaps", minority_threshold=0.99,
    )
    assert np.array_equal(a, b)


def test_merge_rejects_bad_arguments():
    x, _, _ = _t_discordant_record()
    stub = WidthStub()
    with pytest.raises(ValueError):
        detect_r_peaks_two_stage(x, FS, stub, polarity="merge", recovery=True)
    with pytest.raises(ValueError):
        detect_r_peaks_two_stage(x, FS, stub, polarity="merge", merge_scope="nope")
    with pytest.raises(ValueError):
        detect_r_peaks_two_stage(x, FS, stub, polarity="merge", merge_feature_scope="nope")


def test_mirrored_view_gives_baseline_noise_large_prominence():
    """Root cause of the minority-stream candidate flood: in the -z view of a
    positive-going record, isoelectric baseline maxima are bounded by QRS
    'valleys', so scipy reports a prominence ~ R-wave depth for them."""
    x, refs, _ = _t_discordant_record()
    z = x - np.median(x)
    scale = estimate_stage1_scale(z, FS, method="windowed_std")
    neg, neg_prom = _candidate_set(-z, FS, scale)
    tol = int(0.075 * FS)
    baseline = np.array([np.min(np.abs(refs - p)) > tol and z[p] > -0.2 for p in neg])
    assert baseline.any()
    assert neg_prom[baseline].max() > 1.5 * scale       # far above the 0.5*scale acceptance bar
    assert np.all(-z[neg[baseline]] < 0.25)             # yet they sit at baseline level


def test_minority_peak_floor_drops_baseline_blips():
    x, refs, is_pvc = _t_discordant_record()
    stub = WidthStub(min_prominence_z=0.0)   # a classifier that does NOT reject blips itself
    kw = dict(scale_method="windowed_std", feature_scope="per_stream", minority_stream=1)
    no_floor = _score_dual_polarity_streams(x, FS, stub, minority_min_peak_scale=0.0, **kw)
    floored = _score_dual_polarity_streams(x, FS, stub, minority_min_peak_scale=1.0, **kw)
    n_neg = lambda d: int((d["stream"] == 1).sum())
    assert n_neg(floored) < n_neg(no_floor)
    tol = int(0.075 * FS)
    # every PVC notch survives the floor, and the positive stream is untouched
    for r in refs[is_pvc]:
        assert np.any((floored["stream"] == 1) & (np.abs(floored["peaks"] - r) <= tol))
    assert int((floored["stream"] == 0).sum()) == int((no_floor["stream"] == 0).sum())
