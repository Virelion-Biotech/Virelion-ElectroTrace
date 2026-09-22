import numpy as np
import pytest

from electrotrace.fp_analysis import (
    CATEGORIES,
    analyze_record,
    amplitude_context,
    candidate_scoring_summary,
    classify_false_positives,
    match_peaks_detailed,
    pool_false_positives,
)
from electrotrace.validation import match_peaks

FS = 1000.0
REF = np.array([1000, 2000, 3000])


def test_match_peaks_detailed_agrees_with_match_peaks_on_random_inputs():
    rng = np.random.default_rng(7)
    for _ in range(200):
        ref = np.unique(rng.integers(0, 20000, size=rng.integers(1, 60)))
        det = np.unique(
            np.concatenate(
                [
                    ref[rng.random(ref.size) < 0.8] + rng.integers(-120, 120, size=1),
                    rng.integers(0, 20000, size=rng.integers(0, 40)),
                ]
            )
        )
        det = det[det >= 0]
        m = match_peaks(det, ref, 250.0, 75.0)
        d = match_peaks_detailed(det, ref, 250.0, 75.0)
        assert d.true_positive == m.true_positive
        assert int((~d.detected_matched).sum()) == m.false_positive
        assert int((~d.reference_matched).sum()) == m.false_negative


def test_match_peaks_detailed_validates_like_match_peaks():
    with pytest.raises(ValueError, match="strictly increasing"):
        match_peaks_detailed([5, 5], [1], 250.0)
    with pytest.raises(ValueError, match="non-negative"):
        match_peaks_detailed([-1], [1], 250.0)


def test_false_positive_categories_follow_timing_geometry():
    fps = [1040, 1100, 1300, 2900, 2750, 2500, 500, 3500]
    out = classify_false_positives(fps, REF, FS, tolerance_ms=75.0)
    got = dict(zip(fps, out["category"]))
    assert got[1040] == "duplicate_within_tolerance"
    assert got[1100] == "post_qrs_near"
    assert got[1300] == "post_qrs_t_window"
    assert got[2900] == "pre_qrs_near"
    assert got[2750] == "pre_qrs_p_window"
    assert got[2500] == "mid_interval_other"
    assert got[500] == "outside_reference_span"
    assert got[3500] == "outside_reference_span"
    assert set(out["category"]) <= set(CATEGORIES)


def test_false_positive_offsets_phase_and_nearest_index():
    out = classify_false_positives([1300, 2900], REF, FS)
    assert out["nearest_offset_ms"][0] == pytest.approx(300.0)  # after ref 1000
    assert out["nearest_offset_ms"][1] == pytest.approx(-100.0)  # before ref 3000
    assert out["nearest_ref_index"].tolist() == [0, 2]
    assert out["rr_ms"][0] == pytest.approx(1000.0)
    assert out["phase"][0] == pytest.approx(0.3)
    assert out["dt_prev_ms"][1] == pytest.approx(900.0)
    assert out["dt_next_ms"][1] == pytest.approx(100.0)


def test_classify_handles_empty_inputs():
    assert classify_false_positives([], REF, FS)["category"].size == 0
    out = classify_false_positives([10, 20], [], FS)
    assert list(out["category"]) == ["outside_reference_span"] * 2


def test_band_edges_are_configurable():
    out = classify_false_positives(
        [1300], REF, FS, bands_ms={"post_qrs_near_max": 400.0}
    )
    assert out["category"][0] == "post_qrs_near"


def _spiky(n=4000, fs=FS):
    x = np.zeros(n)
    for c, a in [(1000, 3.0), (2000, 3.0), (3000, 3.0), (1300, 0.9), (2500, -0.9), (1100, 0.2)]:
        x[c - 2 : c + 3] += a * np.hanning(5)
    return x


def test_amplitude_context_ratio_and_polarity():
    x = _spiky()
    fps = np.array([1300, 2500, 1100])
    out = classify_false_positives(fps, REF, FS)
    ctx = amplitude_context(x, FS, fps, REF, out["nearest_ref_index"])
    assert ctx["amp_ratio"][0] == pytest.approx(0.3, abs=0.05)
    assert ctx["same_polarity"][0]
    assert not ctx["same_polarity"][1]  # negative spike beside positive beats
    assert ctx["amp_ratio"][2] < 0.1  # baseline-blip sized


def test_candidate_scoring_summary_perfect_and_single_class():
    cand = np.array([1000, 1300, 2000, 2500, 3000])
    good = candidate_scoring_summary(cand, [0.9, 0.1, 0.95, 0.2, 0.85], REF, FS, 0.5)
    assert good["auc"] == pytest.approx(1.0)
    assert good["oracle_best_f1"] == pytest.approx(1.0)
    assert good["false_candidates_retained_at_threshold"] == 0
    only_true = candidate_scoring_summary(REF, [0.9, 0.9, 0.9], REF, FS, 0.5)
    assert only_true["auc"] is None
    with pytest.raises(ValueError, match="same length"):
        candidate_scoring_summary([1, 2], [0.5], REF, FS, 0.5)


def test_candidate_summary_flags_threshold_problem_when_ranking_is_fine():
    cand = np.array([1000, 1300, 2000, 2500, 3000])
    prob = [0.45, 0.10, 0.48, 0.20, 0.44]  # ranks perfectly, but all under 0.5
    s = candidate_scoring_summary(cand, prob, REF, FS, 0.5)
    assert s["auc"] == pytest.approx(1.0)
    assert s["true_candidates_dropped_at_threshold"] == 3
    assert s["oracle_best_threshold"] < 0.45


def test_analyze_record_end_to_end_and_pooling():
    x = _spiky()
    cand = np.array([1000, 1040, 1300, 2000, 2500, 3000])
    prob = np.array([0.9, 0.8, 0.7, 0.9, 0.6, 0.9])
    res = analyze_record(x, FS, cand, prob, REF, threshold=0.5)
    assert res["metrics"]["false_positive"] == 3
    assert res["detected_over_reference"] == pytest.approx(2.0)
    cats = res["false_positive_categories"]
    assert cats["duplicate_within_tolerance"] == 1
    assert cats["post_qrs_t_window"] == 1
    assert cats["mid_interval_other"] == 1
    table = res["false_positive_table"]
    assert table["probability"].tolist() == pytest.approx([0.8, 0.7, 0.6])

    pooled = pool_false_positives([table, table])
    assert pooled["n_false_positives"] == 6
    assert pooled["category_counts"]["duplicate_within_tolerance"] == 2
    assert sum(pooled["category_fractions"].values()) == pytest.approx(1.0)
    assert pooled["signed_offset_to_nearest_reference_ms"]["n"] == 6
    assert pool_false_positives([])["n_false_positives"] == 0


def test_analyze_record_without_signal_leaves_amplitude_unset():
    cand = np.array([1000, 1300, 2000, 3000])
    res = analyze_record(None, FS, cand, np.full(4, 0.9), REF, threshold=0.5)
    assert np.isnan(res["false_positive_table"]["amp_ratio"]).all()
