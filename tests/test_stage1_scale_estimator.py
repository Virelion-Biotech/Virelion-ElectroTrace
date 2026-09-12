"""Priority-1 fix: Stage-1 candidate generation used to size its prominence
threshold from a single global std(signal). A short high-amplitude
artifact/noise burst inflates that one number and starves candidate
generation over the *entire* record -- this is what happened on INCART
(see docs/PRIORITY1_STAGE1_HANDOFF.md): stage1_candidates / reference_count
as low as ~0.01 on records where gqrs still scored F1 > 0.98.

These tests reproduce that failure mode on a synthetic record (so they run
without any PhysioNet data). Four scale estimators were compared on real
MIT-BIH + INCART data (2026-09-12) before picking a default -- see the
module-level comment above DEFAULT_SCALE_METHOD in validation_detectors.py
for the full numbers. "windowed_std" was adopted; "mad", "windowed_mad",
and "adaptive" were tried and rejected, but are kept as options (with tests)
since they're informative counterexamples and may be useful for future
investigation.
"""
import numpy as np
import pytest

from electrotrace.validation_detectors import (
    DEFAULT_SCALE_METHOD,
    detect_r_peaks,
    estimate_stage1_scale,
)


def _incart_like_signal_with_artifact_burst(seed=0, burst_scale=15.0, burst_duration_s=2.0):
    """~60s @ 257 Hz (INCART's rate), regular QRS-like pulses at 75 bpm, plus
    a short (2s / ~3% of the record) high-amplitude artifact burst."""
    fs = 257.0
    n = int(60.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    signal = 0.01 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.01, size=n)

    rr_samples = int(round(0.8 * fs))  # ~75 bpm
    half_width = max(3, int(round(0.02 * fs)))
    beats = np.arange(rr_samples, n - rr_samples, rr_samples)
    for peak in beats:
        lo, hi = peak - half_width, peak + half_width + 1
        signal[lo:hi] += 1.0 * np.hanning(hi - lo)

    burst_start = int(20 * fs)
    burst_len = int(burst_duration_s * fs)
    signal[burst_start:burst_start + burst_len] += rng.normal(0, burst_scale, size=burst_len)

    return signal, fs, beats, burst_start, burst_len


def _coverage(peaks, references, tolerance):
    peaks = np.asarray(peaks)
    if len(references) == 0:
        return 1.0
    hits = sum(1 for r in references if peaks.size and np.any(np.abs(peaks - r) <= tolerance))
    return hits / len(references)


@pytest.mark.parametrize("seed", range(5))
def test_global_std_starves_candidates_after_artifact_burst(seed):
    """Reproduces the bug: one short artifact burst collapses Stage-1 recall
    across the whole record when scale is a single global std()."""
    signal, fs, beats, burst_start, burst_len = _incart_like_signal_with_artifact_burst(seed=seed)
    tolerance = int(round(0.075 * fs))
    clean_reference = beats[(beats < burst_start - fs) | (beats > burst_start + burst_len + fs)]

    legacy_peaks = detect_r_peaks(signal, fs, scale_method="std")
    assert _coverage(legacy_peaks, clean_reference, tolerance) < 0.2


@pytest.mark.parametrize("seed", range(5))
def test_windowed_mad_recovers_candidate_coverage(seed):
    """windowed_mad also fixes this synthetic burst (its mechanism is sound
    in isolation) -- it was rejected as the shipped default only because of
    a *different* failure mode found on real INCART data (MAD's median
    flips to tracking QRS/T amplitude once a window's elevated fraction
    exceeds ~50%), not because this burst-recovery behavior is wrong."""
    signal, fs, beats, burst_start, burst_len = _incart_like_signal_with_artifact_burst(seed=seed)
    tolerance = int(round(0.075 * fs))
    clean_reference = beats[(beats < burst_start - fs) | (beats > burst_start + burst_len + fs)]

    fixed_peaks = detect_r_peaks(signal, fs, scale_method="windowed_mad")
    assert _coverage(fixed_peaks, clean_reference, tolerance) > 0.95


@pytest.mark.parametrize("seed", range(5))
def test_windowed_std_recovers_candidate_coverage(seed):
    """The shipped fix: windowed_std is not dominated by the short burst
    either, so clean-region QRS candidates are generated again -- and (per
    real-data validation) without windowed_mad's collateral damage on
    higher-occupancy records."""
    signal, fs, beats, burst_start, burst_len = _incart_like_signal_with_artifact_burst(seed=seed)
    tolerance = int(round(0.075 * fs))
    clean_reference = beats[(beats < burst_start - fs) | (beats > burst_start + burst_len + fs)]

    fixed_peaks = detect_r_peaks(signal, fs, scale_method="windowed_std")
    assert _coverage(fixed_peaks, clean_reference, tolerance) > 0.95


def test_default_scale_method_is_windowed_std():
    # windowed_std was adopted 2026-09-12 after real-data validation on
    # full INCART + MIT-BIH held-out: zero MIT-BIH cost (F1 0.9651 both
    # ways) and a substantial improvement on the Priority-1 focus records.
    # mad / windowed_mad / adaptive were all tried and rejected (see the
    # module-level comment above DEFAULT_SCALE_METHOD for the numbers).
    # This test locks in the decision so a future refactor can't silently
    # revert to a disproven default.
    assert DEFAULT_SCALE_METHOD == "windowed_std"


def test_clean_signal_all_methods_agree_reasonably():
    """Sanity check: on a clean record with no artifact, std/mad/windowed_mad
    should all recover essentially the same beats -- the fix should not
    change behavior on well-behaved signals."""
    fs = 360.0
    n = int(30.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    signal = 0.01 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.01, size=n)
    rr = int(round(0.8 * fs))
    half_width = max(3, int(round(0.02 * fs)))
    beats = np.arange(rr, n - rr, rr)
    for peak in beats:
        lo, hi = peak - half_width, peak + half_width + 1
        signal[lo:hi] += 1.0 * np.hanning(hi - lo)

    tolerance = int(round(0.075 * fs))
    for method in ("std", "mad", "windowed_mad", "windowed_std"):
        peaks = detect_r_peaks(signal, fs, scale_method=method)
        assert _coverage(peaks, beats, tolerance) > 0.95


def test_estimate_stage1_scale_rejects_unknown_method():
    with pytest.raises(ValueError):
        estimate_stage1_scale(np.ones(1000), 250.0, method="not_a_real_method")


def test_estimate_stage1_scale_windowed_mad_ignores_a_minority_of_bad_windows():
    fs = 257.0
    n = int(60 * fs)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.02, size=n)
    # corrupt ~5% of the record with a large-amplitude burst
    z[int(10 * fs):int(13 * fs)] += rng.normal(0, 5.0, size=int(3 * fs))

    std_scale = estimate_stage1_scale(z, fs, method="std")
    windowed_scale = estimate_stage1_scale(z, fs, method="windowed_mad")

    # global std is dragged far above the signal's true noise floor (~0.02);
    # windowed_mad should stay close to it.
    assert std_scale > 0.3
    assert windowed_scale < 0.05


def test_adaptive_falls_back_to_windowed_std_only_when_burst_present():
    fs = 257.0
    n = int(60 * fs)
    rng = np.random.default_rng(0)

    # No burst: adaptive should behave like plain std (no fallback triggered).
    clean = rng.normal(0, 0.02, size=n)
    std_scale = estimate_stage1_scale(clean, fs, method="std")
    adaptive_scale = estimate_stage1_scale(clean, fs, method="adaptive")
    assert adaptive_scale == pytest.approx(std_scale, rel=0.05)

    # With a short severe burst: adaptive should detect the inflation and
    # fall back to windowed_std, landing far below the burst-inflated std.
    bursty = clean.copy()
    bursty[int(10 * fs):int(13 * fs)] += rng.normal(0, 5.0, size=int(3 * fs))
    std_scale_bursty = estimate_stage1_scale(bursty, fs, method="std")
    adaptive_scale_bursty = estimate_stage1_scale(bursty, fs, method="adaptive")
    windowed_std_bursty = estimate_stage1_scale(bursty, fs, method="windowed_std")
    assert std_scale_bursty > 0.3  # inflated, as before
    assert adaptive_scale_bursty == pytest.approx(windowed_std_bursty, rel=1e-9)
    assert adaptive_scale_bursty < std_scale_bursty / 2
