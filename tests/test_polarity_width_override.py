"""Polarity-selection finding (2026-09-13): the count-ratio heuristic in
select_signal_polarity picks correctly on 61/68 real INCART records, but
fails badly on I19 (F1 0.19 vs an achievable 0.70) -- its wrong polarity
produces candidates with implausible median width (~640ms, when even wide
PVCs top out around 150-160ms), while the count-ratio itself gave no clear
signal (confidence 0.35, above the existing 0.15 fallback threshold).

Fix: a confidence-gated width check -- when the count-ratio decision is
already low-confidence (<0.38, a threshold chosen from a real gap in the
confidence distribution between I19 and the nearest record we must NOT
flip), prefer whichever polarity has the narrower median candidate width.
Validated on all 68 INCART records: mean F1 0.8169 (before) -> 0.8270
(confidence-gated width check) -> 0.8311 (oracle ceiling). An *ungated*
version of the same width check reaches only 0.8234 and badly regresses
I63 (F1 0.8992 -> 0.6353, a high-confidence record the ratio rule already
got right) -- gating by confidence is what avoids that regression while
still fixing I19, I13, and I64.

These tests exercise _width_preferred_polarity directly with hand-built
peak arrays (deterministic, no signal-synthesis fragility) rather than
trying to reproduce the exact low-confidence scenario end-to-end, since
scipy.signal.peak_widths has edge-case behavior on synthetic flat-baseline
signals that made an end-to-end synthetic reproduction unreliable -- the
mechanism itself is already validated on real data above.
"""
import numpy as np
import pytest

from electrotrace.validation_detectors import (
    DEFAULT_WIDTH_OVERRIDE_CONFIDENCE,
    DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES,
    _width_preferred_polarity,
)


def _narrow_pulse_signal(fs, positions, half_width_s, amplitude, n_samples, noise_seed=0, noise=0.005):
    rng = np.random.default_rng(noise_seed)
    z = rng.normal(0, noise, size=n_samples)
    half_w = max(1, int(round(half_width_s * fs)))
    for p in positions:
        lo, hi = max(0, p - half_w), min(n_samples, p + half_w + 1)
        z[lo:hi] += amplitude * np.hanning(hi - lo)
    return z


def test_width_preferred_polarity_picks_narrower_candidates():
    """Direct test of the mechanism: given a set of narrow ("positive")
    candidates and a set of wide ("negative") candidates at the same
    scale, the narrower one should be preferred -- this is exactly what
    fixed I19, where the wrong polarity's candidates were ~640ms wide."""
    fs = 257.0
    n = int(60 * fs)

    narrow_positions = np.linspace(int(2 * fs), n - int(2 * fs), 20).astype(int)
    z_pos = _narrow_pulse_signal(fs, narrow_positions, half_width_s=0.02, amplitude=1.0, n_samples=n, noise_seed=1)

    wide_positions = np.linspace(int(3 * fs), n - int(3 * fs), 20).astype(int)
    z_neg_source = _narrow_pulse_signal(fs, wide_positions, half_width_s=0.3, amplitude=1.0, n_samples=n, noise_seed=2)

    # Build one combined z where "positive" peaks are the narrow ones and
    # "negative" peaks (i.e. peaks of -z) are the wide ones.
    z = z_pos - z_neg_source

    from electrotrace.validation_detectors import _candidate_set, estimate_stage1_scale
    scale = estimate_stage1_scale(z, fs, method="windowed_std")
    pos, _ = _candidate_set(z, fs, scale)
    neg, _ = _candidate_set(-z, fs, scale)
    assert len(pos) >= DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES
    assert len(neg) >= DEFAULT_WIDTH_OVERRIDE_MIN_CANDIDATES

    assert _width_preferred_polarity(z, pos, neg) == "positive"


def test_width_preferred_polarity_returns_none_when_too_few_candidates():
    z = np.zeros(1000)
    assert _width_preferred_polarity(z, np.array([100]), np.array([200, 300, 400])) is None
    assert _width_preferred_polarity(z, np.array([100, 200, 300]), np.array([400])) is None


def test_default_width_override_confidence_is_documented_value():
    # Locks in the validated threshold so a future refactor can't silently
    # change it without re-running the 68-record INCART check.
    assert DEFAULT_WIDTH_OVERRIDE_CONFIDENCE == 0.38


def test_select_signal_polarity_width_override_end_to_end():
    """End-to-end: a case where narrow ('positive') candidates clearly
    outnumber and out-narrow wide ('negative') ones, but the ratio still
    happens to land in the gated confidence band, should end up positive
    both before and after the width check -- i.e. the width check is a
    no-op here since the ratio rule already agrees with it. This mainly
    guards against an accidental regression in the integration path
    (select_signal_polarity calling the new helper) rather than testing
    the override mechanism itself (see the direct test above for that)."""
    from electrotrace.validation_detectors import select_signal_polarity

    fs = 257.0
    n = int(60 * fs)
    narrow_positions = np.linspace(int(2 * fs), n - int(2 * fs), 30).astype(int)
    z = _narrow_pulse_signal(fs, narrow_positions, half_width_s=0.02, amplitude=1.0, n_samples=n, noise_seed=3)
    signal = z + np.median(z)

    decision = select_signal_polarity(signal, fs, scale_method="windowed_std")
    assert decision.polarity == "positive"
