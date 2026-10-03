import numpy as np

from electrotrace.validation_detectors import recover_stage1_candidates


def _signal_with_peaks():
    fs = 360.0
    t = np.arange(7200) / fs
    signal = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    for peak in [900, 1800, 2700, 3600, 4500, 5400, 6300]:
        lo, hi = peak - 4, peak + 5
        signal[lo:hi] += np.hanning(hi - lo) * 2.0
    return signal


def test_long_gap_recovery_adds_at_most_one_candidate_per_long_gap():
    signal = _signal_with_peaks()
    primary = np.array([900, 1800, 3600, 4500, 5400, 6300])
    recovered, prominences = recover_stage1_candidates(
        signal, 360.0, primary, gap_ratio=1.5
    )
    assert recovered.shape == prominences.shape
    assert len(recovered) <= 1
    assert np.isfinite(prominences).all()
    if len(recovered):
        assert 1800 < recovered[0] < 3600


def test_long_gap_recovery_rejects_unsorted_primary_peaks():
    signal = _signal_with_peaks()
    try:
        recover_stage1_candidates(signal, 360.0, np.array([1800, 900]))
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("unsorted primary peaks should be rejected")
