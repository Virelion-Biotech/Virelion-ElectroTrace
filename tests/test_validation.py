import numpy as np
import pytest

from electrotrace.validation import match_peaks, summarize_records, DetectionMetrics, RecordValidation


def test_match_peaks_one_to_one_with_tolerance():
    result = match_peaks(
        detected_samples=[99, 201, 500],
        reference_samples=[100, 200, 400],
        fs_hz=1000,
        tolerance_ms=5,
    )
    assert result.reference_count == 3
    assert result.detected_count == 3
    assert result.true_positive == 2
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.sensitivity == pytest.approx(2 / 3)
    assert result.positive_predictive_value == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.mean_timing_error_ms == pytest.approx(0.0)
    assert result.median_timing_error_ms == pytest.approx(0.0)
    assert result.median_absolute_timing_error_ms == pytest.approx(1.0)
    assert result.max_absolute_timing_error_ms == pytest.approx(1.0)


def test_match_peaks_rejects_invalid_settings_and_indices():
    with pytest.raises(ValueError):
        match_peaks([1], [1], fs_hz=0)
    with pytest.raises(ValueError):
        match_peaks([1], [1], fs_hz=500, tolerance_ms=0)
    with pytest.raises(ValueError, match="integer-valued"):
        match_peaks([1.5], [1], fs_hz=500)
    with pytest.raises(ValueError, match="non-negative"):
        match_peaks([-1], [1], fs_hz=500)
    with pytest.raises(ValueError, match="strictly increasing"):
        match_peaks([1, 1], [1], fs_hz=500)
    with pytest.raises(ValueError, match="strictly increasing"):
        match_peaks([2, 1], [1], fs_hz=500)


def test_summary_aggregates_counts():
    def metrics(ref, det, tp, fp, fn, mean, median, sd, mae, medae, p95, maxerr):
        sens = tp / ref if ref else 0.0
        ppv = tp / det if det else 0.0
        f1 = 2 * sens * ppv / (sens + ppv) if sens + ppv else 0.0
        return DetectionMetrics(ref, det, tp, fp, fn, sens, ppv, f1, mean, median, sd, medae, mae, p95, maxerr)

    r1 = RecordValidation("a", 360.0, metrics(10, 10, 9, 1, 1, 2.0, 2.0, 1.0, 2.5, 2.0, 4.0, 5.0))
    r2 = RecordValidation("b", 360.0, metrics(20, 21, 18, 3, 2, -1.0, -1.0, 1.5, 2.0, 1.0, 5.0, 7.0))
    summary = summarize_records([r1, r2])
    assert summary["records"] == 2
    assert summary["reference_count"] == 30
    assert summary["detected_count"] == 31
    assert summary["true_positive"] == 27
    assert summary["false_positive"] == 4
    assert summary["false_negative"] == 3


def test_match_peaks_handles_empty_reference_and_detected():
    result = match_peaks([], [], fs_hz=500)
    assert result.true_positive == 0
    assert result.f1 == 0.0
    assert result.mean_timing_error_ms is None
