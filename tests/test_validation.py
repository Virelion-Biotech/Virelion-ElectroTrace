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
    assert result.median_timing_error_ms == pytest.approx(1.0)


def test_match_peaks_rejects_invalid_settings():
    with pytest.raises(ValueError):
        match_peaks([1], [1], fs_hz=0)
    with pytest.raises(ValueError):
        match_peaks([1], [1], fs_hz=500, tolerance_ms=0)


def test_summary_aggregates_counts():
    r1 = RecordValidation("a", 360.0, DetectionMetrics(10, 10, 9, 1, 1, .9, .9, .9, 2., 4.))
    r2 = RecordValidation("b", 360.0, DetectionMetrics(20, 21, 18, 3, 2, .9, 18 / 21, 2 * .9 * (18 / 21) / (.9 + 18 / 21), 3., 5.))
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
