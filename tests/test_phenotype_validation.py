import pytest

from electrotrace.phenotype_validation import aggregate_by_unit, quality_report


def test_quality_report_detects_structural_errors():
    rows = [
        {"r_index": 1, "r_time_s": 0.1, "heart_rate_bpm": 60.0, "rr_prev_s": 1.0, "r_amplitude": 1.2},
        {"r_index": 1, "r_time_s": 0.05, "heart_rate_bpm": 80.0, "rr_prev_s": 1.0, "r_amplitude": 1.0},
    ]
    report = quality_report(rows)
    assert report["duplicate_r_indices"] == 1
    assert report["time_nonmonotonic"] == 1
    assert report["inconsistent_heart_rate"] == 1
    assert report["valid"] is False


def test_unit_aggregation_prevents_beat_level_pseudoreplication():
    rows = [
        {"heart_rate_bpm": 60, "rr_prev_s": 1.0, "r_amplitude": 1.0},
        {"heart_rate_bpm": 66, "rr_prev_s": 0.9, "r_amplitude": 1.2},
        {"heart_rate_bpm": 90, "rr_prev_s": 0.67, "r_amplitude": 0.8},
    ]
    aggregated = aggregate_by_unit(rows, ["subject-a", "subject-a", "subject-b"])
    assert [row["unit_id"] for row in aggregated] == ["subject-a", "subject-b"]
    assert aggregated[0]["n_beats"] == 2
    assert aggregated[0]["heart_rate_bpm"]["n"] == 2


def test_unit_aggregation_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        aggregate_by_unit([{"heart_rate_bpm": 60}], [])
