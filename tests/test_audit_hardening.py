import numpy as np

from electrotrace.phenotype_validation import aggregate_by_unit, quality_report
from electrotrace.provenance import DatasetManifest


def test_manifest_hash_is_stable_across_generation_timestamps():
    base = dict(
        dataset_id="fixture",
        dataset_version="1",
        source="test",
        records=("a", "b"),
        split_manifest={"test": ("a", "b")},
        software_version="1.7.0",
        software_commit="abc",
    )
    first = DatasetManifest(**base, generated_at_utc="2026-08-19T00:00:00+00:00")
    second = DatasetManifest(**base, generated_at_utc="2026-08-19T01:00:00+00:00")
    assert first.sha256() == second.sha256()


def test_phenotype_qc_rejects_fractional_and_negative_indices():
    rows = [
        {"r_index": 1.5, "r_time_s": 0.1, "heart_rate_bpm": 60.0, "r_amplitude": 1.0},
        {"r_index": -1, "r_time_s": 0.2, "heart_rate_bpm": 60.0, "r_amplitude": 1.0},
    ]
    report = quality_report(rows)
    assert report["invalid_r_indices"] == 2
    assert report["valid"] is False


def test_unit_aggregation_rejects_empty_identifier():
    try:
        aggregate_by_unit([{"heart_rate_bpm": 60}], [""])
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty experimental-unit identifier was accepted")
