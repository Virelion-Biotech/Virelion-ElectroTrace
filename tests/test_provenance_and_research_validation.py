import json

import pytest

from electrotrace.provenance import DatasetManifest, manifest_from_dict
from electrotrace.research_validation import build_validation_report, mean_ci95, summarize_records_rigorous
from electrotrace.validation import DetectionMetrics, RecordValidation


def _result(record: str, sensitivity: float, ppv: float) -> RecordValidation:
    tp = 100
    ref = int(round(tp / sensitivity))
    det = int(round(tp / ppv))
    fp = det - tp
    fn = ref - tp
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv)
    metrics = DetectionMetrics(
        reference_count=ref,
        detected_count=det,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        sensitivity=sensitivity,
        positive_predictive_value=ppv,
        f1=f1,
        mean_timing_error_ms=0.5,
        median_timing_error_ms=0.5,
        timing_sd_ms=2.0,
        median_absolute_timing_error_ms=5.0,
        mean_absolute_timing_error_ms=6.0,
        p95_absolute_timing_error_ms=12.0,
        max_absolute_timing_error_ms=20.0,
    )
    return RecordValidation(record=record, fs_hz=360.0, metrics=metrics)


def test_manifest_is_deterministic_and_validates_partition_and_file_hashes():
    manifest = DatasetManifest(
        dataset_id="MIT-BIH",
        dataset_version="1.0.0",
        source="PhysioNet",
        records=("100", "101", "102"),
        record_subject_map={"100": "subject-a", "101": "subject-b", "102": "subject-c"},
        input_files={"100.atr": "a" * 64, "100.dat": "b" * 64},
        annotation_policy="accepted beat symbols",
        detector_config={"polarity": "adaptive", "tolerance_ms": 75},
        split_manifest={"train": ("100",), "calibration": ("101",), "test": ("102",)},
        calibration_records=("101",),
        software_version="1.6.1",
        software_commit="abc123",
    )
    first = manifest.sha256()
    second = manifest.sha256()
    assert first == second
    rebuilt = manifest_from_dict(manifest.to_dict())
    assert rebuilt.sha256() == first
    assert rebuilt.record_subject_map["101"] == "subject-b"


def test_manifest_rejects_overlap():
    manifest = DatasetManifest(
        dataset_id="x", dataset_version="1", source="y", records=("a", "b"),
        split_manifest={"train": ("a",), "test": ("a", "b")},
        software_version="1", software_commit="c",
    )
    with pytest.raises(ValueError, match="one split"):
        manifest.validate()


def test_manifest_rejects_misaligned_subject_ids():
    manifest = DatasetManifest(
        dataset_id="x", dataset_version="1", source="y", records=("a", "b"), subject_ids=("only-a",),
        software_version="1", software_commit="c",
    )
    with pytest.raises(ValueError, match="subject_ids"):
        manifest.validate()


def test_manifest_rejects_bad_input_hash():
    manifest = DatasetManifest(
        dataset_id="x", dataset_version="1", source="y", records=("a",), input_files={"a.dat": "not-a-sha"},
        software_version="1", software_commit="c",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        manifest.validate()


def test_mean_ci95_uses_t_interval_for_small_samples():
    result = mean_ci95([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_summarize_records_rigorous_includes_macro_and_bootstrap():
    results = [
        _result("100", 0.90, 0.80),
        _result("101", 0.80, 0.90),
        _result("102", 0.85, 0.85),
    ]
    summary = summarize_records_rigorous(results, n_bootstrap=500, seed=7)
    assert summary["records"] == 3
    assert summary["pooled"]["reference_count"] > 0
    assert summary["macro_record"]["f1"]["n"] == 3
    assert summary["macro_record"]["mean_absolute_timing_error_ms"]["n"] == 3
    assert summary["bootstrap_macro_record_mean"]["f1"]["n_bootstrap"] == 500
    assert summary["bootstrap_macro_record_mean"]["f1"]["estimand"] == "macro_record_mean"


def test_validation_report_is_self_contained_and_json_serializable(tmp_path):
    results = [_result("100", 0.90, 0.80), _result("101", 0.80, 0.90)]
    manifest = DatasetManifest(
        dataset_id="fixture",
        dataset_version="1",
        source="test",
        records=("100", "101", "102"),
        split_manifest={"test": ("100", "101", "102")},
        software_version="1.6.1",
        software_commit="abc",
    )
    report = build_validation_report(
        manifest,
        results,
        detector_name="fixture-detector",
        detector_parameters={"prominence": 0.5},
        annotation_extension="atr",
        beat_symbols=("N",),
        tolerance_ms=75,
        n_bootstrap=250,
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["manifest_sha256"] == manifest.sha256()
    assert loaded["failures"] == ["102"]
