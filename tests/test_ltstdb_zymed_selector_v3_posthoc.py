import json

import pytest

from electrotrace.validation import match_peaks
from scripts import analyze_ltstdb_zymed_selector_v3_posthoc as audit


def _first_run():
    return {
        "schema": "electrotrace.external_ltstdb_zymed_selector_v3_validation/v1",
        "evidence_status": "prospective_external_selector_v3_evaluation_first_run",
        "evaluation_integrity": {
            "records_scored": 18,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "third_channel_loaded": False,
            "retraining": False,
        },
    }


def test_load_first_run_requires_complete_label_free_prospective_artifact(tmp_path):
    p = tmp_path / "first.json"
    p.write_text(json.dumps(_first_run()), encoding="utf-8")
    report = audit.load_first_run(p)
    assert report["evaluation_integrity"]["records_scored"] == 18

    broken = _first_run()
    broken["evaluation_integrity"]["third_channel_loaded"] = True
    p.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="channel-2 exclusion"):
        audit.load_first_run(p)


def test_quality_differences_empty_for_exact_reproduction():
    archived = {
        "retained_probability_p50": 0.9,
        "retained_qrs_band_fraction": 0.6,
        "retention_fraction": 0.5,
        "stage1_candidate_count": 10,
        "retained_count": 5,
        "selected_polarity": "positive",
        "polarity_confidence": 0.8,
    }
    assert audit._quality_differences(dict(archived), archived) == []


def test_quality_differences_report_drift_without_hiding_it():
    archived = {
        "retained_probability_p50": 0.9,
        "retained_qrs_band_fraction": 0.6,
        "retention_fraction": 0.5,
        "stage1_candidate_count": 10,
        "retained_count": 5,
        "selected_polarity": "positive",
        "polarity_confidence": 0.8,
    }
    actual = dict(archived)
    actual["retained_count"] = 4
    differences = audit._quality_differences(actual, archived)
    assert differences == [
        {
            "field": "retained_count",
            "actual": 4,
            "archived": 5,
            "absolute_difference": 1.0,
        }
    ]


def test_assert_selected_metrics_accepts_exact_result():
    metrics = match_peaks([100, 300], [100, 300], 250.0, tolerance_ms=75.0)
    audit._assert_selected_metrics("s30661", metrics, metrics.to_dict())


def test_assert_selected_metrics_rejects_drift():
    metrics = match_peaks([100, 300], [100, 300], 250.0, tolerance_ms=75.0)
    archived = metrics.to_dict()
    archived["false_positive"] = 1
    with pytest.raises(SystemExit, match="does not reproduce archived"):
        audit._assert_selected_metrics("s30661", metrics, archived)


def test_remote_retry_recovers_transient_failure(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary 502")
        return "ok"

    monkeypatch.setattr(audit.time, "sleep", sleeps.append)

    assert audit._call_with_retry(flaky, attempts=4, base_delay_s=0.5) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_remote_retry_reraises_last_failure(monkeypatch):
    monkeypatch.setattr(audit.time, "sleep", lambda _: None)

    def broken():
        raise RuntimeError("persistent failure")

    with pytest.raises(RuntimeError, match="persistent failure"):
        audit._call_with_retry(broken, attempts=2, base_delay_s=0)

def test_verify_local_source_identity_accepts_exact_hashes(monkeypatch, tmp_path):
    first = {"dataset_input_hashes": {"RECORDS": "a" * 64, "s30671.dat": "b" * 64}}
    monkeypatch.setattr(
        audit._prospective,
        "verify_dataset",
        lambda root, protocol: dict(first["dataset_input_hashes"]),
    )
    actual = audit._verify_local_source_identity(tmp_path, {"dataset": {}}, first)
    assert actual == first["dataset_input_hashes"]


def test_verify_local_source_identity_rejects_byte_drift(monkeypatch, tmp_path):
    first = {"dataset_input_hashes": {"RECORDS": "a" * 64, "s30671.dat": "b" * 64}}
    monkeypatch.setattr(
        audit._prospective,
        "verify_dataset",
        lambda root, protocol: {"RECORDS": "a" * 64, "s30671.dat": "c" * 64},
    )
    with pytest.raises(SystemExit, match="do not match the immutable first-run hashes"):
        audit._verify_local_source_identity(tmp_path, {"dataset": {}}, first)


def test_verify_local_source_identity_requires_first_run_hash_manifest(tmp_path):
    with pytest.raises(SystemExit, match="lacks dataset_input_hashes"):
        audit._verify_local_source_identity(tmp_path, {"dataset": {}}, {})

