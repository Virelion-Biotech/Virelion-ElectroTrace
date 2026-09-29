import json

import pytest

from electrotrace.validation import match_peaks
from scripts import analyze_svdb_selector_v2_posthoc as audit


def _first_run():
    return {
        "schema": "electrotrace.external_svdb_selector_v2_validation/v1",
        "evidence_status": "prospective_external_selector_v2_evaluation_first_run",
        "evaluation_integrity": {
            "records_scored": 78,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "retraining": False,
        },
    }


def test_load_first_run_requires_complete_label_free_prospective_artifact(tmp_path):
    p = tmp_path / "first.json"
    p.write_text(json.dumps(_first_run()), encoding="utf-8")
    assert audit.load_first_run(p)["evaluation_integrity"]["records_scored"] == 78

    broken = _first_run()
    broken["evaluation_integrity"]["lead_selection_used_reference_annotations"] = True
    p.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="label-free"):
        audit.load_first_run(p)


def test_assert_quality_accepts_exact_reproduction():
    archived = {
        "retained_probability_p50": 0.9,
        "retained_qrs_band_fraction": 0.6,
        "retention_fraction": 0.5,
        "stage1_candidate_count": 10,
        "retained_count": 5,
        "selected_polarity": "positive",
        "polarity_confidence": 0.8,
    }
    audit._assert_quality("800", 0, dict(archived), archived)


def test_assert_quality_rejects_drift():
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
    actual["retention_fraction"] = 0.4
    with pytest.raises(SystemExit, match="does not reproduce archived"):
        audit._assert_quality("800", 0, actual, archived)


def test_assert_selected_metrics_accepts_exact_result():
    metrics = match_peaks([100, 300], [100, 300], 128.0, tolerance_ms=75.0)
    archived = metrics.to_dict()
    audit._assert_selected_metrics("800", metrics, archived)


def test_assert_selected_metrics_rejects_drift():
    metrics = match_peaks([100, 300], [100, 300], 128.0, tolerance_ms=75.0)
    archived = metrics.to_dict()
    archived["f1"] = 0.5
    with pytest.raises(SystemExit, match="does not reproduce archived"):
        audit._assert_selected_metrics("800", metrics, archived)
