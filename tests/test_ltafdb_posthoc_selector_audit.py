import json

import pytest

from electrotrace.validation import match_peaks
from scripts import analyze_ltafdb_lead_selector_posthoc as audit


def _first_run():
    return {
        "schema": "electrotrace.external_ltafdb_lead_selector_validation/v1",
        "evidence_status": "prospective_external_lead_selector_evaluation_first_run",
        "evaluation_integrity": {
            "records_scored": 84,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "retraining": False,
        },
    }


def test_load_first_run_requires_clean_complete_prospective_artifact(tmp_path):
    p = tmp_path / "first.json"
    p.write_text(json.dumps(_first_run()), encoding="utf-8")
    assert audit.load_first_run(p)["evaluation_integrity"]["records_scored"] == 84

    broken = _first_run()
    broken["evaluation_integrity"]["lead_selection_used_reference_annotations"] = True
    p.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="label-free"):
        audit.load_first_run(p)


def test_assert_reproduces_archived_accepts_identical_selected_result():
    metrics = match_peaks([100, 300], [100, 300], 128.0, tolerance_ms=75.0)
    archived = {
        "selected_channel": 1,
        "channel0_retained_probability_p50": 0.7,
        "channel1_retained_probability_p50": 0.9,
        **metrics.to_dict(),
    }
    audit._assert_reproduces_archived(
        "x", archived, 1, [0.7, 0.9], metrics
    )


def test_assert_reproduces_archived_rejects_channel_or_metric_drift():
    metrics = match_peaks([100, 300], [100, 300], 128.0, tolerance_ms=75.0)
    archived = {
        "selected_channel": 0,
        "channel0_retained_probability_p50": 0.7,
        "channel1_retained_probability_p50": 0.9,
        **metrics.to_dict(),
    }
    with pytest.raises(SystemExit, match="recomputed selector chose channel"):
        audit._assert_reproduces_archived("x", archived, 1, [0.7, 0.9], metrics)

    archived["selected_channel"] = 1
    archived["f1"] = 0.5
    with pytest.raises(SystemExit, match="does not reproduce archived"):
        audit._assert_reproduces_archived("x", archived, 1, [0.7, 0.9], metrics)
