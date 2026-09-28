import json

import pytest

from scripts import analyze_edb_posthoc_failures as edb


def _first_run():
    return {
        "schema": "electrotrace.external_edb_prospective_validation/v1",
        "evidence_status": "prospective_external_evaluation_first_run",
        "evaluation_integrity": {
            "records_scored": 90,
            "records_skipped": 0,
            "retraining": False,
        },
        "record_results": [
            {"record": "good", "f1": 0.95},
            {"record": "bad2", "f1": 0.10},
            {"record": "bad1", "f1": 0.40},
        ],
    }


def _primary(*, f1=0.3, sens=0.2, ppv=0.9, stage1=0.2, coverage=0.2, suppression=0.3):
    return {
        "metrics": {
            "f1": f1,
            "sensitivity": sens,
            "positive_predictive_value": ppv,
        },
        "stage1_candidates_over_reference": stage1,
        "candidate_reference_coverage": coverage,
        "suppression_rate": suppression,
    }


def test_load_first_run_requires_complete_prospective_report(tmp_path):
    p = tmp_path / "first.json"
    p.write_text(json.dumps(_first_run()), encoding="utf-8")
    report = edb.load_first_run(p)
    assert report["evaluation_integrity"]["records_scored"] == 90

    broken = _first_run()
    broken["evaluation_integrity"]["records_skipped"] = 1
    p.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="complete 90-record cohort"):
        edb.load_first_run(p)


def test_low_tail_selection_uses_archived_f1_and_is_sorted():
    selected = edb.select_low_tail(_first_run(), 0.80)
    assert [r["record"] for r in selected] == ["bad2", "bad1"]


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        (_primary(stage1=0.2, coverage=0.2), "candidate_starvation"),
        (
            _primary(stage1=0.9, coverage=0.3),
            "candidate_misalignment_or_wrong_deflections",
        ),
        (
            _primary(stage1=1.1, coverage=0.9, suppression=0.8, sens=0.2),
            "stage2_over_suppression",
        ),
        (
            _primary(stage1=1.1, coverage=0.7, suppression=0.2, sens=0.4, ppv=0.3),
            "over_detection_or_mixed_fp",
        ),
        (
            _primary(stage1=1.1, coverage=0.7, suppression=0.2, sens=0.2, ppv=0.9),
            "mixed_or_other",
        ),
        (
            _primary(f1=0.9, stage1=0.1, coverage=0.1),
            "not_low_tail",
        ),
    ],
)
def test_primary_failure_taxonomy(primary, expected):
    assert edb.classify_primary(primary, cutoff=0.80) == expected


def test_build_row_marks_alternate_lead_delta():
    archived = {"f1": 0.2}
    primary = _primary(f1=0.2, sens=0.1, ppv=0.9, stage1=0.2, coverage=0.2)
    primary["stage2_candidate_scoring"] = {"auc": 0.8, "oracle_best_f1": 0.5}
    alternate = {
        "metrics": {
            "f1": 0.8,
            "sensitivity": 0.8,
            "positive_predictive_value": 0.8,
        }
    }
    row = edb.build_row("eXXXX", archived, primary, alternate, 0.80)
    assert row["channel1_minus_channel0_f1"] == pytest.approx(0.6)
    assert row["heuristic_failure_mode"] == "candidate_starvation"
