import json

import pytest

from scripts import evaluate_frozen_model_incart_complete as incart


def _audit_row(record, *, action, coverage=None, reason=None):
    return {
        "record": record,
        "action": action,
        "reason": reason,
        "alignment_checked": coverage is not None,
        "alignment_coverage": coverage,
        "alignment_median_offset_ms": 0.0 if coverage is not None else None,
        "n_leading_invalid": 1 if record in incart.MALFORMED_REFERENCE_RECORDS else 0,
        "n_trailing_invalid": 0,
        "n_interior_invalid": 0,
        "n_reference_kept": 100,
    }


def _certified_report(approve_i57=False):
    rows = {}
    for detector in ("gqrs", "sqrs"):
        detector_rows = []
        for record in sorted(incart.MALFORMED_REFERENCE_RECORDS):
            if record == "I57" and not approve_i57:
                detector_rows.append(
                    _audit_row(
                        record,
                        action="excluded",
                        coverage=0.32,
                        reason="repair rejected",
                    )
                )
            else:
                detector_rows.append(
                    _audit_row(record, action="kept_after_dropping_edges", coverage=0.95)
                )
        rows[detector] = {
            "annotation_audit": {"records": detector_rows},
            "records": [],
        }
    return {
        "schema": incart.EXPECTED_AUDIT_SCHEMA,
        "protocol": {
            "annotation_policy": "drop_edges",
            "tolerance_ms": 75.0,
            "channel": 0,
        },
        "results": rows,
    }


def test_certified_audit_authorizes_only_independently_aligned_repairs(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_certified_report()), encoding="utf-8")

    _, authorization = incart.load_certified_repair_audit(path)

    assert incart.certified_authorized(authorization["I04"])
    assert not incart.certified_authorized(authorization["I57"])
    assert {x["detector"] for x in authorization["I04"] if x["approved"]} == {
        "gqrs",
        "sqrs",
    }


def test_certified_audit_can_authorize_i57_only_when_evidence_changes(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_certified_report(approve_i57=True)), encoding="utf-8")

    _, authorization = incart.load_certified_repair_audit(path)

    assert incart.certified_authorized(authorization["I57"])


def test_certified_audit_fails_closed_on_wrong_protocol(tmp_path):
    report = _certified_report()
    report["protocol"]["tolerance_ms"] = 100.0
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit, match="tolerance mismatch"):
        incart.load_certified_repair_audit(path)


def test_bootstrap_macro_is_record_level_and_deterministic():
    rows = [
        {"sensitivity": 0.8, "positive_predictive_value": 0.9, "f1": 0.85},
        {"sensitivity": 1.0, "positive_predictive_value": 0.7, "f1": 0.82},
        {"sensitivity": 0.6, "positive_predictive_value": 1.0, "f1": 0.75},
    ]

    first = incart._bootstrap_macro(rows, replicates=200, seed=42)
    second = incart._bootstrap_macro(rows, replicates=200, seed=42)

    assert first == second
    assert first["unit"] == "record"
    assert first["replicates"] == 200
    assert set(first["intervals"]) == {
        "sensitivity",
        "positive_predictive_value",
        "f1",
    }


def test_canonical_incart_manifest_is_exactly_75_records():
    assert len(incart.CANONICAL_RECORDS) == 75
    assert incart.CANONICAL_RECORDS[0] == "I01"
    assert incart.CANONICAL_RECORDS[-1] == "I75"
    assert len(set(incart.CANONICAL_RECORDS)) == 75


def test_certified_consensus_requires_both_detectors(tmp_path):
    path = tmp_path / "audit.json"
    report = _certified_report(approve_i57=True)
    report["results"]["sqrs"]["annotation_audit"]["records"] = [
        {
            **row,
            "action": "excluded",
            "alignment_coverage": 0.2,
        }
        if row["record"] == "I57"
        else row
        for row in report["results"]["sqrs"]["annotation_audit"]["records"]
    ]
    path.write_text(json.dumps(report), encoding="utf-8")
    _, authorization = incart.load_certified_repair_audit(path)

    assert incart.certified_authorized(authorization["I57"])
    assert not incart.certified_consensus_authorized(authorization["I57"])


def test_certified_consensus_accepts_two_independent_approvals(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(_certified_report(approve_i57=True)),
        encoding="utf-8",
    )
    _, authorization = incart.load_certified_repair_audit(path)

    assert incart.certified_consensus_authorized(authorization["I57"])
