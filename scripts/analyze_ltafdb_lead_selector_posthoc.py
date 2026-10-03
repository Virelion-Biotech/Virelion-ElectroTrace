#!/usr/bin/env python3
"""Post-hoc comparator audit for the exposed LTAFDB selector run.

LTAFDB is already exposed by the immutable first prospective selector run.
This script may compare both leads, but it must not alter or relabel that
prospective result. It first reproduces the archived selected-lead result
record-by-record, then reports fixed-channel and oracle comparators.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.lead_selection import choose_two_lead_channel
from electrotrace.validation import RecordValidation, match_peaks, summarize_records
from electrotrace.validation_detectors import detect_r_peaks_two_stage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_prospective = importlib.import_module("scripts.evaluate_ltafdb_lead_selector_prospective")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_first_run(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "electrotrace.external_ltafdb_lead_selector_validation/v1":
        raise SystemExit("unexpected LTAFDB first-run schema")
    if report.get("evidence_status") != "prospective_external_lead_selector_evaluation_first_run":
        raise SystemExit("input is not the archived first prospective selector run")
    integrity = report.get("evaluation_integrity", {})
    if integrity.get("records_scored") != 84 or integrity.get("records_skipped") != 0:
        raise SystemExit("archived prospective report must contain all 84 records")
    if integrity.get("lead_selection_used_reference_annotations") is not False:
        raise SystemExit("archived run does not prove label-free lead selection")
    if integrity.get("reference_annotations_loaded_after_lead_selection") is not True:
        raise SystemExit("archived run does not prove annotation-after-selection ordering")
    if integrity.get("retraining") is not False:
        raise SystemExit("archived prospective report unexpectedly indicates retraining")
    return report


def _same_number(actual, expected, *, atol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _summary(records: list[RecordValidation]) -> dict:
    out = summarize_records(records)
    f1s = np.asarray([r.metrics.f1 for r in records], dtype=float)
    out["macro_mean_record_f1"] = float(np.mean(f1s))
    out["macro_median_record_f1"] = float(np.median(f1s))
    out["min_record_f1"] = float(np.min(f1s))
    out["max_record_f1"] = float(np.max(f1s))
    return out


def _metric_payload(metrics) -> dict:
    return metrics.to_dict()


def _assert_reproduces_archived(record: str, archived: dict, selected_channel: int, p50: list[float], metrics) -> None:
    if int(archived["selected_channel"]) != int(selected_channel):
        raise SystemExit(
            f"{record}: recomputed selector chose channel {selected_channel}, "
            f"archived run chose {archived['selected_channel']}"
        )
    for idx in (0, 1):
        key = f"channel{idx}_retained_probability_p50"
        if not _same_number(p50[idx], archived[key]):
            raise SystemExit(
                f"{record}: recomputed {key}={p50[idx]} does not match archived {archived[key]}"
            )
    for key in (
        "reference_count", "detected_count", "true_positive", "false_positive",
        "false_negative", "sensitivity", "positive_predictive_value", "f1",
    ):
        actual = getattr(metrics, key)
        expected = archived[key]
        if isinstance(expected, int):
            same = int(actual) == int(expected)
        else:
            same = _same_number(actual, expected)
        if not same:
            raise SystemExit(
                f"{record}: selected-lead {key}={actual} does not reproduce archived {expected}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltafdb-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--polarity-threshold-report", type=Path, required=True)
    ap.add_argument("--first-run-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    first = load_first_run(args.first_run_report)
    protocol = _prospective.load_locked_protocol()
    records = list(protocol["dataset"]["records"])
    if [r["record"] for r in first["record_results"]] != records:
        raise SystemExit("archived first-run record order does not match locked protocol")

    model = CandidateSuppressor.load(args.model)
    _prospective.verify_model(model, protocol)
    derivation, width_gate, v2_gate = _prospective._load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("polarity derivation schema mismatch")
    if not _same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("v2 gate mismatch")
    if not _same_number(width_gate, detector["expected_width_override_confidence"]):
        raise SystemExit("width gate mismatch")

    archived_by = {str(r["record"]): r for r in first["record_results"]}
    window = protocol["dataset"]["window"]
    start = int(window["start_sample"])
    stop = int(window["stop_sample_exclusive"])
    fs = float(protocol["dataset"]["expected_sampling_frequency_hz"])
    scale_method = str(detector["stage1_scale_method"])
    tolerance_ms = float(detector["tolerance_ms"])
    beat_symbols = frozenset(detector["beat_symbols"])
    selector_cfg = protocol["lead_selector"]

    channel0_results = []
    channel1_results = []
    selected_results = []
    oracle_results = []
    rows = []

    for i, record in enumerate(records, start=1):
        rec = wfdb.rdrecord(
            str(args.ltafdb_dir / record),
            sampfrom=start,
            sampto=stop,
            channels=[0, 1],
            physical=True,
        )
        if not _same_number(rec.fs, fs):
            raise SystemExit(f"{record}: unexpected sampling frequency {rec.fs}")
        signals = np.asarray(rec.p_signal, dtype=float)
        if signals.shape != (stop - start, 2) or not np.isfinite(signals).all():
            raise SystemExit(f"{record}: invalid frozen signal window {signals.shape}")

        detections = []
        p50 = []
        for ch in (0, 1):
            retained, probabilities = detect_r_peaks_two_stage(
                signals[:, ch],
                fs,
                model,
                polarity=str(detector["polarity"]),
                recovery=bool(detector["recovery"]),
                scale_method=scale_method,
                width_override_confidence=width_gate,
                v2_gate_confidence=v2_gate,
                threshold=float(model.metadata.threshold),
            )
            detections.append(np.asarray(retained, dtype=int))
            p50.append(
                _prospective.retained_probability_p50(
                    probabilities,
                    empty_value=float(selector_cfg["empty_retained_probability_p50"]),
                )
            )

        selected = choose_two_lead_channel(
            p50[0],
            p50[1],
            primary_floor=float(selector_cfg["primary_retained_probability_p50_floor"]),
        )

        annotation = wfdb.rdann(
            str(args.ltafdb_dir / record),
            str(detector["annotation_extension"]),
            sampfrom=start,
            sampto=stop,
        )
        reference, _ = _prospective._reference_from_annotation(
            annotation, beat_symbols, stop - start
        )
        metrics = [
            match_peaks(detections[ch], reference, fs, tolerance_ms=tolerance_ms)
            for ch in (0, 1)
        ]
        archived = archived_by[record]
        _assert_reproduces_archived(record, archived, selected, p50, metrics[selected])

        oracle = 1 if metrics[1].f1 > metrics[0].f1 else 0
        channel0_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[0]))
        channel1_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[1]))
        selected_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[selected]))
        oracle_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[oracle]))

        rows.append({
            "record": record,
            "selected_channel": int(selected),
            "oracle_channel_posthoc": int(oracle),
            "channel0_retained_probability_p50": p50[0],
            "channel1_retained_probability_p50": p50[1],
            "channel0": _metric_payload(metrics[0]),
            "channel1": _metric_payload(metrics[1]),
            "selected": _metric_payload(metrics[selected]),
            "selected_minus_channel0_f1": float(metrics[selected].f1 - metrics[0].f1),
            "oracle_minus_selected_f1": float(metrics[oracle].f1 - metrics[selected].f1),
        })
        print(f"[{i:02d}/84] {record}: reproduced", flush=True)

    switched = [r for r in rows if r["selected_channel"] == 1]
    report = {
        "schema": "electrotrace.ltafdb_lead_selector_posthoc_audit/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "posthoc_exposed_dataset_comparator_analysis",
        "git_head": git_head(),
        "guardrail": (
            "LTAFDB is exposed. This comparator analysis must not alter or replace "
            "the immutable prospective selector result and cannot be used to retune "
            "selector v1 while retaining an LTAFDB prospective claim."
        ),
        "inputs": {
            "first_run_report": str(args.first_run_report),
            "first_run_sha256": _prospective.sha256_file(args.first_run_report),
            "model": str(args.model),
            "model_sha256": _prospective.sha256_file(args.model),
            "polarity_derivation": str(args.polarity_threshold_report),
            "polarity_derivation_sha256": _prospective.sha256_file(args.polarity_threshold_report),
        },
        "reproduction": {
            "all_84_archived_selected_lead_results_reproduced": True,
            "selected_channel_and_p50_reproduced": True,
        },
        "summaries": {
            "channel0_fixed_posthoc": _summary(channel0_results),
            "channel1_fixed_posthoc": _summary(channel1_results),
            "prospective_selector_v1_reproduced": _summary(selected_results),
            "oracle_best_channel_posthoc": _summary(oracle_results),
        },
        "switched_records": switched,
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nComparator summaries:")
    print(json.dumps(report["summaries"], indent=2, sort_keys=True))
    print("Switched records:")
    for row in switched:
        print(
            row["record"],
            "delta_vs_ch0=",
            f"{row['selected_minus_channel0_f1']:.6f}",
            "selected_f1=",
            f"{row['selected']['f1']:.6f}",
            "ch0_f1=",
            f"{row['channel0']['f1']:.6f}",
        )
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
