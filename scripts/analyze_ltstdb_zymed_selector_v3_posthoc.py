#!/usr/bin/env python3
"""Post-hoc fixed-lead comparator audit for exposed Zymed LTSTDB selector-v3.

This script is development-only and must only be used after the immutable first
prospective selector-v3 artifact exists. It requires the archived selected
channel and selected-lead metrics to reproduce exactly, while recording any
cross-run drift in label-free selector inputs before reporting fixed-channel
and oracle comparators.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.lead_quality import compute_lead_quality
from electrotrace.lead_selection import choose_two_lead_channel_v3
from electrotrace.validation import RecordValidation, match_peaks, summarize_records

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_prospective = importlib.import_module(
    "scripts.evaluate_ltstdb_zymed_selector_v3_prospective"
)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _call_with_retry(
    func,
    *args,
    attempts: int = 5,
    base_delay_s: float = 2.0,
    **kwargs,
):
    """Retry transient remote WFDB reads without changing scientific inputs."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay_s * attempt
            print(
                f"WFDB remote read failed on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_first_run(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "electrotrace.external_ltstdb_zymed_selector_v3_validation/v1":
        raise SystemExit("unexpected Zymed LTSTDB selector-v3 schema")
    if report.get("evidence_status") != "prospective_external_selector_v3_evaluation_first_run":
        raise SystemExit("input is not the archived first prospective selector-v3 run")
    integrity = report.get("evaluation_integrity", {})
    if integrity.get("records_scored") != 18 or integrity.get("records_skipped") != 0:
        raise SystemExit("archived report must contain all 18 records")
    if integrity.get("lead_selection_used_reference_annotations") is not False:
        raise SystemExit("archived run does not prove label-free lead selection")
    if integrity.get("reference_annotations_loaded_after_lead_selection") is not True:
        raise SystemExit("archived run does not prove annotation-after-selection ordering")
    if integrity.get("third_channel_loaded") is not False:
        raise SystemExit("archived run does not prove channel-2 exclusion")
    if integrity.get("retraining") is not False:
        raise SystemExit("archived run unexpectedly indicates retraining")
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


def _quality_differences(actual: dict, archived: dict) -> list[dict]:
    differences: list[dict] = []
    for key in (
        "retained_probability_p50",
        "retained_qrs_band_fraction",
        "retention_fraction",
        "stage1_candidate_count",
        "retained_count",
        "selected_polarity",
        "polarity_confidence",
    ):
        expected = archived[key]
        got = actual[key]
        if isinstance(expected, str):
            same = got == expected
        elif isinstance(expected, int):
            same = int(got) == int(expected)
        else:
            same = _same_number(got, expected)
        if same:
            continue
        item = {"field": key, "actual": got, "archived": expected}
        if not isinstance(expected, str):
            try:
                item["absolute_difference"] = abs(float(got) - float(expected))
            except (TypeError, ValueError):
                pass
        differences.append(item)
    return differences


def _assert_selected_metrics(record: str, actual, archived: dict) -> None:
    for key in (
        "reference_count",
        "detected_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "sensitivity",
        "positive_predictive_value",
        "f1",
    ):
        got = getattr(actual, key)
        expected = archived[key]
        same = int(got) == int(expected) if isinstance(expected, int) else _same_number(got, expected)
        if not same:
            raise SystemExit(
                f"{record}: selected-lead {key}={got} does not reproduce archived {expected}"
            )


def _verify_local_source_identity(root: Path, protocol: dict, first_run: dict) -> dict[str, str]:
    """Require byte-identical local inputs before reproducing archived metrics."""
    expected = first_run.get("dataset_input_hashes")
    if not isinstance(expected, dict) or not expected:
        raise SystemExit("archived first-run report lacks dataset_input_hashes")

    actual = _prospective.verify_dataset(root, protocol)
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    mismatched = sorted(
        key for key in expected_keys & actual_keys if str(actual[key]) != str(expected[key])
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected[:8]))
        if mismatched:
            preview = ",".join(
                f"{key}:{expected[key]}!={actual[key]}" for key in mismatched[:4]
            )
            details.append("mismatched=" + preview)
        raise SystemExit(
            "local Zymed LTSTDB source bytes do not match the immutable first-run hashes; "
            + "; ".join(details)
        )
    return actual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--ltstdb-dir", type=Path)
    source.add_argument(
        "--pn-dir",
        help="PhysioNet directory for bounded remote streaming, e.g. ltstdb/1.0.0",
    )
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

    verified_source_hashes = None
    if args.ltstdb_dir is not None:
        verified_source_hashes = _verify_local_source_identity(
            args.ltstdb_dir, protocol, first
        )
        print(
            f"Verified {len(verified_source_hashes)} local source hashes against "
            "the immutable prospective first run.",
            flush=True,
        )

    model = CandidateSuppressor.load(args.model)
    _prospective._sv.verify_model(model, protocol)
    derivation, width_gate, v2_gate = _prospective._sv._load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("polarity derivation schema mismatch")
    if not _same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("v2 polarity gate mismatch")
    if not _same_number(width_gate, detector["expected_width_override_confidence"]):
        raise SystemExit("width gate mismatch")

    dataset = protocol["dataset"]
    selector = protocol["lead_selector"]
    window = dataset["window"]
    start = int(window["start_sample"])
    stop = int(window["stop_sample_exclusive"])
    fs = float(dataset["expected_sampling_frequency_hz"])
    duration_minutes = float(window["duration_seconds"]) / 60.0
    tolerance_ms = float(detector["tolerance_ms"])
    beat_symbols = frozenset(detector["beat_symbols"])
    archived_by = {str(r["record"]): r for r in first["record_results"]}

    channel0_results: list[RecordValidation] = []
    channel1_results: list[RecordValidation] = []
    selected_results: list[RecordValidation] = []
    oracle_results: list[RecordValidation] = []
    rows = []
    input_drift_records = []

    for i, record in enumerate(records, start=1):
        record_name = str(args.ltstdb_dir / record) if args.ltstdb_dir else record
        remote_kwargs = {} if args.ltstdb_dir else {"pn_dir": str(args.pn_dir)}
        record_reader = wfdb.rdrecord
        record_args = {
            "sampfrom": start,
            "sampto": stop,
            "channels": [0, 1],
            "physical": True,
            **remote_kwargs,
        }
        rec = (
            record_reader(record_name, **record_args)
            if args.ltstdb_dir
            else _call_with_retry(record_reader, record_name, **record_args)
        )
        if not _same_number(rec.fs, fs):
            raise SystemExit(f"{record}: unexpected sampling frequency {rec.fs}")
        signals = np.asarray(rec.p_signal, dtype=float)
        if signals.shape != (stop - start, 2) or not np.isfinite(signals).all():
            raise SystemExit(f"{record}: invalid frozen signal geometry {signals.shape}")

        detections = []
        qualities = []
        rates = []
        for ch in (0, 1):
            retained, _, quality = compute_lead_quality(
                signals[:, ch],
                fs,
                model,
                scale_method=str(detector["stage1_scale_method"]),
                v2_gate_confidence=v2_gate,
                width_override_confidence=width_gate,
                threshold=float(model.metadata.threshold),
            )
            retained = np.asarray(retained, dtype=int)
            detections.append(retained)
            qualities.append(quality)
            rates.append(float(len(retained) / duration_minutes))

        selected = choose_two_lead_channel_v3(
            rates[0],
            rates[1],
            primary_rate_ceiling_bpm=float(selector["primary_retained_rate_ceiling_bpm"]),
            alternate_rate_floor_bpm=float(selector["alternate_retained_rate_floor_bpm"]),
            minimum_rate_ratio=float(selector["minimum_alternate_to_primary_rate_ratio"]),
        )

        annotation_args = {
            "sampfrom": start,
            "sampto": stop - 1,
            **remote_kwargs,
        }
        annotation = (
            wfdb.rdann(
                record_name,
                str(detector["annotation_extension"]),
                **annotation_args,
            )
            if args.ltstdb_dir
            else _call_with_retry(
                wfdb.rdann,
                record_name,
                str(detector["annotation_extension"]),
                **annotation_args,
            )
        )
        reference, _ = _prospective._sv._reference_from_annotation(
            annotation, beat_symbols, stop - start
        )
        metrics = [
            match_peaks(detections[ch], reference, fs, tolerance_ms=tolerance_ms)
            for ch in (0, 1)
        ]

        archived = archived_by[record]
        if int(archived["selected_channel"]) != int(selected):
            raise SystemExit(
                f"{record}: recomputed selector chose {selected}, "
                f"archived chose {archived['selected_channel']}"
            )
        record_drift = {
            "record": record,
            "archived_selected_channel": int(archived["selected_channel"]),
            "channels": [],
        }
        for ch in (0, 1):
            channel_drift = []
            archived_rate = archived[f"channel{ch}_retained_rate_bpm"]
            if not _same_number(rates[ch], archived_rate):
                channel_drift.append(
                    {
                        "field": "retained_rate_bpm",
                        "actual": rates[ch],
                        "archived": archived_rate,
                        "absolute_difference": abs(float(rates[ch]) - float(archived_rate)),
                    }
                )
            channel_drift.extend(
                _quality_differences(
                    qualities[ch].to_dict(),
                    archived[f"channel{ch}_quality"],
                )
            )
            if channel_drift:
                record_drift["channels"].append(
                    {
                        "channel": ch,
                        "was_archived_selected_channel": ch == int(archived["selected_channel"]),
                        "differences": channel_drift,
                    }
                )
        if record_drift["channels"]:
            input_drift_records.append(record_drift)

        # The prospective scientific result is the selected channel and its scored
        # metrics. These remain strict invariants. Label-free feature/probability
        # values from the exposed alternate path are audited, not silently treated
        # as bitwise-portable across separate executions.
        _assert_selected_metrics(record, metrics[selected], archived)

        oracle = 1 if metrics[1].f1 > metrics[0].f1 else 0
        channel0_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[0]))
        channel1_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[1]))
        selected_results.append(
            RecordValidation(record=record, fs_hz=fs, metrics=metrics[selected])
        )
        oracle_results.append(RecordValidation(record=record, fs_hz=fs, metrics=metrics[oracle]))

        rows.append(
            {
                "record": record,
                "selected_channel": int(selected),
                "oracle_channel_posthoc": int(oracle),
                "channel0_retained_rate_bpm": rates[0],
                "channel1_retained_rate_bpm": rates[1],
                "channel0_quality": qualities[0].to_dict(),
                "channel1_quality": qualities[1].to_dict(),
                "channel0": metrics[0].to_dict(),
                "channel1": metrics[1].to_dict(),
                "selected": metrics[selected].to_dict(),
                "archived_label_free_input_drift": record_drift["channels"],
                "selected_minus_channel0_f1": float(
                    metrics[selected].f1 - metrics[0].f1
                ),
                "oracle_minus_selected_f1": float(
                    metrics[oracle].f1 - metrics[selected].f1
                ),
            }
        )
        drift_count = sum(len(ch["differences"]) for ch in record_drift["channels"])
        print(
            f"[{i:02d}/18] {record}: selected result reproduced; "
            f"label-free input drift fields={drift_count}",
            flush=True,
        )

    switched = [r for r in rows if r["selected_channel"] == 1]
    improved = [r for r in switched if r["selected_minus_channel0_f1"] > 1e-12]
    regressed = [r for r in switched if r["selected_minus_channel0_f1"] < -1e-12]

    report = {
        "schema": "electrotrace.ltstdb_zymed_selector_v3_posthoc_audit/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "posthoc_exposed_dataset_comparator_analysis",
        "git_head": git_head(),
        "guardrail": (
            "Zymed LTSTDB is exposed after the immutable first prospective selector-v3 "
            "run. This audit cannot alter or replace that result."
        ),
        "inputs": {
            "data_source": (
                {"mode": "local", "path": str(args.ltstdb_dir)}
                if args.ltstdb_dir
                else {"mode": "physionet_remote_bounded", "pn_dir": str(args.pn_dir)}
            ),
            "first_run_report": str(args.first_run_report),
            "first_run_sha256": _prospective._sv.sha256_file(args.first_run_report),
            "source_identity": {
                "verified_against_first_run_hashes": verified_source_hashes is not None,
                "verified_file_count": (
                    len(verified_source_hashes) if verified_source_hashes is not None else 0
                ),
            },
            "model": str(args.model),
            "model_sha256": _prospective._sv.sha256_file(args.model),
            "polarity_derivation": str(args.polarity_threshold_report),
            "polarity_derivation_sha256": _prospective._sv.sha256_file(
                args.polarity_threshold_report
            ),
        },
        "reproduction": {
            "all_18_archived_selected_channel_decisions_reproduced": True,
            "all_18_archived_selected_lead_results_reproduced": True,
            "label_free_input_bitwise_equivalence": len(input_drift_records) == 0,
            "label_free_input_drift_record_count": len(input_drift_records),
            "label_free_input_drift_records": input_drift_records,
            "interpretation": (
                "Selected channel decisions and selected-lead outcome metrics are strict "
                "reproduction invariants. Any cross-run drift in label-free channel inputs "
                "is retained explicitly because this is exposed-data post-hoc analysis, "
                "not a replacement prospective result."
            ),
            "channel2_loaded": False,
        },
        "summaries": {
            "channel0_fixed_posthoc": _summary(channel0_results),
            "channel1_fixed_posthoc": _summary(channel1_results),
            "prospective_selector_v3_reproduced": _summary(selected_results),
            "oracle_best_channel_posthoc": _summary(oracle_results),
        },
        "switch_audit": {
            "selected_channel1_records": len(switched),
            "switches_improved_vs_channel0": len(improved),
            "switches_regressed_vs_channel0": len(regressed),
            "switches_tied_vs_channel0": len(switched) - len(improved) - len(regressed),
        },
        "switched_records": switched,
        "records": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nComparator summaries:")
    print(json.dumps(report["summaries"], indent=2, sort_keys=True))
    print("Switch audit:", json.dumps(report["switch_audit"], sort_keys=True))
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
