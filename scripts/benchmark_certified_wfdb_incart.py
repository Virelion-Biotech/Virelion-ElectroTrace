#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    DetectionMetrics,
    RecordValidation,
    summarize_records,
    validate_record,
)

REFERENCE_BEAT_SYMBOLS = sorted(DEFAULT_BEAT_SYMBOLS)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_reference_samples(record_base: Path) -> np.ndarray:
    ann = wfdb.rdann(str(record_base), "atr")
    refs = np.asarray(
        [
            int(sample)
            for sample, symbol in zip(ann.sample, ann.symbol)
            if symbol in REFERENCE_BEAT_SYMBOLS
        ],
        dtype=int,
    )
    return refs


def run_detector(executable: str, record_base: Path) -> np.ndarray:
    completed = subprocess.run(
        [executable, "-r", str(record_base), "-s", "0"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} failed for {record_base.name}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    qrs_file = record_base.with_suffix(".qrs")
    if not qrs_file.exists():
        raise RuntimeError(f"{executable} completed but did not create {qrs_file}")
    ann = wfdb.rdann(str(record_base), "qrs")
    detected = np.asarray(
        [int(s) for s, sym in zip(ann.sample, ann.symbol) if sym != "|"],
        dtype=int,
    )
    if detected.size and np.any(np.diff(detected) <= 0):
        raise RuntimeError(f"{executable}: non-increasing detections")
    return detected


def list_incart_records(data_dir: Path) -> list[str]:
    local = sorted(p.stem for p in data_dir.glob("*.hea"))
    if local:
        return local
    return [Path(r).name for r in wfdb.get_record_list("incartdb")]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--detectors", default="gqrs,sqrs")
    args = parser.parse_args()

    detectors = [x.strip() for x in args.detectors.split(",") if x.strip()]
    for executable in detectors:
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not installed.")

    if not args.data_dir.exists():
        raise SystemExit(f"Missing INCART directory: {args.data_dir}")

    record_names = list_incart_records(args.data_dir)
    if not record_names:
        raise SystemExit(f"No INCART records found under {args.data_dir}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    global_skipped = []

    for detector_name in detectors:
        detector_dir = args.work_dir / detector_name
        detector_dir.mkdir(parents=True, exist_ok=True)
        detector_results = []
        skipped = []
        print(f"\n===== {detector_name} =====")

        for idx, record_name in enumerate(record_names, start=1):
            source_base = args.data_dir / record_name
            target_base = detector_dir / record_name
            try:
                for suffix in (".hea", ".dat", ".atr"):
                    source = source_base.with_suffix(suffix)
                    target = target_base.with_suffix(suffix)
                    if not source.exists():
                        raise FileNotFoundError(source)
                    shutil.copy2(source, target)

                refs = load_reference_samples(target_base)
                if refs.size and np.any(refs < 0):
                    reason = "negative annotation sample index"
                    skipped.append({"record": record_name, "reason": reason})
                    print(f"[{idx}/{len(record_names)}] {record_name}: SKIPPED: {reason}")
                    continue

                detected = run_detector(detector_name, target_base)

                def detector_fn(signal, fs_hz, _detected=detected):
                    return _detected

                result = validate_record(
                    str(target_base),
                    detector_fn,
                    channel=0,
                    annotation_extension="atr",
                    beat_symbols=REFERENCE_BEAT_SYMBOLS,
                    tolerance_ms=args.tolerance_ms,
                )
                payload = result.to_dict()
                payload["detector"] = detector_name
                detector_results.append(payload)
                print(
                    f"[{idx}/{len(record_names)}] {record_name}: "
                    f"Sens={payload['sensitivity']:.6f} "
                    f"PPV={payload['positive_predictive_value']:.6f} "
                    f"F1={payload['f1']:.6f}"
                )
            except Exception as exc:
                skipped.append({"record": record_name, "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[{idx}/{len(record_names)}] {record_name}: SKIPPED: {exc}")

        validation_results = []
        for p in detector_results:
            metrics = DetectionMetrics(
                reference_count=int(p["reference_count"]),
                detected_count=int(p["detected_count"]),
                true_positive=int(p["true_positive"]),
                false_positive=int(p["false_positive"]),
                false_negative=int(p["false_negative"]),
                sensitivity=float(p["sensitivity"]),
                positive_predictive_value=float(p["positive_predictive_value"]),
                f1=float(p["f1"]),
                mean_timing_error_ms=p.get("mean_timing_error_ms"),
                median_timing_error_ms=p.get("median_timing_error_ms"),
                timing_sd_ms=p.get("timing_sd_ms"),
                median_absolute_timing_error_ms=p.get("median_absolute_timing_error_ms"),
                mean_absolute_timing_error_ms=p.get("mean_absolute_timing_error_ms"),
                p95_absolute_timing_error_ms=p.get("p95_absolute_timing_error_ms"),
                max_absolute_timing_error_ms=p.get("max_absolute_timing_error_ms"),
            )
            validation_results.append(
                RecordValidation(record=p["record"], fs_hz=float(p["fs_hz"]), metrics=metrics)
            )

        all_results[detector_name] = {
            "records": detector_results,
            "skipped_records": skipped,
            "summary": summarize_records(validation_results)
            if validation_results
            else {"records": 0, "note": "no usable records"},
        }
        global_skipped.extend({**x, "detector": detector_name} for x in skipped)

    result = {
        "schema": "electrotrace.certified_wfdb_incart_baselines/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Virelion-Biotech/Virelion-ElectroTrace",
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wfdb_python_version": getattr(wfdb, "__version__", "unknown"),
        "protocol": {
            "dataset": "PhysioNet INCART",
            "database": "incartdb",
            "expected_records": 75,
            "records_discovered": len(record_names),
            "record_names": record_names,
            "channel": 0,
            "annotation_extension": "atr",
            "beat_symbols": REFERENCE_BEAT_SYMBOLS,
            "tolerance_ms": args.tolerance_ms,
            "skip_negative_annotation_indices": True,
            "purpose": "External domain-shift baseline for comparison with ElectroTrace INCART two-stage external study",
            "comparable_electrotrace_report": "validation_reports/incart_two_stage_external_full_2026-09-09.json",
            "certified_tools": {name: shutil.which(name) for name in detectors},
            "tool_versions": {},
        },
        "results": all_results,
        "skipped_records_flat": global_skipped,
    }

    for executable in detectors:
        try:
            result["protocol"]["tool_versions"][executable] = subprocess.run(
                [executable, "-h"], text=True, capture_output=True
            ).stderr.strip()[:1000]
        except Exception:
            result["protocol"]["tool_versions"][executable] = "unknown"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n===== FINAL =====")
    for detector_name, payload in all_results.items():
        print(detector_name)
        print(json.dumps(payload["summary"], indent=2))
        print("skipped:", len(payload["skipped_records"]))
    print("\nWritten:", args.output)

    if all(payload.get("summary", {}).get("records", 0) == 0 for payload in all_results.values()):
        raise SystemExit("No usable INCART records for any detector. Do not interpret this run.")


if __name__ == "__main__":
    main()
