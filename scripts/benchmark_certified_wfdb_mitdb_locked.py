#!/usr/bin/env python3
"""Benchmark certified WFDB gqrs/sqrs on the locked MIT-BIH evaluation split.

This is a reference-binary comparison. It intentionally reuses the exact
12-record locked split and 75 ms matching tolerance from the historical
ElectroTrace 1.8.1 validation artifact. No detector parameters are tuned on
these records.
"""
from __future__ import annotations

import argparse
import json
import os
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
    RecordValidation,
    match_peaks,
    summarize_records,
)

LOCKED_TEST_RECORDS = [
    "105",
    "118",
    "122",
    "201",
    "207",
    "209",
    "214",
    "219",
    "230",
    "231",
    "232",
    "234",
]
REFERENCE_BEAT_SYMBOLS = frozenset(DEFAULT_BEAT_SYMBOLS)
DETECTORS = ("gqrs", "sqrs")
REQUIRED_SUFFIXES = (".hea", ".dat", ".atr")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def wfdb_toolkit_version() -> str:
    commands = (
        ["wfdb-config", "--version"],
        ["dpkg-query", "-W", "-f=${Version}", "wfdb"],
    )
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            continue
        value = (completed.stdout or completed.stderr).strip()
        if completed.returncode == 0 and value:
            return value
    return "unknown"


def validate_locked_inputs(data_dir: Path) -> None:
    missing: list[str] = []
    for record in LOCKED_TEST_RECORDS:
        for suffix in REQUIRED_SUFFIXES:
            path = data_dir / f"{record}{suffix}"
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(str(path))
    if missing:
        raise SystemExit(
            "Locked MIT-BIH input set is incomplete; missing/non-empty check failed: "
            + ", ".join(missing)
        )


def run_detector(executable: str, record_base: Path) -> np.ndarray:
    completed = subprocess.run(
        [executable, "-r", record_base.name, "-s", "0"],
        cwd=record_base.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} failed for {record_base.name}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    qrs_file = record_base.with_suffix(".qrs")
    if not qrs_file.exists():
        raise RuntimeError(f"{executable} completed but did not create {qrs_file}")
    annotation = wfdb.rdann(str(record_base), "qrs")
    detected = np.asarray(
        [int(sample) for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol != "|"],
        dtype=np.int64,
    )
    if detected.size and np.any(np.diff(detected) <= 0):
        raise RuntimeError(f"{record_base.name}: {executable} detections are not strictly increasing")
    return detected


def reference_beats(record_base: Path) -> np.ndarray:
    annotation = wfdb.rdann(str(record_base), "atr")
    reference = np.asarray(
        [
            int(sample)
            for sample, symbol in zip(annotation.sample, annotation.symbol)
            if symbol in REFERENCE_BEAT_SYMBOLS
        ],
        dtype=np.int64,
    )
    if reference.size and (np.any(reference < 0) or np.any(np.diff(reference) <= 0)):
        raise RuntimeError(f"{record_base.name}: invalid locked reference annotations")
    return reference


def evaluate_detector(
    detector_name: str,
    *,
    data_dir: Path,
    work_dir: Path,
    tolerance_ms: float,
) -> tuple[list[RecordValidation], str]:
    executable = shutil.which(detector_name)
    if executable is None:
        raise SystemExit(f"Required certified WFDB detector is unavailable: {detector_name}")

    detector_dir = work_dir / detector_name
    detector_dir.mkdir(parents=True, exist_ok=True)
    results: list[RecordValidation] = []

    for record in LOCKED_TEST_RECORDS:
        source_base = data_dir / record
        target_base = detector_dir / record
        for suffix in REQUIRED_SUFFIXES:
            shutil.copy2(source_base.with_suffix(suffix), target_base.with_suffix(suffix))

        header = wfdb.rdheader(str(target_base))
        fs_hz = float(header.fs)
        reference = reference_beats(target_base)
        detected = run_detector(executable, target_base)
        metrics = match_peaks(detected, reference, fs_hz, tolerance_ms=tolerance_ms)
        results.append(RecordValidation(record=record, fs_hz=fs_hz, metrics=metrics))

    return results, executable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    args = parser.parse_args()

    if not np.isfinite(args.tolerance_ms) or args.tolerance_ms <= 0:
        raise SystemExit("--tolerance-ms must be positive and finite")

    validate_locked_inputs(args.data_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    detector_payload: dict[str, dict] = {}
    executable_paths: dict[str, str] = {}

    for detector_name in DETECTORS:
        results, executable = evaluate_detector(
            detector_name,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            tolerance_ms=float(args.tolerance_ms),
        )
        if [result.record for result in results] != LOCKED_TEST_RECORDS:
            raise SystemExit(f"{detector_name}: locked record order drifted")
        executable_paths[detector_name] = executable
        detector_payload[detector_name] = {
            "summary": summarize_records(results),
            "records": [result.to_dict() for result in results],
        }

    report = {
        "schema": "electrotrace.certified_wfdb_mitdb_locked/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "Virelion-Biotech/Virelion-ElectroTrace",
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wfdb_python_version": getattr(wfdb, "__version__", "unknown"),
        "wfdb_toolkit_version": wfdb_toolkit_version(),
        "wfdb_toolkit_source_release": os.environ.get("WFDB_TOOLKIT_SOURCE_RELEASE", "unknown"),
        "protocol": {
            "dataset": "MIT-BIH Arrhythmia Database",
            "physionet_database": "mitdb",
            "split": "historical locked 12-record evaluation split",
            "records": LOCKED_TEST_RECORDS,
            "records_expected": 12,
            "channel": 0,
            "annotation_extension": "atr",
            "beat_symbols": sorted(REFERENCE_BEAT_SYMBOLS),
            "tolerance_ms": float(args.tolerance_ms),
            "detector_configuration": {
                "gqrs": {"signal": 0, "threshold": "WFDB default 1.00"},
                "sqrs": {"signal": 0, "threshold": "WFDB default 500"},
            },
            "parameter_tuning_on_locked_records": False,
            "comparison_scope": (
                "Reference-binary comparison on the same locked record split; "
                "not prospective evidence for ElectroTrace mechanisms."
            ),
        },
        "executables": executable_paths,
        "results": detector_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name in DETECTORS:
        summary = detector_payload[name]["summary"]
        print(
            f"{name}: records={summary['records']} "
            f"Sens={summary['sensitivity']:.6f} "
            f"PPV={summary['positive_predictive_value']:.6f} "
            f"F1={summary['f1']:.6f}"
        )
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
