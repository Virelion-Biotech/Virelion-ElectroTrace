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
    validate_record,
    summarize_records,
    RecordValidation,
    DetectionMetrics,
)


MITDB_TEST_RECORDS = [
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


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_detector(executable, record_base):
    command = [
        executable,
        "-r",
        str(record_base),
        "-s",
        "0",
    ]

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} failed for {record_base.name}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    qrs_file = record_base.with_suffix(".qrs")

    if not qrs_file.exists():
        raise RuntimeError(
            f"{executable} completed but did not create {qrs_file}"
        )

    ann = wfdb.rdann(
        str(record_base),
        "qrs",
    )

    detected = np.asarray(
        [
            int(sample)
            for sample, symbol in zip(
                ann.sample,
                ann.symbol,
            )
            if symbol != "|"
        ],
        dtype=int,
    )

    if detected.size and np.any(
        np.diff(detected) <= 0
    ):
        raise RuntimeError(
            f"{executable}: non-increasing detections"
        )
    return detected


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=75.0,
    )

    args = parser.parse_args()

    for executable in ("gqrs", "sqrs"):
        if shutil.which(executable) is None:
            raise RuntimeError(
                f"{executable} is not installed."
            )

    args.work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for detector_name in ("gqrs", "sqrs"):
        detector_dir = (
            args.work_dir / detector_name
        )

        detector_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        detector_results = []

        print(
            f"\n===== {detector_name} ====="
        )

        for record_name in MITDB_TEST_RECORDS:
            source_base = (
                args.data_dir / record_name
            )

            target_base = (
                detector_dir / record_name
            )

            for suffix in (
                ".hea",
                ".dat",
                ".atr",
            ):
                source = source_base.with_suffix(
                    suffix
                )
                target = target_base.with_suffix(
                    suffix
                )

                if not source.exists():
                    raise FileNotFoundError(
                        source
                    )

                shutil.copy2(
                    source,
                    target,
                )

            detected = run_detector(
                detector_name,
                target_base,
            )

            def detector_fn(signal, fs_hz):
                return detected

            result = validate_record(
                str(target_base),
                detector_fn,
                channel=0,
                annotation_extension="atr",
                beat_symbols=sorted(
                    DEFAULT_BEAT_SYMBOLS
                ),
                tolerance_ms=args.tolerance_ms,
            )

            payload = result.to_dict()
            payload["detector"] = detector_name

            detector_results.append(
                payload
            )

            print(
                f"{record_name}: "
                f"Sens={payload['sensitivity']:.6f} "
                f"PPV={payload['positive_predictive_value']:.6f} "
                f"F1={payload['f1']:.6f}"
            )

        validation_results = []

        for p in detector_results:
            metrics = DetectionMetrics(
                reference_count=int(
                    p["reference_count"]
                ),
                detected_count=int(
                    p["detected_count"]
                ),
                true_positive=int(
                    p["true_positive"]
                ),
                false_positive=int(
                    p["false_positive"]
                ),
                false_negative=int(
                    p["false_negative"]
                ),
                sensitivity=float(
                    p["sensitivity"]
                ),
                positive_predictive_value=float(
                    p["positive_predictive_value"]
                ),
                f1=float(p["f1"]),
                mean_timing_error_ms=p.get(
                    "mean_timing_error_ms"
                ),
                median_timing_error_ms=p.get(
                    "median_timing_error_ms"
                ),
                timing_sd_ms=p.get(
                    "timing_sd_ms"
                ),
                median_absolute_timing_error_ms=p.get(
                    "median_absolute_timing_error_ms"
                ),
                mean_absolute_timing_error_ms=p.get(
                    "mean_absolute_timing_error_ms"
                ),
                p95_absolute_timing_error_ms=p.get(
                    "p95_absolute_timing_error_ms"
                ),
                max_absolute_timing_error_ms=p.get(
                    "max_absolute_timing_error_ms"
                ),
            )

            validation_results.append(
                RecordValidation(
                    record=p["record"],
                    fs_hz=float(p["fs_hz"]),
                    metrics=metrics,
                )
            )

        all_results[detector_name] = {
            "records": detector_results,
            "summary": summarize_records(
                validation_results
            ),
        }

    result = {
        "schema": (
            "electrotrace.certified_wfdb_baselines/"
            "v1"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "repository": (
            "Virelion-Biotech/"
            "Virelion-ElectroTrace"
        ),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wfdb_python_version": (
            getattr(wfdb, "__version__", "unknown")
        ),
        "protocol": {
            "dataset": "MIT-BIH Arrhythmia Database",
            "database": "mitdb",
            "records": MITDB_TEST_RECORDS,
            "channel": 0,
            "annotation_extension": "atr",
            "beat_symbols": sorted(
                DEFAULT_BEAT_SYMBOLS
            ),
            "tolerance_ms": args.tolerance_ms,
            "certified_tools": {
                "gqrs": shutil.which("gqrs"),
                "sqrs": shutil.which("sqrs"),
            },
            "tool_versions": {},
        },
        "results": all_results,
    }

    for executable in ("gqrs", "sqrs"):
        try:
            result["protocol"]["tool_versions"][
                executable
            ] = subprocess.run(
                [executable, "-h"],
                text=True,
                capture_output=True,
            ).stderr.strip()[:1000]
        except Exception:
            result["protocol"]["tool_versions"][
                executable
            ] = "unknown"

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== FINAL =====")

    for detector_name, payload in all_results.items():
        print(
            detector_name,
            json.dumps(
                payload["summary"],
                indent=2,
            ),
        )

    print(
        "\nWritten:",
        args.output,
    )


if __name__ == "__main__":
    main()
