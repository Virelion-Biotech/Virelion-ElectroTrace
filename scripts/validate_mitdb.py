#!/usr/bin/env python3
"""Run rigorous ElectroTrace R-peak validation against MIT-BIH.

The dataset is downloaded to a local cache and is never committed to the repository.
The report keeps the exact record list, detector configuration, immutable
manifest hash, pooled metrics, macro record-level metrics, bootstrap intervals,
per-record results, and any record-level failures together.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import wfdb

from electrotrace import __version__
from electrotrace.provenance import DatasetManifest
from electrotrace.research_validation import build_validation_report, write_validation_report
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, validate_record
from electrotrace.validation_detectors import detect_r_peaks


def _resolve_software_commit(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("GITHUB_SHA"),
        os.environ.get("ELECTROTRACE_GIT_COMMIT"),
    ]
    for value in candidates:
        if value and value.strip():
            return value.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError(
        "Unable to resolve the software commit. Run inside a Git checkout or "
        "set --software-commit / ELECTROTRACE_GIT_COMMIT / GITHUB_SHA."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/physionet/mitdb")
    parser.add_argument("--output", default="validation_reports/mitdb_rpeak_validation.json")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--polarity", choices=("positive", "negative", "adaptive"), default="adaptive")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--software-commit", default=None)
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.tolerance_ms <= 0:
        parser.error("--tolerance-ms must be positive")
    if args.channel < 0:
        parser.error("--channel must be non-negative")

    software_commit = _resolve_software_commit(args.software_commit)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(cache), keep_subdirs=False)
    records = list(wfdb.get_record_list("mitdb"))
    record_paths = [cache / record for record in records]

    def detector(signal, fs_hz):
        return detect_r_peaks(signal, fs_hz, polarity=args.polarity)

    results = []
    record_errors: dict[str, dict[str, str]] = {}
    for record, path in zip(records, record_paths):
        try:
            results.append(
                validate_record(
                    path,
                    detector,
                    channel=args.channel,
                    annotation_extension="atr",
                    beat_symbols=sorted(DEFAULT_BEAT_SYMBOLS),
                    tolerance_ms=args.tolerance_ms,
                )
            )
        except Exception as exc:
            record_errors[record] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    if not results:
        raise RuntimeError("Validation failed for every requested MIT-BIH record")

    manifest = DatasetManifest(
        dataset_id="MIT-BIH Arrhythmia Database",
        dataset_version=args.dataset_version,
        source="PhysioNet/WFDB",
        records=tuple(records),
        annotation_policy=f"ElectroTrace validation beat-symbol whitelist: {','.join(sorted(DEFAULT_BEAT_SYMBOLS))}",
        detector_config={
            "detector": "electrotrace.validation_detectors:detect_r_peaks",
            "polarity": args.polarity,
            "channel": args.channel,
            "matching_tolerance_ms": args.tolerance_ms,
        },
        split_manifest={"validation": tuple(records)},
        software_version=__version__,
        software_commit=software_commit,
    )
    report = build_validation_report(
        manifest,
        results,
        detector_name="electrotrace.validation_detectors:detect_r_peaks",
        detector_parameters={
            "polarity": args.polarity,
            "channel": args.channel,
        },
        annotation_extension="atr",
        beat_symbols=sorted(DEFAULT_BEAT_SYMBOLS),
        tolerance_ms=args.tolerance_ms,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["record_count_requested"] = len(records)
    report["record_count_successful"] = len(results)
    report["record_errors"] = record_errors

    output = write_validation_report(report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report written to {output}")
    print(f"Manifest SHA-256: {report['manifest_sha256']}")
    print(f"Software commit: {software_commit}")
    if record_errors:
        print(f"Record failures: {len(record_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
