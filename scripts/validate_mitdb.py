#!/usr/bin/env python3
"""Run rigorous ElectroTrace R-peak validation against MIT-BIH.

The dataset is downloaded to a local cache and is never committed to the repository.
A definitive run is fail-closed: every requested record must validate successfully.
Diagnostic reports are still written before an incomplete run exits non-zero.
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
from electrotrace.research_validation import build_validation_report, summarize_records_rigorous, write_validation_report
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, validate_record
from electrotrace.validation_detectors import detect_r_peaks, select_signal_polarity

PRIMARY_TOLERANCE_MS = 75.0
SENSITIVITY_TOLERANCES_MS = (50.0, 75.0, 100.0, 150.0)


def _resolve_software_commit(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("GITHUB_SHA"), os.environ.get("ELECTROTRACE_GIT_COMMIT")]
    for value in candidates:
        if value and value.strip():
            return value.strip()
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5)
        value = result.stdout.strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("Unable to resolve the software commit. Run inside Git or set --software-commit / ELECTROTRACE_GIT_COMMIT / GITHUB_SHA.")


def _run(records, record_paths, detector, channel, tolerance_ms):
    results = []
    errors: dict[str, dict[str, str]] = {}
    for record, path in zip(records, record_paths):
        try:
            results.append(validate_record(path, detector, channel=channel, annotation_extension="atr", beat_symbols=sorted(DEFAULT_BEAT_SYMBOLS), tolerance_ms=tolerance_ms))
        except Exception as exc:
            errors[record] = {"type": type(exc).__name__, "message": str(exc)}
    return results, errors


def _polarity_audit(records, record_paths, channel):
    audit: dict[str, dict] = {}
    for record, path in zip(records, record_paths):
        try:
            wf_record = wfdb.rdrecord(str(path), channels=[int(channel)], physical=False)
            signal = wf_record.p_signal[:, 0] if wf_record.p_signal is not None else wf_record.d_signal[:, 0]
            decision = select_signal_polarity(signal, float(wf_record.fs))
            audit[record] = {
                "polarity": decision.polarity,
                "confidence": decision.confidence,
                "positive_score": decision.positive_score,
                "negative_score": decision.negative_score,
                "positive_candidates": decision.positive_candidates,
                "negative_candidates": decision.negative_candidates,
            }
        except Exception as exc:
            audit[record] = {"error": f"{type(exc).__name__}: {exc}"}
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/physionet/mitdb")
    parser.add_argument("--output", default="validation_reports/mitdb_rpeak_validation.json")
    parser.add_argument("--tolerance-ms", type=float, default=PRIMARY_TOLERANCE_MS)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--polarity", choices=("positive", "negative", "adaptive"), default="adaptive")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--software-commit", default=None)
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.tolerance_ms != PRIMARY_TOLERANCE_MS:
        parser.error(f"Definitive runs require the locked primary tolerance of {PRIMARY_TOLERANCE_MS:g} ms")
    if args.channel != 0:
        parser.error("Definitive runs require the locked primary channel index 0")
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

    results, record_errors = _run(records, record_paths, detector, args.channel, args.tolerance_ms)

    manifest = DatasetManifest(
        dataset_id="MIT-BIH Arrhythmia Database",
        dataset_version=args.dataset_version,
        source="PhysioNet/WFDB",
        records=tuple(records),
        annotation_policy="ElectroTrace primary validation beat-symbol whitelist: " + ",".join(sorted(DEFAULT_BEAT_SYMBOLS)),
        detector_config={
            "detector": "electrotrace.validation_detectors:detect_r_peaks",
            "polarity": args.polarity,
            "channel": args.channel,
            "minimum_peak_distance_ms": 250.0,
            "primary_matching_tolerance_ms": PRIMARY_TOLERANCE_MS,
        },
        split_manifest={"validation": tuple(records)},
        software_version=__version__,
        software_commit=software_commit,
    )
    report = build_validation_report(
        manifest,
        results,
        detector_name="electrotrace.validation_detectors:detect_r_peaks",
        detector_parameters={"polarity": args.polarity, "channel": args.channel, "minimum_peak_distance_ms": 250.0},
        annotation_extension="atr",
        beat_symbols=sorted(DEFAULT_BEAT_SYMBOLS),
        tolerance_ms=args.tolerance_ms,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    tolerance_sensitivity = {}
    for tolerance_ms in SENSITIVITY_TOLERANCES_MS:
        if tolerance_ms == PRIMARY_TOLERANCE_MS:
            tol_results = results
            tol_errors = record_errors
        else:
            tol_results, tol_errors = _run(records, record_paths, detector, args.channel, tolerance_ms)
        tolerance_sensitivity[str(tolerance_ms)] = {
            "status": "complete" if not tol_errors and len(tol_results) == len(records) else "incomplete",
            "summary": summarize_records_rigorous(tol_results, n_bootstrap=args.bootstrap, seed=args.seed) if tol_results else None,
            "record_failures": tol_errors,
        }

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["record_count_requested"] = len(records)
    report["record_count_successful"] = len(results)
    report["record_count_failed"] = len(record_errors)
    report["record_errors"] = record_errors
    report["validation_status"] = "complete" if not record_errors and len(results) == len(records) else "incomplete"
    report["primary_tolerance_ms"] = PRIMARY_TOLERANCE_MS
    report["preplanned_tolerance_sensitivity_ms"] = list(SENSITIVITY_TOLERANCES_MS)
    report["tolerance_sensitivity"] = tolerance_sensitivity
    report["full_record_evaluation"] = True
    report["test_period_analysis"] = "not run; requires a separately frozen standardized test-period manifest/protocol"
    report["channel_policy"] = "locked primary channel index 0; no annotation-informed channel selection"
    report["polarity_policy"] = "locked adaptive polarity option; polarity selection is unsupervised and record-local"
    report["polarity_audit"] = _polarity_audit(records, record_paths, args.channel)
    report["streaming_claim"] = "none; retrospective full-record normalization is permitted"
    report["minimum_peak_distance_policy"] = "250 ms; true events below this separation are unresolved by design"
    report["reference_scope"] = "all annotations matching the frozen beat-symbol whitelist"

    output = write_validation_report(report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report written to {output}")
    print(f"Manifest SHA-256: {report['manifest_sha256']}")
    print(f"Software commit: {software_commit}")
    print(f"Validation status: {report['validation_status']}")
    if record_errors:
        print(f"Record failures: {len(record_errors)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
