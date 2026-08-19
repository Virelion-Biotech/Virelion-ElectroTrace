#!/usr/bin/env python3
"""Run rigorous ElectroTrace R-peak validation against MIT-BIH.

The dataset is downloaded to a local cache and is never committed to the repository.
The report keeps the exact record list, detector configuration, immutable
manifest hash, pooled metrics, macro record-level metrics, bootstrap intervals,
and per-record results together.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import wfdb

from electrotrace import __version__
from electrotrace.provenance import DatasetManifest
from electrotrace.research_validation import build_validation_report, write_validation_report
from electrotrace.validation import validate_record
from electrotrace.validation_detectors import detect_r_peaks, select_signal_polarity


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
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(cache), keep_subdirs=False)
    records = list(wfdb.get_record_list("mitdb"))
    record_paths = [cache / record for record in records]

    if args.polarity == "adaptive":
        detector = lambda signal, fs_hz: detect_r_peaks(signal, fs_hz, polarity="adaptive")
    else:
        detector = lambda signal, fs_hz: detect_r_peaks(signal, fs_hz, polarity=args.polarity)

    results = [
        validate_record(
            path,
            detector,
            channel=args.channel,
            annotation_extension="atr",
            tolerance_ms=args.tolerance_ms,
        )
        for path in record_paths
    ]

    manifest = DatasetManifest(
        dataset_id="MIT-BIH Arrhythmia Database",
        dataset_version=args.dataset_version,
        source="PhysioNet/WFDB",
        records=tuple(records),
        annotation_policy="default ElectroTrace beat-symbol whitelist from electrotrace.validation",
        detector_config={
            "detector": "electrotrace.validation_detectors:detect_r_peaks",
            "polarity": args.polarity,
            "channel": args.channel,
            "matching_tolerance_ms": args.tolerance_ms,
        },
        split_manifest={"validation": tuple(records)},
        software_version=__version__,
        software_commit=os.environ.get("GITHUB_SHA", "unknown"),
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
        beat_symbols=None,
        tolerance_ms=args.tolerance_ms,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["record_count_requested"] = len(records)

    output = write_validation_report(report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report written to {output}")
    print(f"Manifest SHA-256: {report['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
