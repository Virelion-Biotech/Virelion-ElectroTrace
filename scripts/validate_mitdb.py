#!/usr/bin/env python3
"""Run the ElectroTrace R-peak detector against the MIT-BIH Arrhythmia Database.

The dataset is downloaded to a local cache and is never committed to the repository.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import wfdb

from electrotrace import __version__
from electrotrace.validation import summarize_records, validate_record
from electrotrace.validation_detectors import detect_r_peaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/physionet/mitdb")
    parser.add_argument("--output", default="validation_reports/mitdb_rpeak_validation.json")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(cache), keep_subdirs=False)
    records = wfdb.get_record_list("mitdb")
    record_paths = [cache / record for record in records]

    results = [
        validate_record(
            path,
            detect_r_peaks,
            channel=args.channel,
            annotation_extension="atr",
            tolerance_ms=args.tolerance_ms,
        )
        for path in record_paths
    ]

    report = {
        "schema": "electrotrace.external_validation/v1",
        "software_version": __version__,
        "dataset": "MIT-BIH Arrhythmia Database",
        "dataset_slug": "mitdb",
        "records": records,
        "channel": args.channel,
        "detector": "electrotrace.validation_detectors:detect_r_peaks",
        "annotation_extension": "atr",
        "tolerance_ms": args.tolerance_ms,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_results": [r.to_dict() for r in results],
        "summary": summarize_records(results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
