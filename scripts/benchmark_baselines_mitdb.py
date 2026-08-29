#!/usr/bin/env python3
"""Locked baseline comparison on the same MIT-BIH held-out split as two-stage.

Uses the identical seed=42 / test_fraction=0.25 record list and the same
validate_record scoring (75 ms tolerance, DEFAULT_BEAT_SYMBOLS).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace import __version__
from electrotrace.baseline_detectors import BASELINE_DETECTORS
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, DetectionMetrics, RecordValidation, summarize_records, validate_record
from electrotrace.validation_detectors import detect_r_peaks


def _provenance() -> dict:
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = "unknown"
    pkgs = {}
    for name in ("numpy", "scipy", "wfdb"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = "missing"
    return {
        "git_head": git_head,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": pkgs,
        "electrotrace_version": __version__,
    }


def _held_out_records(seed: int, test_fraction: float) -> list[str]:
    records = list(wfdb.get_record_list("mitdb"))
    rng = np.random.default_rng(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    return sorted(shuffled[:n_test])


def _eval_detector(name: str, detector, records: list[str], data_dir: Path) -> dict:
    payloads = []
    for record in records:
        base = str(data_dir / record)

        def det(signal, fs_hz, _d=detector):
            return _d(signal, fs_hz)

        result = validate_record(
            base, det, channel=0, annotation_extension="atr",
            beat_symbols=sorted(DEFAULT_BEAT_SYMBOLS), tolerance_ms=75,
        )
        payloads.append(result.to_dict())

    results = []
    for p in payloads:
        metrics = DetectionMetrics(
            reference_count=int(p["reference_count"]), detected_count=int(p["detected_count"]),
            true_positive=int(p["true_positive"]), false_positive=int(p["false_positive"]),
            false_negative=int(p["false_negative"]), sensitivity=float(p["sensitivity"]),
            positive_predictive_value=float(p["positive_predictive_value"]), f1=float(p["f1"]),
            mean_timing_error_ms=p.get("mean_timing_error_ms"), median_timing_error_ms=p.get("median_timing_error_ms"),
            timing_sd_ms=p.get("timing_sd_ms"),
            median_absolute_timing_error_ms=p.get("median_absolute_timing_error_ms"),
            mean_absolute_timing_error_ms=p.get("mean_absolute_timing_error_ms"),
            p95_absolute_timing_error_ms=p.get("p95_absolute_timing_error_ms"),
            max_absolute_timing_error_ms=p.get("max_absolute_timing_error_ms"),
        )
        results.append(RecordValidation(record=p["record"], fs_hz=float(p["fs_hz"]), metrics=metrics))

    summary = summarize_records(results)
    return {
        "detector": name,
        "summary": summary,
        "record_results": [
            {
                "record": p["record"],
                "sensitivity": p["sensitivity"],
                "positive_predictive_value": p["positive_predictive_value"],
                "f1": p["f1"],
                "true_positive": p["true_positive"],
                "false_positive": p["false_positive"],
                "false_negative": p["false_negative"],
            }
            for p in payloads
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="validation_reports/mitdb_baseline_comparison_locked.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_records = _held_out_records(args.seed, args.test_fraction)

    detectors = {
        "electrotrace_stage1_adaptive": lambda s, fs: detect_r_peaks(s, fs, polarity="adaptive"),
        **BASELINE_DETECTORS,
    }

    comparisons = []
    for name, fn in detectors.items():
        print(f"Evaluating {name} on {len(test_records)} records...", flush=True)
        comparisons.append(_eval_detector(name, fn, test_records, data_dir))
        s = comparisons[-1]["summary"]
        print(
            f"  {name}: sens={s['sensitivity']:.4f} ppv={s['positive_predictive_value']:.4f} f1={s['f1']:.4f}",
            flush=True,
        )

    report = {
        "schema": "electrotrace.baseline_comparison/v1",
        "software_version": __version__,
        "provenance": _provenance(),
        "protocol": {
            "primary_endpoint": "held_out_test_records_only",
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "tolerance_ms": 75,
            "test_records": test_records,
            "note": "Same split as two-stage locked protocol. Classical detectors are research-grade reimplementations, not certified reference binaries.",
        },
        "comparisons": comparisons,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
