#!/usr/bin/env python3
"""Evaluate a frozen two-stage model on the locked MIT-BIH held-out split.

Every prior MIT-BIH number for the v4/windowed_std model
(``incart_mitbih_model_windowed_std_2026-09-09.skops``) was produced with
``--polarity positive`` (see ``mitdb_windowed_std_polarityfix.json``), not the
adaptive polarity the 1.8.1 lock and ``docs/VALIDATION.md`` both specify. That
means no valid non-regression number exists yet for the current model
generation. This script does not retrain anything; it loads a model file as-is
and scores it against the 12 records held out at the 1.8.1 freeze
(seed=42, test_fraction=0.25), with adaptive polarity, so the result is
comparable to ``mitdb_two_stage_locked_1.8.1.json``.

Usage:
  python -u scripts/evaluate_frozen_model_mitdb.py \
      --mitdb-dir .cache/physionet/mitdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --scale-method windowed_std

Pass --threshold to score at a different operating point without touching the
model file's own stored threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    RecordValidation,
    match_peaks,
    summarize_records,
)
from electrotrace.validation_detectors import detect_r_peaks_two_stage
from electrotrace.wfdb_records import (
    POLICIES,
    RecordExcluded,
    load_annotated_record,
    summarize_audits,
)

LOCKED_HELDOUT_RECORDS = [
    "105", "118", "122", "201", "207", "209",
    "214", "219", "230", "231", "232", "234",
]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict:
    versions = {}
    for module_name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "missing"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mitdb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scale-method", default="windowed_std")
    parser.add_argument(
        "--width-override-confidence",
        type=float,
        required=True,
        help="Frozen polarity width-override cutoff produced from MIT-BIH development data. Required to prevent held-out retuning.",
    )
    parser.add_argument(
        "--polarity", default="adaptive", choices=["adaptive", "positive", "negative"]
    )
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the model's stored threshold. Defaults to the model's own metadata.threshold.",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Defaults to the locked 12-record held-out split.",
    )
    parser.add_argument("--annotation-policy", choices=POLICIES, default="error")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_reports/mitdb_frozen_model_evaluation.json"),
    )
    args = parser.parse_args()

    model = CandidateSuppressor.load(args.model)
    threshold = float(model.metadata.threshold if args.threshold is None else args.threshold)
    names = args.records or LOCKED_HELDOUT_RECORDS

    results = []
    skipped = []
    audits = []
    for index, name in enumerate(names, start=1):
        try:
            annotated = load_annotated_record(
                args.mitdb_dir / name,
                beat_symbols=DEFAULT_BEAT_SYMBOLS,
                tolerance_ms=args.tolerance_ms,
                policy=args.annotation_policy,
            )
        except RecordExcluded as exc:
            audits.append(exc.audit)
            skipped.append({"record": name, "reason": exc.reason})
            print(f"[{index}/{len(names)}] {name}: SKIPPED: {exc.reason}", flush=True)
            continue
        audits.append(annotated.audit)

        retained, _ = detect_r_peaks_two_stage(
            annotated.signal,
            annotated.fs_hz,
            model,
            polarity=args.polarity,
            recovery=args.recovery,
            scale_method=args.scale_method,
            width_override_confidence=args.width_override_confidence,
            threshold=threshold,
        )
        metrics = match_peaks(
            retained, annotated.reference, annotated.fs_hz, tolerance_ms=args.tolerance_ms
        )
        results.append(
            RecordValidation(record=name, fs_hz=annotated.fs_hz, metrics=metrics)
        )
        print(
            f"[{index}/{len(names)}] {name}: sens={metrics.sensitivity:.4f} "
            f"ppv={metrics.positive_predictive_value:.4f} f1={metrics.f1:.4f}",
            flush=True,
        )

    if not results:
        raise SystemExit("No usable records; nothing to report.")

    summary = summarize_records(results)
    known_207 = next((r for r in results if r.record == "207"), None)

    report = {
        "schema": "electrotrace.mitdb_frozen_model_evaluation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Non-regression check: evaluate an already-trained model on the locked MIT-BIH "
            "held-out split without retraining, so it is directly comparable to "
            "mitdb_two_stage_locked_1.8.1.json for the same model generation."
        ),
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "metadata": model.metadata.to_dict(),
        },
        "protocol": {
            "dataset": "MIT-BIH held-out split (1.8.1 lock, seed=42, test_fraction=0.25)",
            "records": names,
            "polarity": args.polarity,
            "recovery": args.recovery,
            "scale_method": args.scale_method,
            "width_override_confidence": args.width_override_confidence,
            "tolerance_ms": args.tolerance_ms,
            "operating_threshold": threshold,
            "threshold_overridden": args.threshold is not None,
            "annotation_policy": args.annotation_policy,
            "retraining": False,
        },
        "annotation_audit": {
            "summary": summarize_audits(audits),
            "records": [a.to_dict() for a in audits],
        },
        "summary": summary,
        "record_results": [r.to_dict() for r in results],
        "skipped_records": skipped,
        "record_207_note": (
            f"F1={known_207.metrics.f1:.4f}" if known_207 else "not evaluated"
        ) + (
            "; polarity threshold is frozen before scoring; September's --polarity positive experiments all showed 207 collapsing "
            "to roughly F1 0.26. Check whether adaptive polarity fixes it here."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nWritten:", args.output)
    print("Skipped:", len(skipped))
    return 0 if not skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
