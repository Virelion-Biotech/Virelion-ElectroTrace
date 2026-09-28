#!/usr/bin/env python3
"""Evaluate a frozen two-stage model on the locked MIT-BIH split.

This is a non-regression/audit runner. For adaptive polarity, historical
inspection of MIT-BIH record 207 and pooled MIT-BIH behavior influenced the
mechanism itself, so a clean threshold re-derivation does not retroactively
make adaptive-polarity results prospective held-out evidence. Reports emitted
by this script label that limitation explicitly.

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
import math
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
from electrotrace.validation_detectors import (
    DEFAULT_V2_GATE_CONFIDENCE,
    detect_r_peaks_two_stage,
)
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


DERIVATION_SCHEMA = "electrotrace.mitdb_polarity_thresholds_extended_derivation/v2"
EXPECTED_DEVELOPMENT_POOL_SIZE = 36
EXPECTED_DERIVATION_CORE_HASHES = EXPECTED_DEVELOPMENT_POOL_SIZE * 3


def _confidence_arg(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence gate must be a number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("confidence gate must be finite and between 0 and 1")
    return value


def _load_polarity_threshold_report(path: Path) -> tuple[dict, float, float]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read polarity threshold report {path}: {exc}") from exc

    if report.get("schema") != DERIVATION_SCHEMA:
        raise SystemExit(
            f"unsupported polarity threshold report schema: {report.get('schema')!r}; "
            f"expected {DERIVATION_SCHEMA!r}"
        )
    if report.get("freeze_eligible") is not True:
        raise SystemExit("polarity threshold report is not freeze_eligible")
    if report.get("selection_status") != "development_only_full_pool":
        raise SystemExit("polarity threshold report was not derived from the full development pool")

    report_head = str(report.get("git_head") or "")
    current_head = git_head()
    if not report_head or report_head == "unknown" or current_head == "unknown":
        raise SystemExit("polarity threshold report/current checkout lacks a verifiable git commit")
    if report_head != current_head:
        raise SystemExit(
            f"polarity threshold report git_head {report_head} does not match current checkout {current_head}"
        )

    input_hashes = report.get("input_hashes")
    if not isinstance(input_hashes, dict) or len(input_hashes) != EXPECTED_DERIVATION_CORE_HASHES:
        raise SystemExit(
            "polarity threshold report does not contain the complete 36-record core input hash set"
        )

    protocol = report.get("protocol") or {}
    if protocol.get("development_pool_size") != EXPECTED_DEVELOPMENT_POOL_SIZE:
        raise SystemExit(
            f"polarity threshold report development_pool_size must be {EXPECTED_DEVELOPMENT_POOL_SIZE}"
        )
    if protocol.get("locked_heldout_labels_used") is not False:
        raise SystemExit("polarity threshold report does not prove locked held-out exclusion")
    if protocol.get("incart_used") is not False:
        raise SystemExit("polarity threshold report used INCART and cannot authorize locked evaluation")
    if protocol.get("diagnostic_subset") is not False:
        raise SystemExit("diagnostic-subset polarity report cannot authorize locked evaluation")

    recommended = report.get("recommended_thresholds") or {}
    try:
        v2_gate = float(recommended["v2_gate_confidence"])
        width_gate = float(recommended["width_override_confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("polarity threshold report has invalid recommended_thresholds") from exc
    for name, value in (
        ("v2_gate_confidence", v2_gate),
        ("width_override_confidence", width_gate),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"polarity threshold report contains invalid {name}: {value!r}")
    return report, width_gate, v2_gate


def _same_gate(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mitdb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scale-method", default="windowed_std")
    parser.add_argument(
        "--width-override-confidence",
        type=_confidence_arg,
        default=None,
        help=(
            "Frozen polarity width-override cutoff. Required unless "
            "--polarity-threshold-report supplies the verified frozen value."
        ),
    )
    parser.add_argument(
        "--v2-gate-confidence",
        type=_confidence_arg,
        default=None,
        help=(
            "Count-ratio confidence below which polarity_v2 takes over. The historical default "
            f"is {DEFAULT_V2_GATE_CONFIDENCE:.2f}. A non-default value requires "
            "--polarity-threshold-report so it cannot be tuned directly on this evaluator."
        ),
    )
    parser.add_argument(
        "--polarity-threshold-report",
        type=Path,
        default=None,
        help=(
            "Freeze-eligible v2 derivation artifact from "
            "derive_polarity_thresholds_mitdb_extended.py. When supplied, its "
            "recommended_thresholds are authoritative."
        ),
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

    derivation_report = None
    derivation_metadata = None
    if args.polarity_threshold_report is not None:
        derivation_report, report_width, report_v2 = _load_polarity_threshold_report(
            args.polarity_threshold_report
        )
        if (
            args.width_override_confidence is not None
            and not _same_gate(args.width_override_confidence, report_width)
        ):
            raise SystemExit(
                "--width-override-confidence disagrees with the derivation report's frozen value"
            )
        if args.v2_gate_confidence is not None and not _same_gate(args.v2_gate_confidence, report_v2):
            raise SystemExit(
                "--v2-gate-confidence disagrees with the derivation report's frozen value"
            )
        width_override_confidence = report_width
        v2_gate_confidence = report_v2
        derivation_metadata = {
            "path": str(args.polarity_threshold_report),
            "sha256": sha256_file(args.polarity_threshold_report),
            "schema": derivation_report["schema"],
            "selection_status": derivation_report.get("selection_status"),
            "recommended_thresholds": derivation_report.get("recommended_thresholds"),
        }
    else:
        if args.width_override_confidence is None:
            raise SystemExit(
                "--width-override-confidence is required unless --polarity-threshold-report is supplied"
            )
        width_override_confidence = args.width_override_confidence
        v2_gate_confidence = (
            DEFAULT_V2_GATE_CONFIDENCE
            if args.v2_gate_confidence is None
            else args.v2_gate_confidence
        )
        if not _same_gate(v2_gate_confidence, DEFAULT_V2_GATE_CONFIDENCE):
            raise SystemExit(
                "A non-default --v2-gate-confidence requires --polarity-threshold-report"
            )

    model = CandidateSuppressor.load(args.model)
    threshold = float(model.metadata.threshold if args.threshold is None else args.threshold)

    names = list(LOCKED_HELDOUT_RECORDS if args.records is None else args.records)
    if len(names) != len(set(names)):
        raise SystemExit("--records must not contain duplicates")
    outside_locked = sorted(set(names) - set(LOCKED_HELDOUT_RECORDS))
    if outside_locked:
        raise SystemExit(
            "--records may only select from the locked MIT-BIH split: "
            + ", ".join(outside_locked)
        )
    full_locked_split = len(names) == len(LOCKED_HELDOUT_RECORDS) and set(names) == set(LOCKED_HELDOUT_RECORDS)

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
            width_override_confidence=width_override_confidence,
            v2_gate_confidence=v2_gate_confidence,
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

    if not full_locked_split or args.threshold is not None:
        evidence_status = "diagnostic_locked_subset_or_operating_point_override"
    elif args.polarity == "adaptive":
        evidence_status = "legacy_validation_non_regression"
    else:
        evidence_status = "locked_heldout_model_evaluation"

    report = {
        "schema": "electrotrace.mitdb_frozen_model_evaluation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": evidence_status,
        "purpose": (
            "Non-regression audit of an already-trained model on the locked MIT-BIH split. "
            "For adaptive polarity this is legacy validation, not prospective held-out evidence, "
            "because the adaptive mechanism was historically informed by record 207/pooled MIT-BIH behavior."
        ),
        "evaluation_integrity": {
            "full_locked_split": full_locked_split,
            "retraining": False,
            "adaptive_polarity_historically_informed_by_locked_data": args.polarity == "adaptive",
            "threshold_derivation_report_verified": derivation_metadata is not None,
            "prospective_adaptive_validation": False if args.polarity == "adaptive" else None,
            "interpretation": (
                "legacy non-regression only for adaptive polarity"
                if args.polarity == "adaptive"
                else "model evaluation on the locked split; no adaptive-polarity claim"
            ),
        },
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
            "width_override_confidence": width_override_confidence,
            "v2_gate_confidence": v2_gate_confidence,
            "v2_gate_confidence_is_historical_default": _same_gate(
                v2_gate_confidence, DEFAULT_V2_GATE_CONFIDENCE
            ),
            "polarity_threshold_derivation": derivation_metadata,
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
            "; this remains legacy validation for adaptive polarity because record 207 historically "
            "informed the mechanism. A clean threshold derivation prevents further tuning but cannot "
            "retroactively restore prospective held-out status."
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
