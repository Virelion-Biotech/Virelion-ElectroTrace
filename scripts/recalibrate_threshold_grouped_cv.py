#!/usr/bin/env python3
"""Is a single recalibrated threshold enough, or does INCART need real fixing?

The false-positive forensics run (2026-09-22) found Stage-2 AUC near ceiling
(median 0.998) even on the 13 worst over-detecting INCART records, with 74% of
false positives sitting in the post-QRS T-wave window. High AUC with poor
precision at the fixed threshold is the signature of a threshold/calibration
problem, not a ranking problem -- so before touching features or architecture,
try recalibrating the threshold honestly.

"Honestly" means record-grouped cross-validation, not a single sweep against
labels you are about to report performance on. This script:

  1. Scores every Stage-1 candidate (the RF itself is frozen throughout) on
     the MIT-BIH calibration records (never the 12-record held-out test split)
     and on INCART records passed under --incart-annotation-policy.
  2. Runs GroupKFold (group = record) over the combined pool, so a record's
     candidates never appear in both a fold's train and test side.
  3. Per fold, picks a threshold on training folds with the same F1/min-recall
     rule as the shipped threshold, then scores held-out records.
  4. Pools every held-out record exactly once into a macro-average per database.
  5. Also fits one threshold on the full combined pool as a deployment
     candidate. That number uses INCART labels and is not a validated result.

This never touches the MIT-BIH 12-record held-out split and never retrains
the RF.

Usage:
  python -u scripts/recalibrate_threshold_grouped_cv.py \
      --mitdb-dir .cache/physionet/mitdb --incart-dir .cache/physionet/incartdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --folds 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor, label_candidates
from electrotrace.threshold_selection import select_threshold_for_f1
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, match_peaks
from electrotrace.validation_detectors import detect_r_peaks_two_stage
from electrotrace.wfdb_records import (
    POLICIES,
    RecordExcluded,
    load_annotated_record,
    summarize_audits,
)

MITDB_CALIBRATION_RECORDS = ["115", "202", "208", "220", "221", "222", "233"]

MIN_RECALL = 0.97
TOLERANCE_S = 0.075


@dataclass
class RecordCandidates:
    record: str
    database: str
    fs_hz: float
    candidates: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    reference: np.ndarray


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


def collect_record(
    model: CandidateSuppressor,
    base: Path,
    database: str,
    *,
    scale_method: str,
    policy: str,
    tolerance_ms: float,
) -> tuple[RecordCandidates | None, dict | None]:
    """Score one record's Stage-1 candidates. Returns (data, skip_reason)."""
    try:
        annotated = load_annotated_record(
            base,
            beat_symbols=DEFAULT_BEAT_SYMBOLS,
            tolerance_ms=tolerance_ms,
            policy=policy,
        )
    except RecordExcluded as exc:
        return None, {
            "record": base.name,
            "database": database,
            "reason": exc.reason,
            "audit": exc.audit,
        }

    candidates, probabilities = detect_r_peaks_two_stage(
        annotated.signal,
        annotated.fs_hz,
        model,
        polarity="adaptive",
        recovery=False,
        scale_method=scale_method,
        threshold=0.0,
    )
    candidates = np.asarray(candidates, dtype=np.int64)
    order = np.argsort(candidates, kind="stable")
    candidates = candidates[order]
    probabilities = np.asarray(probabilities, dtype=float)[order]
    labels = label_candidates(
        candidates, annotated.reference, annotated.fs_hz, tolerance_s=TOLERANCE_S
    )
    return (
        RecordCandidates(
            record=base.name,
            database=database,
            fs_hz=annotated.fs_hz,
            candidates=candidates,
            probabilities=probabilities,
            labels=labels,
            reference=annotated.reference,
        ),
        {
            "record": base.name,
            "database": database,
            "reason": None,
            "audit": annotated.audit,
        },
    )


def evaluate_at_threshold(
    records: list[RecordCandidates], threshold: float, tolerance_ms: float
) -> list[dict]:
    rows = []
    for r in records:
        detected = r.candidates[r.probabilities >= threshold]
        metrics = match_peaks(
            detected, r.reference, r.fs_hz, tolerance_ms=tolerance_ms
        )
        rows.append(
            {
                "record": r.record,
                "database": r.database,
                "threshold": float(threshold),
                **metrics.to_dict(),
            }
        )
    return rows


def macro_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n_records": 0}
    return {
        "n_records": len(rows),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
        "mean_ppv": float(np.mean([r["positive_predictive_value"] for r in rows])),
        "median_f1": float(np.median([r["f1"] for r in rows])),
        "min_f1": float(np.min([r["f1"] for r in rows])),
    }


def group_k_fold(
    groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(groups.tolist()))
    n_splits = max(2, min(n_splits, n_groups))
    try:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:
        splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def run_cv(
    records: list[RecordCandidates],
    n_splits: int,
    seed: int,
    tolerance_ms: float,
) -> dict:
    group_names = np.array([r.record for r in records])
    owner = (
        np.concatenate(
            [np.full(len(r.candidates), i) for i, r in enumerate(records)]
        )
        if records
        else np.array([])
    )
    all_prob = (
        np.concatenate([r.probabilities for r in records])
        if records
        else np.array([])
    )
    all_label = (
        np.concatenate([r.labels for r in records]) if records else np.array([])
    )
    group_per_candidate = group_names[owner] if len(owner) else np.array([])

    folds = group_k_fold(group_names, n_splits, seed)
    fold_reports = []
    pooled_rows: list[dict] = []
    by_index = {i: r for i, r in enumerate(records)}

    for fold_id, (train_record_idx, test_record_idx) in enumerate(folds):
        train_records = {group_names[i] for i in train_record_idx}
        cand_train = np.isin(group_per_candidate, list(train_records))
        if int(all_label[cand_train].sum()) == 0:
            continue
        threshold = select_threshold_for_f1(
            all_label[cand_train],
            all_prob[cand_train],
            min_recall=MIN_RECALL,
        )
        test_records = [by_index[i] for i in test_record_idx]
        rows = evaluate_at_threshold(test_records, threshold, tolerance_ms)
        pooled_rows.extend(rows)
        fold_reports.append(
            {
                "fold": fold_id,
                "threshold": threshold,
                "n_train_records": len(train_records),
                "n_test_records": len(test_records),
                "test_records": sorted(r.record for r in test_records),
                "summary": macro_summary(rows),
            }
        )

    thresholds = [f["threshold"] for f in fold_reports]
    return {
        "n_folds": len(fold_reports),
        "fold_thresholds": thresholds,
        "threshold_mean": float(np.mean(thresholds)) if thresholds else None,
        "threshold_std": float(np.std(thresholds)) if len(thresholds) > 1 else 0.0 if thresholds else None,
        "threshold_min": float(np.min(thresholds)) if thresholds else None,
        "threshold_max": float(np.max(thresholds)) if thresholds else None,
        "folds": fold_reports,
        "pooled_record_results": pooled_rows,
        "pooled_summary": macro_summary(pooled_rows),
        "pooled_summary_by_database": {
            db: macro_summary(
                [r for r in pooled_rows if r["database"] == db]
            )
            for db in sorted({r.database for r in records})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mitdb-dir", type=Path, required=True)
    parser.add_argument("--incart-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scale-method", default="windowed_std")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--incart-annotation-policy",
        choices=POLICIES,
        default="drop_edges",
        help="drop_edges (default): include the 6 repaired records; error: the historical 68.",
    )
    parser.add_argument(
        "--incart-records",
        nargs="*",
        default=None,
        help="Defaults to every local INCART record under --incart-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation_reports/experiments/2026-09-recalibration/threshold_grouped_cv.json"
        ),
    )
    args = parser.parse_args()

    model = CandidateSuppressor.load(args.model)
    current_threshold = float(model.metadata.threshold)

    records: list[RecordCandidates] = []
    skipped: list[dict] = []
    audits = []

    print("Scoring MIT-BIH calibration records (never the 12-record held-out split)...")
    for name in MITDB_CALIBRATION_RECORDS:
        data, info = collect_record(
            model,
            args.mitdb_dir / name,
            "mitdb_calibration",
            scale_method=args.scale_method,
            policy="error",
            tolerance_ms=args.tolerance_ms,
        )
        if info:
            audits.append(info["audit"])
        if data is None:
            skipped.append(
                {
                    "record": name,
                    "database": "mitdb_calibration",
                    "reason": info["reason"],
                }
            )
            print(f"  {name}: SKIPPED: {info['reason']}")
            continue
        records.append(data)
        print(
            f"  {name}: {len(data.candidates)} candidates, "
            f"{int(data.labels.sum())} true"
        )

    incart_names = args.incart_records or sorted(
        p.stem for p in args.incart_dir.glob("*.hea")
    )
    print(
        f"Scoring {len(incart_names)} INCART records "
        f"(annotation-policy={args.incart_annotation_policy})..."
    )
    for index, name in enumerate(incart_names, start=1):
        data, info = collect_record(
            model,
            args.incart_dir / name,
            "incart",
            scale_method=args.scale_method,
            policy=args.incart_annotation_policy,
            tolerance_ms=args.tolerance_ms,
        )
        if info:
            audits.append(info["audit"])
        if data is None:
            skipped.append(
                {
                    "record": name,
                    "database": "incart",
                    "reason": info["reason"],
                }
            )
            print(
                f"  [{index}/{len(incart_names)}] {name}: SKIPPED: "
                f"{info['reason']}"
            )
            continue
        records.append(data)
        if index % 10 == 0 or index == len(incart_names):
            print(f"  [{index}/{len(incart_names)}] {name}: done")

    if len(records) < 4:
        raise SystemExit(
            "Not enough usable records for grouped CV; "
            "check --incart-dir / --mitdb-dir."
        )

    baseline_rows = evaluate_at_threshold(records, current_threshold, args.tolerance_ms)
    cv = run_cv(records, args.folds, args.seed, args.tolerance_ms)

    all_prob = np.concatenate([r.probabilities for r in records])
    all_label = np.concatenate([r.labels for r in records])
    deployment_candidate_threshold = select_threshold_for_f1(
        all_label, all_prob, min_recall=MIN_RECALL
    )
    candidate_rows = evaluate_at_threshold(
        records, deployment_candidate_threshold, args.tolerance_ms
    )

    input_hashes = {}
    for r in records:
        base = (
            args.mitdb_dir if r.database == "mitdb_calibration" else args.incart_dir
        ) / r.record
        for source in sorted(base.parent.glob(f"{r.record}.*")):
            if source.is_file():
                input_hashes[f"{r.record}::{source.name}"] = sha256_file(source)

    report = {
        "schema": "electrotrace.threshold_recalibration_grouped_cv/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "development_only",
        "evidence_note": (
            "INCART and the MIT-BIH calibration records are development data for this "
            "model generation. Pooled CV numbers below are an estimate of how a "
            "recalibrated threshold generalizes across records it was not tuned on, "
            "but they are not a substitute for a fresh held-out database."
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
        "input_hashes": input_hashes,
        "protocol": {
            "mitdb_calibration_records": MITDB_CALIBRATION_RECORDS,
            "mitdb_held_out_records_excluded": True,
            "incart_records_requested": incart_names,
            "incart_annotation_policy": args.incart_annotation_policy,
            "scale_method": args.scale_method,
            "polarity": "adaptive",
            "recovery": False,
            "tolerance_ms": args.tolerance_ms,
            "match_tolerance_s_for_labeling": TOLERANCE_S,
            "min_recall_floor": MIN_RECALL,
            "cv_folds_requested": args.folds,
            "cv_group": "record (GroupKFold; candidates from one record stay in one fold)",
            "seed": args.seed,
            "retraining": False,
        },
        "annotation_audit": {
            "summary": summarize_audits(audits),
            "records": [a.to_dict() for a in audits],
        },
        "current_fixed_threshold": {
            "value": current_threshold,
            "summary_all": macro_summary(baseline_rows),
            "summary_by_database": {
                db: macro_summary(
                    [r for r in baseline_rows if r["database"] == db]
                )
                for db in sorted({r.database for r in records})
            },
        },
        "grouped_cv": cv,
        "deployment_candidate_threshold": {
            "value": deployment_candidate_threshold,
            "note": (
                "Fit on the full combined pool (MIT-BIH calibration + INCART) using the "
                "same rule as the shipped threshold. Uses INCART labels directly -- "
                "diagnostic only, not a validated setting. Compare against "
                "grouped_cv.threshold_mean/std before considering any deployment use."
            ),
            "summary_all": macro_summary(candidate_rows),
            "summary_by_database": {
                db: macro_summary(
                    [r for r in candidate_rows if r["database"] == db]
                )
                for db in sorted({r.database for r in records})
            },
        },
        "skipped_records": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n===== CURRENT FIXED THRESHOLD =====")
    print(
        f"threshold={current_threshold:.4f}",
        json.dumps(report["current_fixed_threshold"]["summary_by_database"]),
    )
    print("\n===== GROUPED CV (pooled, each record held out once) =====")
    print(
        f"{cv['n_folds']} folds, threshold mean={cv['threshold_mean']:.4f} "
        f"std={cv['threshold_std']:.4f} "
        f"range=[{cv['threshold_min']:.4f}, {cv['threshold_max']:.4f}]"
    )
    print(json.dumps(cv["pooled_summary_by_database"]))
    print("\n===== DEPLOYMENT-CANDIDATE THRESHOLD (diagnostic; needs a fresh database) =====")
    print(
        f"threshold={deployment_candidate_threshold:.4f}",
        json.dumps(report["deployment_candidate_threshold"]["summary_by_database"]),
    )
    print("\nWritten:", args.output)
    print("Skipped:", len(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
