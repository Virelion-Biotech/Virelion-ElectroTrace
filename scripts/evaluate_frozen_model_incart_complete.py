#!/usr/bin/env python3
"""Complete frozen-v4 INCART characterization with audited reference repairs.

INCART is already exposed development data for the v4/windowed-std generation.
This runner does not retrain, recalibrate, or select parameters. It evaluates
the already-frozen detector across the complete official 75-record manifest,
while refusing to silently repair malformed references unless an independent
certified WFDB audit authorizes the edge-only repair.

A source record whose official annotations cannot pass the reference-integrity
rule remains an explicit source-reference exclusion. It is never replaced with
detector-derived labels merely to force a 75/75 score count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.lead_quality import compute_lead_quality
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, RecordValidation, match_peaks, summarize_records
from electrotrace.wfdb_records import (
    AnnotatedRecord,
    RecordExcluded,
    clean_reference_annotations,
    load_annotated_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RECORDS = [f"I{i:02d}" for i in range(1, 76)]
MALFORMED_REFERENCE_RECORDS = {"I04", "I17", "I35", "I44", "I57", "I72", "I74"}
EXPECTED_AUDIT_SCHEMA = "electrotrace.certified_wfdb_incart_baselines/v1"
MIN_CERTIFIED_ALIGNMENT_COVERAGE = 0.5


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "missing"
    return out


def _same_number(actual, expected, *, atol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def verify_local_manifest(incart_dir: Path) -> dict[str, str]:
    records_path = incart_dir / "RECORDS"
    if not records_path.is_file():
        raise SystemExit(f"Missing official RECORDS manifest: {records_path}")
    actual = [
        line.strip()
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if actual != CANONICAL_RECORDS:
        raise SystemExit(
            "INCART RECORDS manifest mismatch; refusing a changed cohort. "
            f"expected={CANONICAL_RECORDS}, actual={actual}"
        )

    missing = []
    for record in CANONICAL_RECORDS:
        for suffix in (".hea", ".dat", ".atr"):
            path = incart_dir / f"{record}{suffix}"
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(path.name)
    if missing:
        raise SystemExit(
            "Incomplete INCART core files; refusing a silently smaller cohort: "
            + ", ".join(missing[:12])
            + (f" (+{len(missing)-12} more)" if len(missing) > 12 else "")
        )

    hashes = {"RECORDS": sha256_file(records_path)}
    for record in CANONICAL_RECORDS:
        for suffix in (".hea", ".dat", ".atr"):
            key = f"{record}{suffix}"
            hashes[key] = sha256_file(incart_dir / key)
    return hashes


def _audit_by_record(detector_payload: dict) -> dict[str, dict]:
    records = detector_payload.get("annotation_audit", {}).get("records", [])
    return {str(row["record"]): row for row in records}


def load_certified_repair_audit(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise SystemExit("Unexpected certified INCART audit schema")
    protocol = report.get("protocol", {})
    if protocol.get("annotation_policy") != "drop_edges":
        raise SystemExit("Certified INCART audit did not use drop_edges")
    if float(protocol.get("tolerance_ms", -1)) != 75.0:
        raise SystemExit("Certified INCART audit tolerance mismatch")
    if int(protocol.get("channel", -1)) != 0:
        raise SystemExit("Certified INCART audit channel mismatch")

    results = report.get("results", {})
    missing_detectors = [name for name in ("gqrs", "sqrs") if name not in results]
    if missing_detectors:
        raise SystemExit(f"Certified audit missing detectors: {missing_detectors}")

    detector_audits = {
        name: _audit_by_record(results[name]) for name in ("gqrs", "sqrs")
    }
    authorization: dict[str, list[dict]] = {}
    for record in sorted(MALFORMED_REFERENCE_RECORDS):
        observations = []
        for detector in ("gqrs", "sqrs"):
            row = detector_audits[detector].get(record)
            if row is None:
                raise SystemExit(
                    f"Certified audit has no annotation-integrity row for {detector}/{record}"
                )
            approved = (
                row.get("action") == "kept_after_dropping_edges"
                and row.get("alignment_checked") is True
                and row.get("alignment_coverage") is not None
                and float(row["alignment_coverage"]) >= MIN_CERTIFIED_ALIGNMENT_COVERAGE
                and int(row.get("n_interior_invalid", -1)) == 0
            )
            observations.append(
                {
                    "detector": detector,
                    "approved": bool(approved),
                    "action": row.get("action"),
                    "reason": row.get("reason"),
                    "alignment_coverage": row.get("alignment_coverage"),
                    "alignment_median_offset_ms": row.get("alignment_median_offset_ms"),
                    "n_leading_invalid": row.get("n_leading_invalid"),
                    "n_trailing_invalid": row.get("n_trailing_invalid"),
                    "n_interior_invalid": row.get("n_interior_invalid"),
                    "n_reference_kept": row.get("n_reference_kept"),
                }
            )
        authorization[record] = observations
    return report, authorization


def certified_authorized(observations: list[dict]) -> bool:
    """At least one certified detector independently authorizes edge repair."""
    return any(bool(row.get("approved")) for row in observations)


def certified_consensus_authorized(observations: list[dict]) -> bool:
    """Require both pinned certified reference detectors to authorize repair."""
    approved = {
        str(row.get("detector"))
        for row in observations
        if bool(row.get("approved"))
    }
    return approved == {"gqrs", "sqrs"}


def _load_certified_detector_peaks(
    certified_work_dir: Path,
    detector: str,
    record: str,
) -> np.ndarray:
    base = certified_work_dir / detector / record
    qrs = base.with_suffix(".qrs")
    if not qrs.is_file() or qrs.stat().st_size <= 0:
        raise SystemExit(
            f"Missing certified detector output required for repair: {qrs}"
        )
    ann = wfdb.rdann(str(base), "qrs")
    peaks = np.asarray(
        [
            int(sample)
            for sample, symbol in zip(ann.sample, ann.symbol)
            if symbol != "|"
        ],
        dtype=np.int64,
    )
    if peaks.size == 0 or np.any(peaks < 0) or np.any(np.diff(peaks) <= 0):
        raise SystemExit(
            f"{detector}/{record}: invalid certified detector peak sequence"
        )
    return peaks


def load_certified_consensus_repaired_record(
    base: Path,
    *,
    certified_work_dir: Path,
    observations: list[dict],
    tolerance_ms: float = 75.0,
) -> AnnotatedRecord:
    """Repair one malformed-edge reference only with two-detector consensus.

    The source annotation itself must still satisfy the normal edge-only
    structural rules. Both pinned certified WFDB detectors are then used as
    independent alignment checks against the exact same cleaned reference.
    """
    record = base.name
    if not certified_consensus_authorized(observations):
        raise SystemExit(
            f"{record}: repair lacks two-detector certified consensus"
        )

    rec = wfdb.rdrecord(str(base), channels=[0], physical=False)
    signal = np.asarray(
        rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0],
        dtype=float,
    )
    fs_hz = float(rec.fs)
    ann = wfdb.rdann(str(base), "atr")

    references: list[np.ndarray] = []
    audits = []
    for detector in ("gqrs", "sqrs"):
        peaks = _load_certified_detector_peaks(
            certified_work_dir, detector, record
        )
        reference, audit = clean_reference_annotations(
            ann.sample,
            ann.symbol,
            n_samples=int(signal.size),
            fs_hz=fs_hz,
            record=record,
            beat_symbols=DEFAULT_BEAT_SYMBOLS,
            policy="drop_edges",
            alignment_peaks=peaks,
            tolerance_ms=tolerance_ms,
            min_alignment_coverage=MIN_CERTIFIED_ALIGNMENT_COVERAGE,
        )
        if audit.action != "kept_after_dropping_edges":
            raise SystemExit(
                f"{record}: {detector} did not validate an edge-only repair"
            )
        references.append(reference)
        audits.append(audit)

    if not np.array_equal(references[0], references[1]):
        raise SystemExit(
            f"{record}: certified detectors validated different cleaned references"
        )

    # The cleaned source reference is determined by the annotation structure;
    # detector outputs are used only to validate alignment, not to create labels.
    return AnnotatedRecord(
        record=record,
        fs_hz=fs_hz,
        signal=signal,
        reference=references[0],
        audit=audits[0],
    )


def _load_frozen_polarity_gates(path: Path) -> tuple[dict, float, float]:
    root = str(REPO_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from scripts.evaluate_frozen_model_mitdb import (
            _load_polarity_threshold_report,
        )
        report, width_gate, v2_gate = _load_polarity_threshold_report(path)
    finally:
        if added:
            sys.path.remove(root)
    if report.get("schema") != "electrotrace.mitdb_polarity_thresholds_extended_derivation/v3":
        raise SystemExit("Unexpected polarity-threshold derivation schema")
    if not _same_number(v2_gate, 0.0) or not _same_number(width_gate, 0.0):
        raise SystemExit(
            f"Expected leakage-safe frozen gates 0.0/0.0; got v2={v2_gate}, width={width_gate}"
        )
    return report, float(width_gate), float(v2_gate)


def verify_frozen_model(model: CandidateSuppressor) -> None:
    metadata = model.metadata.to_dict()
    expected_exact = {
        "model_version": "rf-candidate-suppressor-v4",
        "feature_schema_version": "candidate-features-v4",
        "calibration_method": "locked_MITBIH_calibration_records",
    }
    for key, expected in expected_exact.items():
        if metadata.get(key) != expected:
            raise SystemExit(
                f"Frozen model metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    expected_numeric = {
        "threshold": 0.2751648051220502,
        "target_recall": 0.995,
        "random_seed": 42,
        "n_estimators": 150,
    }
    for key, expected in expected_numeric.items():
        if not _same_number(metadata.get(key), expected):
            raise SystemExit(
                f"Frozen model metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )


def _bootstrap_macro(record_rows: list[dict], *, replicates: int, seed: int) -> dict:
    if not record_rows:
        return {"replicates": replicates, "seed": seed, "intervals": {}}
    matrix = np.asarray(
        [
            [
                float(row["sensitivity"]),
                float(row["positive_predictive_value"]),
                float(row["f1"]),
            ]
            for row in record_rows
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    n = len(record_rows)
    boot = np.empty((replicates, 3), dtype=float)
    for i in range(replicates):
        sample = matrix[rng.integers(0, n, size=n)]
        boot[i] = np.mean(sample, axis=0)
    names = ("sensitivity", "positive_predictive_value", "f1")
    intervals = {}
    for j, name in enumerate(names):
        intervals[name] = {
            "lower_95": float(np.percentile(boot[:, j], 2.5)),
            "upper_95": float(np.percentile(boot[:, j], 97.5)),
        }
    return {
        "method": "percentile_record_bootstrap_of_macro_record_mean",
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "record",
        "intervals": intervals,
    }


def _macro_metrics(record_rows: list[dict]) -> dict:
    if not record_rows:
        return {"records": 0}
    return {
        "records": len(record_rows),
        "mean_sensitivity": float(np.mean([row["sensitivity"] for row in record_rows])),
        "mean_positive_predictive_value": float(
            np.mean([row["positive_predictive_value"] for row in record_rows])
        ),
        "mean_f1": float(np.mean([row["f1"] for row in record_rows])),
        "median_f1": float(np.median([row["f1"] for row in record_rows])),
        "min_f1": float(np.min([row["f1"] for row in record_rows])),
        "max_f1": float(np.max([row["f1"] for row in record_rows])),
    }


def _pooled_from_rows(record_rows: list[dict]) -> dict:
    tp = sum(int(row["true_positive"]) for row in record_rows)
    fp = sum(int(row["false_positive"]) for row in record_rows)
    fn = sum(int(row["false_negative"]) for row in record_rows)
    ref = sum(int(row["reference_count"]) for row in record_rows)
    detected = sum(int(row["detected_count"]) for row in record_rows)
    sensitivity = tp / ref if ref else 0.0
    ppv = tp / detected if detected else 0.0
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if sensitivity + ppv else 0.0
    return {
        "records": len(record_rows),
        "reference_count": ref,
        "detected_count": detected,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "positive_predictive_value": float(ppv),
        "f1": float(f1),
    }


def _certified_comparison(report: dict, scored_records: list[str]) -> dict:
    wanted = set(scored_records)
    out = {}
    for detector in ("gqrs", "sqrs"):
        rows = [
            row
            for row in report["results"][detector].get("records", [])
            if str(row["record"]) in wanted
        ]
        present = {str(row["record"]) for row in rows}
        missing = sorted(wanted - present)
        out[detector] = {
            "same_electrotrace_scored_cohort": not missing,
            "missing_records": missing,
            "pooled": _pooled_from_rows(rows),
            "macro": _macro_metrics(rows),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incart-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--polarity-threshold-report", type=Path, required=True)
    parser.add_argument("--certified-reference-audit", type=Path, required=True)
    parser.add_argument("--certified-work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    input_hashes = verify_local_manifest(args.incart_dir)
    certified_report, authorization = load_certified_repair_audit(
        args.certified_reference_audit
    )
    derivation, width_gate, v2_gate = _load_frozen_polarity_gates(
        args.polarity_threshold_report
    )
    model = CandidateSuppressor.load(args.model)
    verify_frozen_model(model)

    model_sidecar = Path(str(args.model) + ".json")
    if not model_sidecar.is_file():
        raise SystemExit(f"Missing model sidecar: {model_sidecar}")

    record_results = []
    validation_results = []
    reference_exclusions = []
    polarity_counts: Counter[str] = Counter()

    for index, record in enumerate(CANONICAL_RECORDS, start=1):
        observations = authorization.get(record, [])
        if record in MALFORMED_REFERENCE_RECORDS and not certified_consensus_authorized(
            observations
        ):
            reference_exclusions.append(
                {
                    "record": record,
                    "reason": "edge repair lacks two-detector certified consensus",
                    "certified_audit": observations,
                }
            )
            print(
                f"[{index:02d}/75] {record}: SOURCE-REFERENCE EXCLUSION "
                "(no two-detector certified repair consensus)",
                flush=True,
            )
            continue

        try:
            if record in MALFORMED_REFERENCE_RECORDS:
                annotated = load_certified_consensus_repaired_record(
                    args.incart_dir / record,
                    certified_work_dir=args.certified_work_dir,
                    observations=observations,
                    tolerance_ms=75.0,
                )
            else:
                annotated = load_annotated_record(
                    args.incart_dir / record,
                    channel=0,
                    extension="atr",
                    beat_symbols=DEFAULT_BEAT_SYMBOLS,
                    tolerance_ms=75.0,
                    policy="drop_edges",
                )
        except RecordExcluded as exc:
            reference_exclusions.append(
                {
                    "record": record,
                    "reason": exc.reason,
                    "pan_tompkins_reference_audit": exc.audit.to_dict(),
                    "certified_audit": observations,
                }
            )
            print(
                f"[{index:02d}/75] {record}: SOURCE-REFERENCE EXCLUSION: {exc.reason}",
                flush=True,
            )
            continue

        if annotated.signal is None:
            raise SystemExit(f"{record}: signal unexpectedly unavailable")
        if record in MALFORMED_REFERENCE_RECORDS:
            if annotated.audit.action != "kept_after_dropping_edges":
                raise SystemExit(
                    f"{record}: certified repair was expected but local audit action was "
                    f"{annotated.audit.action!r}"
                )
            if int(annotated.audit.n_interior_invalid) != 0:
                raise SystemExit(f"{record}: interior invalid reference detected")
        elif annotated.audit.action != "kept_unchanged":
            raise SystemExit(
                f"{record}: previously clean record unexpectedly required reference repair"
            )

        retained, _, quality = compute_lead_quality(
            annotated.signal,
            annotated.fs_hz,
            model,
            scale_method="windowed_std",
            v2_gate_confidence=v2_gate,
            width_override_confidence=width_gate,
            threshold=float(model.metadata.threshold),
        )
        retained = np.asarray(retained, dtype=np.int64)
        metrics = match_peaks(
            retained,
            annotated.reference,
            annotated.fs_hz,
            tolerance_ms=75.0,
        )
        result = RecordValidation(record=record, fs_hz=annotated.fs_hz, metrics=metrics)
        validation_results.append(result)

        quality_payload = quality.to_dict()
        polarity = str(quality_payload["selected_polarity"])
        polarity_counts[polarity] += 1
        row = result.to_dict()
        row.update(
            {
                "channel": 0,
                "selected_polarity": polarity,
                "polarity_confidence": float(quality_payload["polarity_confidence"]),
                "stage1_candidate_count": int(quality_payload["stage1_candidate_count"]),
                "stage2_retained_count": int(quality_payload["retained_count"]),
                "retention_fraction": float(quality_payload["retention_fraction"]),
                "reference_audit": annotated.audit.to_dict(),
                "certified_repair_authorization": observations,
            }
        )
        record_results.append(row)
        print(f"[{index:02d}/75] {record}: scored", flush=True)

    if not validation_results:
        raise SystemExit("No INCART records were scientifically scorable")

    scored_names = [row["record"] for row in record_results]
    summary = summarize_records(validation_results)
    summary["macro"] = _macro_metrics(record_results)
    summary["macro_record_bootstrap_95"] = _bootstrap_macro(
        record_results, replicates=args.bootstrap, seed=args.seed
    )

    certified = _certified_comparison(certified_report, scored_names)
    integrity = {
        "official_records_requested": 75,
        "official_manifest_complete": True,
        "records_scored": len(record_results),
        "records_excluded_for_source_reference_integrity": len(reference_exclusions),
        "all_nonexcluded_records_scored": True,
        "reference_exclusions": reference_exclusions,
        "repair_policy": (
            "edge-only repair requires structural validity plus independent alignment "
            "authorization from both pinned certified WFDB detectors; detector outputs "
            "validate the cleaned source annotations and never replace labels"
        ),
        "incart_labels_used_for_model_training": False,
        "incart_labels_used_for_threshold_selection": False,
        "retraining": False,
        "model_threshold_overridden": False,
        "detector_protocol_overridden": False,
    }

    report = {
        "schema": "electrotrace.incart_frozen_v4_complete_characterization/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "exposed_development_database_complete_source_cohort_characterization",
        "interpretation_guardrail": (
            "INCART informed v4/windowed-std development and is not a fresh external "
            "validation set. This report characterizes the complete official source cohort "
            "under explicit reference-integrity exclusions; excluded source annotations are "
            "not replaced with detector-derived labels."
        ),
        "repository": "Virelion-Biotech/Virelion-ElectroTrace",
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "protocol": {
            "dataset": "PhysioNet St Petersburg INCART 12-lead Arrhythmia Database",
            "physionet_database": "incartdb",
            "official_records": CANONICAL_RECORDS,
            "channel": 0,
            "annotation_extension": "atr",
            "beat_symbols": sorted(DEFAULT_BEAT_SYMBOLS),
            "tolerance_ms": 75.0,
            "stage1_scale_method": "windowed_std",
            "polarity": "adaptive",
            "v2_gate_confidence": v2_gate,
            "width_override_confidence": width_gate,
            "recovery": False,
            "operating_threshold": float(model.metadata.threshold),
            "bootstrap_replicates": int(args.bootstrap),
            "bootstrap_seed": int(args.seed),
            "bootstrap_unit": "record",
        },
        "evaluation_integrity": integrity,
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "sidecar_path": str(model_sidecar),
            "sidecar_sha256": sha256_file(model_sidecar),
            "metadata": model.metadata.to_dict(),
        },
        "polarity_threshold_derivation": {
            "path": str(args.polarity_threshold_report),
            "sha256": sha256_file(args.polarity_threshold_report),
            "schema": derivation.get("schema"),
            "recommended_thresholds": derivation.get("recommended_thresholds"),
        },
        "certified_reference_audit": {
            "path": str(args.certified_reference_audit),
            "sha256": sha256_file(args.certified_reference_audit),
            "schema": certified_report.get("schema"),
            "wfdb_toolkit_source_release": certified_report.get(
                "wfdb_toolkit_source_release",
                certified_report.get("protocol", {}).get("wfdb_toolkit_source_release"),
            ),
            "repair_authorization": authorization,
        },
        "input_hashes": input_hashes,
        "selected_polarity_counts": dict(sorted(polarity_counts.items())),
        "summary": summary,
        "certified_wfdb_same_scored_cohort_comparison": certified,
        "record_results": record_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n===== FINAL INCART CHARACTERIZATION =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Reference exclusions:", json.dumps(reference_exclusions, indent=2))
    print("Written:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
