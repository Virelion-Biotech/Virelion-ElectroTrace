#!/usr/bin/env python3
"""Run the preregistered one-shot prospective EDB validation.

This runner is intentionally narrow. It evaluates the already-frozen
rf-candidate-suppressor-v4 / windowed_std ElectroTrace detector on the European
ST-T Database using the immutable protocol in
validation_protocols/edb_prospective_v1.json.

There are deliberately no CLI switches for channel, tolerance, polarity,
recovery, scale method, model threshold, v2 gate, or width gate. Changing any
of those after EDB outcomes are visible would make EDB development/exposed
data rather than prospective validation data.
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

from electrotrace import __version__
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    RecordValidation,
    match_peaks,
    summarize_records,
)
from electrotrace.validation_detectors import (
    detect_r_peaks_two_stage,
    select_signal_polarity,
)
from electrotrace.wfdb_records import load_annotated_record

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "validation_protocols" / "edb_prospective_v1.json"
EXPECTED_PROTOCOL_SHA256 = "05a1aa75069f21fb80ac649da9a800aa60c32d80a12e01ea17fce22c63cd3e8f"
OUTPUT_PATH = (
    REPO_ROOT
    / "validation_reports"
    / "experiments"
    / "2026-09-edb-prospective"
    / "edb_frozen_v4_prospective_first_run.json"
)
REQUIRED_SUFFIXES = (".hea", ".dat", ".atr")


def _load_polarity_threshold_report(path: Path):
    # Imported lazily so direct execution of this sibling script remains robust
    # without violating module-level import ordering.
    root = str(REPO_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from scripts.evaluate_frozen_model_mitdb import (
            _load_polarity_threshold_report as loader,
        )
        return loader(path)
    finally:
        if added:
            sys.path.remove(root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "missing"
    return out


def _same_number(actual, expected) -> bool:
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def load_locked_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise SystemExit(f"Missing preregistered protocol: {PROTOCOL_PATH}")
    actual_hash = sha256_file(PROTOCOL_PATH)
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit(
            "Preregistered EDB protocol has changed. Refusing prospective scoring. "
            f"Expected SHA-256 {EXPECTED_PROTOCOL_SHA256}, got {actual_hash}."
        )
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid preregistered protocol JSON: {exc}") from exc

    if protocol.get("schema") != "electrotrace.prospective_external_protocol/v1":
        raise SystemExit("Unexpected prospective protocol schema")
    if protocol.get("protocol_id") != "edb-frozen-v4-windowed-std-2026-09-28":
        raise SystemExit("Unexpected prospective protocol id")
    return protocol


def read_local_records(edb_dir: Path) -> list[str]:
    records_file = edb_dir / "RECORDS"
    if not records_file.is_file():
        raise SystemExit(
            f"Missing {records_file}. Download the complete PhysioNet edb database."
        )
    return [
        line.strip()
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_dataset(edb_dir: Path, expected_records: list[str]) -> dict[str, str]:
    actual_records = read_local_records(edb_dir)
    if actual_records != expected_records:
        missing = sorted(set(expected_records) - set(actual_records))
        extra = sorted(set(actual_records) - set(expected_records))
        raise SystemExit(
            "Local EDB RECORDS list does not exactly match the preregistered "
            f"90-record cohort. missing={missing}, extra={extra}"
        )

    missing_files: list[str] = []
    for record in expected_records:
        for suffix in REQUIRED_SUFFIXES:
            path = edb_dir / f"{record}{suffix}"
            if not path.is_file():
                missing_files.append(path.name)
    if missing_files:
        preview = ", ".join(missing_files[:12])
        more = "" if len(missing_files) <= 12 else f" (+{len(missing_files)-12} more)"
        raise SystemExit(
            "Incomplete EDB core files; refusing a silently smaller cohort: "
            f"{preview}{more}"
        )

    hashes = {"RECORDS": sha256_file(edb_dir / "RECORDS")}
    for record in expected_records:
        for suffix in REQUIRED_SUFFIXES:
            key = f"{record}{suffix}"
            hashes[key] = sha256_file(edb_dir / key)
    return hashes


def verify_model(model: CandidateSuppressor, protocol: dict) -> None:
    expected = protocol["frozen_model"]
    metadata = model.metadata.to_dict()

    exact_fields = (
        "model_version",
        "feature_schema_version",
        "calibration_method",
    )
    for field in exact_fields:
        if metadata.get(field) != expected[field]:
            raise SystemExit(
                f"Frozen model metadata mismatch for {field}: "
                f"{metadata.get(field)!r} != {expected[field]!r}"
            )

    numeric_fields = (
        "operating_threshold",
        "target_recall",
        "n_training_candidates",
        "n_positive_candidates",
        "n_negative_candidates",
        "random_seed",
        "n_estimators",
        "calibration_candidates",
    )
    model_values = dict(metadata)
    model_values["operating_threshold"] = metadata.get("threshold")
    for field in numeric_fields:
        if not _same_number(model_values.get(field), expected[field]):
            raise SystemExit(
                f"Frozen model metadata mismatch for {field}: "
                f"{model_values.get(field)!r} != {expected[field]!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--polarity-threshold-report", type=Path, required=True)
    parser.add_argument(
        "--confirm-first-scored-run",
        action="store_true",
        help=(
            "Required acknowledgement that EDB outcomes have not previously been "
            "used to tune this model generation. After this run, EDB is exposed."
        ),
    )
    args = parser.parse_args()

    if not args.confirm_first_scored_run:
        raise SystemExit(
            "Refusing to score EDB without --confirm-first-scored-run. "
            "The first scored EDB result is one-shot prospective evidence."
        )
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"Refusing to overwrite first-run prospective artifact: {OUTPUT_PATH}"
        )

    protocol = load_locked_protocol()
    expected_records = list(protocol["dataset"]["records"])
    if len(expected_records) != 90 or len(set(expected_records)) != 90:
        raise SystemExit("Preregistered EDB cohort must contain 90 unique records")

    print("Preflight: hashing and verifying the exact 90-record EDB cohort...", flush=True)
    input_hashes = verify_dataset(args.edb_dir, expected_records)

    derivation_report, width_gate, v2_gate = _load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector_protocol = protocol["detector"]
    if derivation_report.get("schema") != detector_protocol[
        "polarity_threshold_report_schema"
    ]:
        raise SystemExit("Polarity derivation schema does not match preregistration")
    if not _same_number(v2_gate, detector_protocol["expected_v2_gate_confidence"]):
        raise SystemExit(
            f"Frozen v2 gate mismatch: {v2_gate} != "
            f"{detector_protocol['expected_v2_gate_confidence']}"
        )
    if not _same_number(
        width_gate, detector_protocol["expected_width_override_confidence"]
    ):
        raise SystemExit(
            f"Frozen width gate mismatch: {width_gate} != "
            f"{detector_protocol['expected_width_override_confidence']}"
        )

    model = CandidateSuppressor.load(args.model)
    verify_model(model, protocol)

    channel = int(detector_protocol["channel"])
    tolerance_ms = float(detector_protocol["tolerance_ms"])
    scale_method = str(detector_protocol["stage1_scale_method"])
    beat_symbols = frozenset(detector_protocol["beat_symbols"])
    if beat_symbols != DEFAULT_BEAT_SYMBOLS:
        raise SystemExit(
            "Preregistered beat-symbol set no longer matches ElectroTrace "
            "DEFAULT_BEAT_SYMBOLS; resolve explicitly before scoring EDB."
        )

    results: list[RecordValidation] = []
    record_payloads: list[dict] = []
    audits: list[dict] = []
    polarity_counts: Counter[str] = Counter()

    for index, record in enumerate(expected_records, start=1):
        try:
            annotated = load_annotated_record(
                args.edb_dir / record,
                channel=channel,
                extension=str(detector_protocol["annotation_extension"]),
                beat_symbols=beat_symbols,
                tolerance_ms=tolerance_ms,
                policy=str(protocol["integrity"]["annotation_policy"]),
            )
            if annotated.signal is None:
                raise RuntimeError("signal unexpectedly not loaded")

            polarity = select_signal_polarity(
                annotated.signal,
                annotated.fs_hz,
                scale_method=scale_method,
                width_override_confidence=width_gate,
                v2_gate_confidence=v2_gate,
            )
            retained, _ = detect_r_peaks_two_stage(
                annotated.signal,
                annotated.fs_hz,
                model,
                polarity=str(detector_protocol["polarity"]),
                recovery=bool(detector_protocol["recovery"]),
                scale_method=scale_method,
                width_override_confidence=width_gate,
                v2_gate_confidence=v2_gate,
                threshold=float(model.metadata.threshold),
            )
            metrics = match_peaks(
                retained,
                annotated.reference,
                annotated.fs_hz,
                tolerance_ms=tolerance_ms,
            )
        except Exception as exc:
            raise SystemExit(
                f"Prospective EDB run aborted on record {record}; no partial "
                f"primary result is valid. {type(exc).__name__}: {exc}"
            ) from exc

        result = RecordValidation(
            record=record,
            fs_hz=annotated.fs_hz,
            metrics=metrics,
        )
        payload = result.to_dict()
        payload["channel"] = channel
        payload["selected_polarity"] = polarity.polarity
        payload["polarity_confidence"] = float(polarity.confidence)
        payload["reference_annotation_count"] = int(len(annotated.reference))
        results.append(result)
        record_payloads.append(payload)
        audits.append(annotated.audit.to_dict())
        polarity_counts[polarity.polarity] += 1

        print(
            f"[{index:02d}/90] {record}: "
            f"polarity={polarity.polarity} "
            f"sens={metrics.sensitivity:.4f} "
            f"ppv={metrics.positive_predictive_value:.4f} "
            f"f1={metrics.f1:.4f}",
            flush=True,
        )

    if len(results) != 90:
        raise SystemExit("Prospective EDB primary report requires exactly 90 records")

    summary = summarize_records(results)
    f1_values = np.asarray([row.metrics.f1 for row in results], dtype=float)
    summary["macro_mean_record_f1"] = float(np.mean(f1_values))
    summary["macro_median_record_f1"] = float(np.median(f1_values))
    summary["min_record_f1"] = float(np.min(f1_values))
    summary["max_record_f1"] = float(np.max(f1_values))

    model_sidecar = Path(str(args.model) + ".json")
    if not model_sidecar.is_file():
        raise SystemExit(f"Missing model metadata sidecar after successful load: {model_sidecar}")

    report = {
        "schema": "electrotrace.external_edb_prospective_validation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "prospective_external_evaluation_first_run",
        "dataset_exposure_after_this_run": "exposed_do_not_reuse_as_prospective_for_changed_model",
        "protocol": protocol,
        "protocol_file": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "evaluation_integrity": {
            "all_preregistered_records_scored": True,
            "records_scored": len(results),
            "records_skipped": 0,
            "channel_selected_before_scoring": True,
            "edb_labels_used_for_model_selection": False,
            "edb_labels_used_for_threshold_selection": False,
            "edb_labels_used_for_polarity_tuning": False,
            "edb_labels_used_for_scale_selection": False,
            "retraining": False,
            "model_threshold_overridden": False,
            "detector_protocol_overridden": False,
            "first_scored_run_acknowledged": True,
        },
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
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
            "schema": derivation_report.get("schema"),
            "git_head": derivation_report.get("git_head"),
            "recommended_thresholds": derivation_report.get("recommended_thresholds"),
            "implementation_hashes": derivation_report.get("implementation_hashes"),
        },
        "dataset_input_hashes": input_hashes,
        "selected_polarity_counts": dict(sorted(polarity_counts.items())),
        "annotation_audits": audits,
        "record_results": record_payloads,
        "summary": summary,
        "interpretation_guardrail": (
            "This is the one-shot prospective EDB result for this frozen model "
            "generation. After inspection, EDB is exposed. Any model or protocol "
            "change motivated by these outcomes requires a different untouched "
            "database for prospective validation."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n===== PROSPECTIVE EDB SUMMARY =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Selected polarities:", dict(sorted(polarity_counts.items())))
    print("Written:", OUTPUT_PATH)
    print("EDB is now exposed for this project; do not retune and reuse it as prospective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
