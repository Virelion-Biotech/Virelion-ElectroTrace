#!/usr/bin/env python3
"""Run the preregistered first prospective LTAFDB lead-selector validation.

Lead selection is label-free and occurs before reference annotations are read.
There are intentionally no CLI overrides for channels, window, selector floor,
detector settings, model threshold, polarity gates, or scoring tolerance.
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
from electrotrace.lead_selection import (
    EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
    LEAD_SELECTOR_VERSION,
    choose_two_lead_channel,
)
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, RecordValidation, match_peaks, summarize_records
from electrotrace.validation_detectors import detect_r_peaks_two_stage

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "validation_protocols" / "ltafdb_lead_selector_prospective_v1.json"
EXPECTED_PROTOCOL_GIT_BLOB_SHA = "54e14638526cf43447ae8a7fdf9b34e7a56366c7"
OUTPUT_PATH = (
    REPO_ROOT
    / "validation_reports"
    / "experiments"
    / "2026-09-ltafdb-prospective"
    / "ltafdb_lead_selector_v1_first_run.json"
)
REQUIRED_SUFFIXES = (".hea", ".dat", ".atr")


def _load_polarity_threshold_report(path: Path):
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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "hash-object", str(path)],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception as exc:
        raise SystemExit(f"Unable to verify preregistered protocol Git blob: {exc}") from exc


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "scipy", "sklearn", "skops", "wfdb", "pandas"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "missing"
    return out


def _same_number(actual, expected) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def load_locked_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise SystemExit(f"Missing preregistered protocol: {PROTOCOL_PATH}")
    actual_blob = git_blob_sha(PROTOCOL_PATH)
    if actual_blob != EXPECTED_PROTOCOL_GIT_BLOB_SHA:
        raise SystemExit(
            "Preregistered LTAFDB protocol changed; refusing prospective scoring. "
            f"Expected Git blob {EXPECTED_PROTOCOL_GIT_BLOB_SHA}, got {actual_blob}."
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "electrotrace.prospective_lead_selector_protocol/v1":
        raise SystemExit("Unexpected LTAFDB prospective protocol schema")
    if protocol.get("protocol_id") != "ltafdb-lead-selector-v1-first30m-2026-09-28":
        raise SystemExit("Unexpected LTAFDB prospective protocol id")
    return protocol


def read_local_records(root: Path) -> list[str]:
    path = root / "RECORDS"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return path.read_text(encoding="utf-8").split()


def verify_dataset(root: Path, expected_records: list[str]) -> dict[str, str]:
    actual = read_local_records(root)
    if actual != expected_records:
        missing = sorted(set(expected_records) - set(actual))
        extra = sorted(set(actual) - set(expected_records))
        raise SystemExit(
            "Local LTAFDB RECORDS does not exactly match preregistration: "
            f"missing={missing}, extra={extra}"
        )
    missing_files = []
    for record in expected_records:
        for suffix in REQUIRED_SUFFIXES:
            if not (root / f"{record}{suffix}").is_file():
                missing_files.append(f"{record}{suffix}")
    if missing_files:
        raise SystemExit(
            "Incomplete LTAFDB core files: " + ", ".join(missing_files[:12])
        )
    hashes = {"RECORDS": sha256_file(root / "RECORDS")}
    for record in expected_records:
        for suffix in REQUIRED_SUFFIXES:
            name = f"{record}{suffix}"
            hashes[name] = sha256_file(root / name)
    return hashes


def verify_model(model: CandidateSuppressor, protocol: dict) -> None:
    expected = protocol["frozen_model"]
    metadata = model.metadata.to_dict()
    for field in ("model_version", "feature_schema_version", "calibration_method"):
        if metadata.get(field) != expected[field]:
            raise SystemExit(f"Frozen model metadata mismatch for {field}")
    values = dict(metadata)
    values["operating_threshold"] = metadata.get("threshold")
    for field in (
        "operating_threshold", "target_recall", "n_training_candidates",
        "n_positive_candidates", "n_negative_candidates", "random_seed",
        "n_estimators", "calibration_candidates",
    ):
        if not _same_number(values.get(field), expected[field]):
            raise SystemExit(f"Frozen model metadata mismatch for {field}")


def retained_probability_p50(probabilities: np.ndarray, *, empty_value: float) -> float:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("retained probabilities must be finite one-dimensional data")
    return float(np.median(p)) if p.size else float(empty_value)


def _reference_from_annotation(annotation, beat_symbols: frozenset[str], stop_sample: int) -> tuple[np.ndarray, dict]:
    samples = np.asarray(annotation.sample, dtype=int)
    symbols = list(annotation.symbol)
    if samples.ndim != 1 or len(symbols) != len(samples):
        raise ValueError("invalid WFDB annotation structure")
    if np.any(samples < 0) or np.any(samples >= stop_sample):
        raise ValueError("annotation samples fall outside frozen evaluation window")
    mask = np.asarray([symbol in beat_symbols for symbol in symbols], dtype=bool)
    reference = samples[mask]
    if reference.size == 0:
        raise ValueError("no reference beats remain after frozen beat-symbol filtering")
    if reference.size > 1 and np.any(np.diff(reference) <= 0):
        raise ValueError("reference beats are not strictly increasing")
    return reference, {
        "total_annotations_in_window": int(samples.size),
        "reference_beats_kept": int(reference.size),
        "nonbeat_annotations_dropped": int((~mask).sum()),
    }


def evaluate_record(
    root: Path,
    record: str,
    *,
    protocol: dict,
    model: CandidateSuppressor,
    v2_gate: float,
    width_gate: float,
) -> tuple[RecordValidation, dict, dict]:
    dataset = protocol["dataset"]
    detector = protocol["detector"]
    selector = protocol["lead_selector"]
    window = dataset["window"]
    start = int(window["start_sample"])
    stop = int(window["stop_sample_exclusive"])
    expected_fs = float(dataset["expected_sampling_frequency_hz"])
    scale_method = str(detector["stage1_scale_method"])

    # Critical leakage barrier: signals and model outputs only before lead choice.
    rec = wfdb.rdrecord(
        str(root / record),
        sampfrom=start,
        sampto=stop,
        channels=list(selector["candidate_channels"]),
        physical=True,
    )
    if not _same_number(rec.fs, expected_fs):
        raise ValueError(f"unexpected sampling frequency {rec.fs}")
    signals = np.asarray(rec.p_signal, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != 2:
        raise ValueError(f"expected exactly two loaded channels, got {signals.shape}")
    if signals.shape[0] != stop - start:
        raise ValueError("record does not contain the complete frozen 30-minute window")
    if not np.isfinite(signals).all():
        raise ValueError("signal window contains non-finite values")

    detections = []
    p50 = []
    for channel in (0, 1):
        retained, probabilities = detect_r_peaks_two_stage(
            signals[:, channel],
            expected_fs,
            model,
            polarity=str(detector["polarity"]),
            recovery=bool(detector["recovery"]),
            scale_method=scale_method,
            width_override_confidence=width_gate,
            v2_gate_confidence=v2_gate,
            threshold=float(model.metadata.threshold),
        )
        detections.append(np.asarray(retained, dtype=int))
        p50.append(
            retained_probability_p50(
                probabilities,
                empty_value=float(selector["empty_retained_probability_p50"]),
            )
        )

    selected = choose_two_lead_channel(
        p50[0],
        p50[1],
        primary_floor=float(selector["primary_retained_probability_p50_floor"]),
    )

    # Reference annotations are deliberately loaded only after selected is frozen.
    annotation = wfdb.rdann(
        str(root / record),
        str(detector["annotation_extension"]),
        sampfrom=start,
        sampto=stop,
    )
    beat_symbols = frozenset(detector["beat_symbols"])
    reference, audit = _reference_from_annotation(annotation, beat_symbols, stop - start)
    metrics = match_peaks(
        detections[selected],
        reference,
        expected_fs,
        tolerance_ms=float(detector["tolerance_ms"]),
    )
    result = RecordValidation(record=record, fs_hz=expected_fs, metrics=metrics)
    payload = result.to_dict()
    payload.update({
        "selected_channel": int(selected),
        "signal_names": list(rec.sig_name or []),
        "channel0_retained_probability_p50": p50[0],
        "channel1_retained_probability_p50": p50[1],
        "channel0_detected_count": int(len(detections[0])),
        "channel1_detected_count": int(len(detections[1])),
        "reference_annotation_count": int(len(reference)),
    })
    audit.update({
        "record": record,
        "lead_selected_before_annotation_load": True,
        "selected_channel": int(selected),
    })
    return result, payload, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltafdb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--polarity-threshold-report", type=Path, required=True)
    parser.add_argument("--confirm-first-scored-run", action="store_true")
    args = parser.parse_args()

    if not args.confirm_first_scored_run:
        raise SystemExit("Refusing to score LTAFDB without --confirm-first-scored-run")
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite first-run prospective artifact: {OUTPUT_PATH}")

    protocol = load_locked_protocol()
    records = list(protocol["dataset"]["records"])
    if len(records) != 84 or len(set(records)) != 84:
        raise SystemExit("Preregistered LTAFDB cohort must contain 84 unique records")
    if protocol["lead_selector"]["version"] != LEAD_SELECTOR_VERSION:
        raise SystemExit("Lead-selector version does not match preregistration")
    if not _same_number(
        protocol["lead_selector"]["primary_retained_probability_p50_floor"],
        EDB_DEVELOPMENT_PRIMARY_P50_FLOOR,
    ):
        raise SystemExit("Lead-selector confidence floor does not match frozen code")
    if frozenset(protocol["detector"]["beat_symbols"]) != DEFAULT_BEAT_SYMBOLS:
        raise SystemExit("Frozen beat-symbol set no longer matches DEFAULT_BEAT_SYMBOLS")

    print("Preflight: verifying and hashing exact 84-record LTAFDB cohort...", flush=True)
    input_hashes = verify_dataset(args.ltafdb_dir, records)
    derivation, width_gate, v2_gate = _load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("Polarity derivation schema mismatch")
    if not _same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("Frozen v2 gate mismatch")
    if not _same_number(width_gate, detector["expected_width_override_confidence"]):
        raise SystemExit("Frozen width gate mismatch")

    model = CandidateSuppressor.load(args.model)
    verify_model(model, protocol)

    results = []
    payloads = []
    audits = []
    selected_channels = Counter()
    for index, record in enumerate(records, start=1):
        try:
            result, payload, audit = evaluate_record(
                args.ltafdb_dir,
                record,
                protocol=protocol,
                model=model,
                v2_gate=v2_gate,
                width_gate=width_gate,
            )
        except Exception as exc:
            raise SystemExit(
                f"Prospective LTAFDB run aborted on record {record}; no partial "
                f"primary result is valid. {type(exc).__name__}: {exc}"
            ) from exc
        results.append(result)
        payloads.append(payload)
        audits.append(audit)
        selected_channels[str(payload["selected_channel"])] += 1
        print(f"[{index:02d}/84] {record}: scored", flush=True)

    if len(results) != 84:
        raise SystemExit("Prospective LTAFDB report requires exactly 84 records")

    summary = summarize_records(results)
    f1_values = np.asarray([row.metrics.f1 for row in results], dtype=float)
    summary["macro_mean_record_f1"] = float(np.mean(f1_values))
    summary["macro_median_record_f1"] = float(np.median(f1_values))
    summary["min_record_f1"] = float(np.min(f1_values))
    summary["max_record_f1"] = float(np.max(f1_values))

    sidecar = Path(str(args.model) + ".json")
    if not sidecar.is_file():
        raise SystemExit(f"Missing model sidecar: {sidecar}")
    report = {
        "schema": "electrotrace.external_ltafdb_lead_selector_validation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "prospective_external_lead_selector_evaluation_first_run",
        "dataset_exposure_after_this_run": "exposed_do_not_reuse_as_prospective_for_changed_selector",
        "protocol": protocol,
        "protocol_file": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "git_blob_sha": EXPECTED_PROTOCOL_GIT_BLOB_SHA,
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "evaluation_integrity": {
            "all_preregistered_records_scored": True,
            "records_scored": 84,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "ltafdb_used_for_model_selection": False,
            "ltafdb_used_for_threshold_selection": False,
            "ltafdb_used_for_lead_selector_tuning": False,
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
            "sidecar_path": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar),
            "metadata": model.metadata.to_dict(),
        },
        "polarity_threshold_derivation": {
            "path": str(args.polarity_threshold_report),
            "sha256": sha256_file(args.polarity_threshold_report),
            "schema": derivation.get("schema"),
            "git_head": derivation.get("git_head"),
            "recommended_thresholds": derivation.get("recommended_thresholds"),
            "implementation_hashes": derivation.get("implementation_hashes"),
        },
        "dataset_input_hashes": input_hashes,
        "selected_channel_counts": dict(sorted(selected_channels.items())),
        "annotation_audits": audits,
        "record_results": payloads,
        "summary": summary,
        "interpretation_guardrail": (
            "This is the first prospective external evaluation of the EDB-informed "
            "two-lead selector v1. LTAFDB is exposed after this run. Any change "
            "motivated by these outcomes requires another untouched dataset."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n===== PROSPECTIVE LTAFDB LEAD-SELECTOR SUMMARY =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Selected channels:", dict(sorted(selected_channels.items())))
    print("Written:", OUTPUT_PATH)
    print("LTAFDB is now exposed; do not retune and reuse it as prospective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
