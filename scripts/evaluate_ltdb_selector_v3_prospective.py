#!/usr/bin/env python3
"""Run the preregistered first prospective SVDB selector-v2 evaluation."""
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
from electrotrace.lead_selection import (
    LEAD_SELECTOR_V3_VERSION,
    STARVATION_ALTERNATE_MIN_RATE_BPM,
    STARVATION_ALTERNATE_MIN_RATE_RATIO,
    STARVATION_PRIMARY_MAX_RATE_BPM,
    STARVATION_PRIMARY_MAX_RETENTION,
    choose_two_lead_channel_v3,
)
from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    RecordValidation,
    match_peaks,
    summarize_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "validation_protocols" / "ltdb_selector_v3_prospective.json"
EXPECTED_PROTOCOL_GIT_BLOB_SHA = "4d45b0b4c4c6743cb62b591d81349250f171fb19"
OUTPUT_PATH = (
    REPO_ROOT / "validation_reports" / "experiments" / "2026-09-ltdb-prospective"
    / "ltdb_selector_v3_first_run.json"
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "hash-object", str(path)], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:
        raise SystemExit(f"Unable to verify protocol Git blob: {exc}") from exc


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
    actual = git_blob_sha(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_GIT_BLOB_SHA:
        raise SystemExit(
            "Preregistered LTDB protocol changed; refusing prospective scoring. "
            f"Expected {EXPECTED_PROTOCOL_GIT_BLOB_SHA}, got {actual}."
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "electrotrace.prospective_lead_selector_protocol/v3":
        raise SystemExit("Unexpected LTDB prospective protocol schema")
    if protocol.get("protocol_id") != "ltdb-starvation-rescue-selector-v3-2026-09-29":
        raise SystemExit("Unexpected LTDB prospective protocol id")
    return protocol


def read_local_records(root: Path) -> list[str]:
    path = root / "RECORDS"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return path.read_text(encoding="utf-8").split()


def verify_dataset(root: Path, protocol: dict) -> dict[str, str]:
    dataset = protocol["dataset"]
    records = list(dataset["records"])
    actual = read_local_records(root)
    if actual != records:
        missing = sorted(set(records) - set(actual))
        extra = sorted(set(actual) - set(records))
        raise SystemExit(
            "Local LTDB RECORDS does not exactly match preregistration: "
            f"missing={missing}, extra={extra}"
        )

    missing_files = []
    for record in records:
        for suffix in REQUIRED_SUFFIXES:
            if not (root / f"{record}{suffix}").is_file():
                missing_files.append(f"{record}{suffix}")
    if missing_files:
        raise SystemExit("Incomplete LTDB core files: " + ", ".join(missing_files[:12]))

    expected_fs = float(dataset["expected_sampling_frequency_hz"])
    required_samples = int(dataset["evaluation_window"]["sampto_exclusive"])
    expected_channel_counts = {
        str(key): int(value)
        for key, value in dataset["expected_channel_counts"].items()
    }
    for record in records:
        header = wfdb.rdheader(str(root / record))
        expected_channels = expected_channel_counts[record]
        if int(header.n_sig) != expected_channels:
            raise SystemExit(
                f"{record}: expected {expected_channels} channels, got {header.n_sig}"
            )
        if int(header.n_sig) < int(dataset["minimum_signal_channels"]):
            raise SystemExit(f"{record}: fewer than two candidate ECG channels")
        if not _same_number(header.fs, expected_fs):
            raise SystemExit(f"{record}: expected {expected_fs} Hz, got {header.fs}")
        if int(header.sig_len) < required_samples:
            raise SystemExit(
                f"{record}: requires at least {required_samples} samples, "
                f"got {header.sig_len}"
            )

    hashes = {"RECORDS": sha256_file(root / "RECORDS")}
    for record in records:
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


def _reference_from_annotation(annotation, beat_symbols: frozenset[str], stop_sample: int):
    samples = np.asarray(annotation.sample, dtype=int)
    symbols = list(annotation.symbol)
    if samples.ndim != 1 or len(symbols) != len(samples):
        raise ValueError("invalid WFDB annotation structure")

    beat_mask = np.asarray([symbol in beat_symbols for symbol in symbols], dtype=bool)
    reference = samples[beat_mask]
    if reference.size == 0:
        raise ValueError("no frozen-symbol reference beats remain")
    if np.any(reference < 0) or np.any(reference >= stop_sample):
        raise ValueError("reference beat samples fall outside frozen record")
    if reference.size > 1 and np.any(np.diff(reference) <= 0):
        raise ValueError("reference beats are not strictly increasing")

    nonbeat_samples = samples[~beat_mask]
    out_of_range_nonbeat = int(
        np.count_nonzero((nonbeat_samples < 0) | (nonbeat_samples >= stop_sample))
    )
    audit = {
        "total_annotations": int(samples.size),
        "reference_beats_kept": int(reference.size),
        "nonbeat_annotations_dropped": int((~beat_mask).sum()),
        "out_of_range_nonbeat_annotations_dropped": out_of_range_nonbeat,
    }
    return reference, audit


def evaluate_record(
    root: Path,
    record: str,
    *,
    protocol: dict,
    model: CandidateSuppressor,
    v2_gate: float,
    width_gate: float,
):
    dataset = protocol["dataset"]
    detector = protocol["detector"]
    selector = protocol["lead_selector"]
    expected_fs = float(dataset["expected_sampling_frequency_hz"])
    window = dataset["evaluation_window"]
    sampfrom = int(window["sampfrom"])
    sampto = int(window["sampto_exclusive"])
    duration_minutes = float(window["duration_seconds"]) / 60.0

    # Leakage barrier: only ECG/model outputs exist before lead selection.
    rec = wfdb.rdrecord(
        str(root / record),
        sampfrom=sampfrom,
        sampto=sampto,
        channels=list(selector["candidate_channels"]),
        physical=True,
    )
    if not _same_number(rec.fs, expected_fs):
        raise ValueError(f"unexpected sampling frequency {rec.fs}")
    signals = np.asarray(rec.p_signal, dtype=float)
    if signals.shape != (sampto - sampfrom, 2):
        raise ValueError(f"invalid frozen signal geometry {signals.shape}")
    if not np.isfinite(signals).all():
        raise ValueError("signal contains non-finite values")

    detections = []
    qualities = []
    for channel in (0, 1):
        retained, _, quality = compute_lead_quality(
            signals[:, channel],
            expected_fs,
            model,
            scale_method=str(detector["stage1_scale_method"]),
            v2_gate_confidence=v2_gate,
            width_override_confidence=width_gate,
            threshold=float(model.metadata.threshold),
        )
        detections.append(np.asarray(retained, dtype=int))
        qualities.append(quality)

    retained_rates = [
        float(quality.retained_count / duration_minutes)
        for quality in qualities
    ]
    selected = choose_two_lead_channel_v3(
        retained_rates[0],
        retained_rates[1],
        qualities[0].retention_fraction,
        qualities[1].retention_fraction,
        qualities[0].retained_probability_p50,
        qualities[1].retained_probability_p50,
        primary_max_rate_bpm=float(selector["primary_max_retained_rate_bpm"]),
        primary_max_retention=float(selector["primary_max_retention_fraction"]),
        alternate_min_rate_bpm=float(selector["alternate_min_retained_rate_bpm"]),
        alternate_min_rate_ratio=float(selector["alternate_min_rate_ratio"]),
    )

    # Reference annotations are loaded only after lead selection is final.
    annotation = wfdb.rdann(
        str(root / record),
        str(detector["annotation_extension"]),
        sampfrom=sampfrom,
        sampto=sampto - 1,
    )
    reference, audit = _reference_from_annotation(
        annotation,
        frozenset(detector["beat_symbols"]),
        sampto - sampfrom,
    )
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
        "channel0_quality": qualities[0].to_dict(),
        "channel1_quality": qualities[1].to_dict(),
        "channel0_retained_rate_bpm": retained_rates[0],
        "channel1_retained_rate_bpm": retained_rates[1],
        "reference_annotation_count": int(reference.size),
    })
    audit.update({
        "record": record,
        "lead_selected_before_annotation_load": True,
        "selected_channel": int(selected),
    })
    return result, payload, audit

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltdb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--polarity-threshold-report", type=Path, required=True)
    parser.add_argument("--confirm-first-scored-run", action="store_true")
    args = parser.parse_args()

    if not args.confirm_first_scored_run:
        raise SystemExit("Refusing to score LTDB without --confirm-first-scored-run")
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite first-run artifact: {OUTPUT_PATH}")

    protocol = load_locked_protocol()
    records = list(protocol["dataset"]["records"])
    if len(records) != 7 or len(set(records)) != 7:
        raise SystemExit("LTDB protocol must contain exactly 7 unique records")

    selector = protocol["lead_selector"]
    if selector["version"] != LEAD_SELECTOR_V3_VERSION:
        raise SystemExit("Selector-v3 version mismatch")
    frozen_constants = (
        ("primary_max_retained_rate_bpm", STARVATION_PRIMARY_MAX_RATE_BPM),
        ("primary_max_retention_fraction", STARVATION_PRIMARY_MAX_RETENTION),
        ("alternate_min_retained_rate_bpm", STARVATION_ALTERNATE_MIN_RATE_BPM),
        ("alternate_min_rate_ratio", STARVATION_ALTERNATE_MIN_RATE_RATIO),
    )
    for key, expected in frozen_constants:
        if not _same_number(selector[key], expected):
            raise SystemExit(f"Selector-v3 frozen constant mismatch for {key}")
    if frozenset(protocol["detector"]["beat_symbols"]) != DEFAULT_BEAT_SYMBOLS:
        raise SystemExit("Frozen beat-symbol set no longer matches DEFAULT_BEAT_SYMBOLS")

    print("Preflight: verifying and hashing exact 7-record LTDB cohort...", flush=True)
    input_hashes = verify_dataset(args.ltdb_dir, protocol)
    derivation, width_gate, v2_gate = _load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("Polarity derivation schema mismatch")
    if not _same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("Frozen v2 polarity gate mismatch")
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
                args.ltdb_dir,
                record,
                protocol=protocol,
                model=model,
                v2_gate=v2_gate,
                width_gate=width_gate,
            )
        except Exception as exc:
            raise SystemExit(
                f"Prospective LTDB run aborted on record {record}; no partial "
                f"primary result is valid. {type(exc).__name__}: {exc}"
            ) from exc
        results.append(result)
        payloads.append(payload)
        audits.append(audit)
        selected_channels[str(payload["selected_channel"])] += 1
        print(f"[{index:02d}/7] {record}: scored", flush=True)

    if len(results) != 7:
        raise SystemExit("Prospective LTDB report requires exactly 7 records")
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
        "schema": "electrotrace.external_ltdb_selector_v3_validation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "prospective_external_selector_v3_evaluation_first_run",
        "dataset_exposure_after_this_run": (
            "exposed_do_not_reuse_as_prospective_for_changed_selector"
        ),
        "protocol": protocol,
        "protocol_file": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "git_blob_sha": EXPECTED_PROTOCOL_GIT_BLOB_SHA,
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "evaluation_integrity": {
            "all_preregistered_records_scored": True,
            "records_scored": 7,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "ltdb_used_for_model_selection": False,
            "ltdb_used_for_threshold_selection": False,
            "ltdb_used_for_lead_selector_tuning": False,
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
            "This is the first prospective untouched-record evaluation of "
            "starvation-rescue selector v3 on MIT-BIH Long-Term ECG Database. "
            "LTDB is now exposed. The same MIT-BIH/Beth Israel ecosystem "
            "limitation applies, and the rate heuristic is not a physiologic "
            "lower-heart-rate validity rule."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n===== PROSPECTIVE LTDB SELECTOR-V3 SUMMARY =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Selected channels:", dict(sorted(selected_channels.items())))
    print("Written:", OUTPUT_PATH)
    print("LTDB is now exposed; do not retune and reuse it as prospective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
