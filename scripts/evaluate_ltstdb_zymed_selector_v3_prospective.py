#!/usr/bin/env python3
"""Run the preregistered prospective Zymed-LTSTDB selector-v3 evaluation."""
from __future__ import annotations

import argparse
import importlib
import json
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
    STARVATION_ALTERNATE_RATE_BPM,
    STARVATION_PRIMARY_RATE_BPM,
    STARVATION_RATE_RATIO,
    choose_two_lead_channel_v3,
)
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS, RecordValidation, match_peaks, summarize_records

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_sv = importlib.import_module("scripts.evaluate_svdb_selector_v2_prospective")
PROTOCOL_PATH = REPO_ROOT / "validation_protocols" / "ltstdb_zymed_selector_v3_prospective.json"
EXPECTED_PROTOCOL_GIT_BLOB_SHA = "c0caa0a83da12ba6176cc19f46a9fe3616d1397f"
OUTPUT_PATH = (
    REPO_ROOT / "validation_reports" / "experiments" / "2026-09-ltstdb-zymed-prospective"
    / "ltstdb_zymed_selector_v3_first_run.json"
)
REQUIRED_SUFFIXES = (".hea", ".dat", ".atr")


def load_locked_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise SystemExit(f"Missing preregistered protocol: {PROTOCOL_PATH}")
    actual = _sv.git_blob_sha(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_GIT_BLOB_SHA:
        raise SystemExit(
            "Preregistered LTSTDB protocol changed; refusing prospective scoring. "
            f"Expected {EXPECTED_PROTOCOL_GIT_BLOB_SHA}, got {actual}."
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "electrotrace.prospective_lead_selector_protocol/v3":
        raise SystemExit("Unexpected LTSTDB prospective protocol schema")
    if protocol.get("protocol_id") != "ltstdb-zymed-starvation-rescue-selector-v3-2026-09-29":
        raise SystemExit("Unexpected LTSTDB prospective protocol id")
    return protocol


def verify_dataset(root: Path, protocol: dict) -> dict[str, str]:
    dataset = protocol["dataset"]
    expected = list(dataset["records"])
    manifest = root / "RECORDS"
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}")
    full_records = manifest.read_text(encoding="utf-8").split()
    three_channel = [record for record in full_records if record.startswith("s3")]
    if three_channel != expected:
        raise SystemExit(
            "Canonical LTSTDB three-channel subset differs from preregistration: "
            f"observed={three_channel}"
        )
    missing = []
    for record in expected:
        for suffix in REQUIRED_SUFFIXES:
            if not (root / f"{record}{suffix}").is_file():
                missing.append(f"{record}{suffix}")
    if missing:
        raise SystemExit("Incomplete Zymed LTSTDB files: " + ", ".join(missing[:12]))

    expected_fs = float(dataset["expected_sampling_frequency_hz"])
    expected_min_channels = int(dataset["expected_min_signal_channels"])
    stop = int(dataset["window"]["stop_sample_exclusive"])
    for record in expected:
        header = wfdb.rdheader(str(root / record))
        if int(header.n_sig) < expected_min_channels:
            raise SystemExit(
                f"{record}: expected at least {expected_min_channels} channels, got {header.n_sig}"
            )
        if not _sv._same_number(header.fs, expected_fs):
            raise SystemExit(f"{record}: expected {expected_fs} Hz, got {header.fs}")
        if int(header.sig_len) < stop:
            raise SystemExit(f"{record}: signal shorter than frozen 30-minute window")

    hashes = {"RECORDS": _sv.sha256_file(manifest)}
    for record in expected:
        for suffix in REQUIRED_SUFFIXES:
            name = f"{record}{suffix}"
            hashes[name] = _sv.sha256_file(root / name)
    return hashes


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
    window = dataset["window"]
    start = int(window["start_sample"])
    stop = int(window["stop_sample_exclusive"])
    expected_fs = float(dataset["expected_sampling_frequency_hz"])
    duration_minutes = float(window["duration_seconds"]) / 60.0

    # Leakage barrier: only candidate channels 0/1 are read before selection.
    rec = wfdb.rdrecord(
        str(root / record),
        sampfrom=start,
        sampto=stop,
        channels=[0, 1],
        physical=True,
    )
    if not _sv._same_number(rec.fs, expected_fs):
        raise ValueError(f"unexpected sampling frequency {rec.fs}")
    signals = np.asarray(rec.p_signal, dtype=float)
    if signals.shape != (stop - start, 2):
        raise ValueError(f"invalid frozen signal geometry {signals.shape}")
    if not np.isfinite(signals).all():
        raise ValueError("signal contains non-finite values")

    detections = []
    qualities = []
    rates = []
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
        rates.append(float(len(retained) / duration_minutes))

    selected = choose_two_lead_channel_v3(
        rates[0],
        rates[1],
        primary_rate_ceiling_bpm=float(selector["primary_retained_rate_ceiling_bpm"]),
        alternate_rate_floor_bpm=float(selector["alternate_retained_rate_floor_bpm"]),
        minimum_rate_ratio=float(selector["minimum_alternate_to_primary_rate_ratio"]),
    )

    # Reference is loaded only after the channel choice is final. WFDB rdann
    # sampto is inclusive, so use stop-1 to match signal support [start, stop).
    annotation = wfdb.rdann(
        str(root / record),
        str(detector["annotation_extension"]),
        sampfrom=start,
        sampto=stop - 1,
    )
    reference, audit = _sv._reference_from_annotation(
        annotation, frozenset(detector["beat_symbols"]), stop - start
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
        "loaded_signal_names": list(rec.sig_name or []),
        "channel0_retained_rate_bpm": rates[0],
        "channel1_retained_rate_bpm": rates[1],
        "channel0_quality": qualities[0].to_dict(),
        "channel1_quality": qualities[1].to_dict(),
        "reference_annotation_count": int(reference.size),
        "third_channel_loaded": False,
    })
    audit.update({
        "record": record,
        "lead_selected_before_annotation_load": True,
        "selected_channel": int(selected),
        "third_channel_loaded": False,
    })
    return result, payload, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltstdb-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--polarity-threshold-report", type=Path, required=True)
    parser.add_argument("--confirm-first-scored-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_first_scored_run:
        raise SystemExit("Refusing to score LTSTDB without --confirm-first-scored-run")
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Refusing to overwrite first-run artifact: {OUTPUT_PATH}")

    protocol = load_locked_protocol()
    records = list(protocol["dataset"]["records"])
    if len(records) != 18 or len(set(records)) != 18:
        raise SystemExit("LTSTDB Zymed protocol must contain exactly 18 unique records")
    selector = protocol["lead_selector"]
    if selector["version"] != LEAD_SELECTOR_V3_VERSION:
        raise SystemExit("Selector-v3 version mismatch")
    if not _sv._same_number(selector["primary_retained_rate_ceiling_bpm"], STARVATION_PRIMARY_RATE_BPM):
        raise SystemExit("Selector-v3 primary rate threshold mismatch")
    if not _sv._same_number(selector["alternate_retained_rate_floor_bpm"], STARVATION_ALTERNATE_RATE_BPM):
        raise SystemExit("Selector-v3 alternate rate threshold mismatch")
    if not _sv._same_number(selector["minimum_alternate_to_primary_rate_ratio"], STARVATION_RATE_RATIO):
        raise SystemExit("Selector-v3 rate ratio mismatch")
    if frozenset(protocol["detector"]["beat_symbols"]) != DEFAULT_BEAT_SYMBOLS:
        raise SystemExit("Frozen beat-symbol set no longer matches DEFAULT_BEAT_SYMBOLS")

    print("Preflight: verifying exact 18-record Zymed LTSTDB cohort...", flush=True)
    input_hashes = verify_dataset(args.ltstdb_dir, protocol)
    derivation, width_gate, v2_gate = _sv._load_polarity_threshold_report(
        args.polarity_threshold_report
    )
    detector = protocol["detector"]
    if derivation.get("schema") != detector["polarity_threshold_report_schema"]:
        raise SystemExit("Polarity derivation schema mismatch")
    if not _sv._same_number(v2_gate, detector["expected_v2_gate_confidence"]):
        raise SystemExit("Frozen v2 polarity gate mismatch")
    if not _sv._same_number(width_gate, detector["expected_width_override_confidence"]):
        raise SystemExit("Frozen width gate mismatch")
    model = CandidateSuppressor.load(args.model)
    _sv.verify_model(model, protocol)

    results = []
    payloads = []
    audits = []
    selected_channels = Counter()
    for index, record in enumerate(records, start=1):
        try:
            result, payload, audit = evaluate_record(
                args.ltstdb_dir, record, protocol=protocol, model=model,
                v2_gate=v2_gate, width_gate=width_gate
            )
        except Exception as exc:
            raise SystemExit(
                f"Prospective LTSTDB v3 run aborted on record {record}; no partial "
                f"primary result is valid. {type(exc).__name__}: {exc}"
            ) from exc
        results.append(result)
        payloads.append(payload)
        audits.append(audit)
        selected_channels[str(payload["selected_channel"])] += 1
        print(f"[{index:02d}/18] {record}: scored", flush=True)

    if len(results) != 18:
        raise SystemExit("Prospective LTSTDB report requires exactly 18 records")
    summary = summarize_records(results)
    f1s = np.asarray([result.metrics.f1 for result in results], dtype=float)
    summary["macro_mean_record_f1"] = float(np.mean(f1s))
    summary["macro_median_record_f1"] = float(np.median(f1s))
    summary["min_record_f1"] = float(np.min(f1s))
    summary["max_record_f1"] = float(np.max(f1s))

    sidecar = Path(str(args.model) + ".json")
    if not sidecar.is_file():
        raise SystemExit(f"Missing model sidecar: {sidecar}")
    report = {
        "schema": "electrotrace.external_ltstdb_zymed_selector_v3_validation/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "prospective_external_selector_v3_evaluation_first_run",
        "dataset_exposure_after_this_run": "exposed_do_not_reuse_as_prospective_for_changed_selector",
        "protocol": protocol,
        "protocol_file": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "git_blob_sha": EXPECTED_PROTOCOL_GIT_BLOB_SHA,
            "sha256": _sv.sha256_file(PROTOCOL_PATH),
        },
        "evaluation_integrity": {
            "all_preregistered_records_scored": True,
            "records_scored": 18,
            "records_skipped": 0,
            "lead_selection_used_reference_annotations": False,
            "reference_annotations_loaded_after_lead_selection": True,
            "third_channel_loaded": False,
            "ltstdb_used_for_model_selection": False,
            "ltstdb_used_for_threshold_selection": False,
            "ltstdb_used_for_lead_selector_tuning": False,
            "retraining": False,
            "model_threshold_overridden": False,
            "detector_protocol_overridden": False,
            "first_scored_run_acknowledged": True,
        },
        "git_head": _sv.git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "package_versions": _sv.package_versions(),
        "model": {
            "path": str(args.model),
            "sha256": _sv.sha256_file(args.model),
            "sidecar_path": str(sidecar),
            "sidecar_sha256": _sv.sha256_file(sidecar),
            "metadata": model.metadata.to_dict(),
        },
        "polarity_threshold_derivation": {
            "path": str(args.polarity_threshold_report),
            "sha256": _sv.sha256_file(args.polarity_threshold_report),
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
            "This is the first prospective external evaluation of starvation-rescue "
            "selector v3 on the untouched Zymed three-channel LTSTDB subset. Only "
            "channels 0 and 1 were available to the selector; LTSTDB is exposed after "
            "this run."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n===== PROSPECTIVE ZYMED LTSTDB SELECTOR-V3 SUMMARY =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Selected channels:", dict(sorted(selected_channels.items())))
    print("Written:", OUTPUT_PATH)
    print("Zymed LTSTDB subset is now exposed; do not retune and reuse it as prospective.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
