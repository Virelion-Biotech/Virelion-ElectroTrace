#!/usr/bin/env python3
"""Resumable QTDB confirmatory validation with per-record checkpoints."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import importlib.util
import sys

from electrotrace.qrs_delineation import DELINEATOR_VERSION
from electrotrace.qtdb_detector_adapter import detect_r_peaks_adaptive

_spec = importlib.util.spec_from_file_location("validate_qtdb", Path(__file__).resolve().parent / "validate_qtdb.py")
_vq = importlib.util.module_from_spec(_spec)
sys.modules["validate_qtdb"] = _vq
_spec.loader.exec_module(_vq)
DATASET_VERSION = _vq.DATASET_VERSION
PRIMARY_EVENT_TOLERANCE_MS = _vq.PRIMARY_EVENT_TOLERANCE_MS
boundary_errors = _vq.boundary_errors
qrs_reference = _vq.qrs_reference
record_hashes = _vq.record_hashes
require_wfdb = _vq.require_wfdb
summarize = _vq.summarize


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--tolerance-ms", type=float, default=PRIMARY_EVENT_TOLERANCE_MS)
    parser.add_argument("--max-records", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    wfdb = require_wfdb()
    head = git_head()
    headers = sorted(args.dataset_dir.glob("*.hea"))
    if args.max_records > 0:
        headers = headers[: args.max_records]
    if not headers:
        raise RuntimeError("No QTDB WFDB headers found")

    if args.checkpoint.exists():
        state = json.loads(args.checkpoint.read_text())
        done = {r["record"] for r in state.get("records_detail", [])}
        print(f"resume: {len(done)} records already done", flush=True)
    else:
        state = {
            "status": "in_progress",
            "dataset": DATASET_VERSION,
            "channel": args.channel,
            "detector": "electrotrace.qtdb_detector_adapter:detect_r_peaks_adaptive",
            "event_tolerance_ms": args.tolerance_ms,
            "delineator_version": DELINEATOR_VERSION,
            "wfdb_version": getattr(wfdb, "__version__", "unknown"),
            "git_head": head,
            "input_file_sha256": {},
            "records_detail": [],
            "failures": {},
        }
        done = set()

    all_onset: list[float] = []
    all_offset: list[float] = []

    for i, header in enumerate(headers, 1):
        base = header.with_suffix("")
        name = base.name
        if name in done:
            continue
        print(f"[{i}/{len(headers)}] {name}", flush=True)
        try:
            record = wfdb.rdrecord(str(base), channels=[args.channel], physical=True)
            signal = np.asarray(record.p_signal[:, 0], dtype=float)
            if not np.isfinite(signal).all():
                raise ValueError("non-finite signal")
            ref_on, ref_off = qrs_reference(wfdb, base)
            detected = np.asarray(detect_r_peaks_adaptive(signal, float(record.fs)), dtype=int)
            if detected.ndim != 1 or (detected.size and np.any(np.diff(detected) <= 0)):
                raise ValueError("detector output must be strictly increasing")
            onset_err, offset_err, pairs = boundary_errors(
                signal, float(record.fs), detected, ref_on, ref_off, args.tolerance_ms
            )
            entry = {
                "record": name,
                "fs_hz": float(record.fs),
                "reference_qrs": int(len(ref_on)),
                "detected_qrs": int(len(detected)),
                "matched_qrs": int(len(pairs)),
                "onset": summarize(onset_err),
                "offset": summarize(offset_err),
                "onset_errors_ms": [float(x) for x in onset_err],
                "offset_errors_ms": [float(x) for x in offset_err],
            }
            state["input_file_sha256"].update(record_hashes(base))
            state["records_detail"].append(entry)
            done.add(name)
        except Exception as exc:
            state["failures"][name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {name}: {exc}", flush=True)

        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    for entry in state["records_detail"]:
        all_onset.extend(entry.get("onset_errors_ms", []))
        all_offset.extend(entry.get("offset_errors_ms", []))

    clean_detail = []
    for entry in state["records_detail"]:
        clean = {k: v for k, v in entry.items() if k not in ("onset_errors_ms", "offset_errors_ms")}
        clean_detail.append(clean)

    result = {
        "status": "complete" if not state["failures"] else "complete_with_failures",
        "dataset": DATASET_VERSION,
        "records": len(clean_detail),
        "records_requested": len(headers),
        "channel": args.channel,
        "detector": state["detector"],
        "event_tolerance_ms": args.tolerance_ms,
        "delineator_version": DELINEATOR_VERSION,
        "wfdb_version": state["wfdb_version"],
        "git_head": head,
        "input_file_sha256": dict(sorted(state["input_file_sha256"].items())),
        "failures": state["failures"],
        "summary": {
            "reference_qrs": int(sum(r["reference_qrs"] for r in clean_detail)),
            "detected_qrs": int(sum(r["detected_qrs"] for r in clean_detail)),
            "matched_qrs": int(sum(r["matched_qrs"] for r in clean_detail)),
            "onset": summarize(all_onset),
            "offset": summarize(all_offset),
        },
        "records_detail": clean_detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"status={result['status']} written={args.output}", flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
