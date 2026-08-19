"""Confirmatory QT Database waveform-boundary validation.

The confirmatory path requires the official ``wfdb`` package and refuses to
fall back to an internal parser. Manual annotations are used only after the
signal detector has produced candidate QRS centers.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib
import json
import subprocess

import numpy as np

from electrotrace.qrs_delineation import delineate_qrs, DELINEATOR_VERSION
from electrotrace.validation import match_peaks

DATASET_VERSION = "qtdb-1.0.0"
PRIMARY_EVENT_TOLERANCE_MS = 75.0


def require_wfdb():
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "Confirmatory QTDB validation requires the official wfdb package; "
            "no fallback parser is permitted."
        ) from exc
    return wfdb


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError("QTDB confirmation requires a Git checkout") from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_detector(spec: str):
    module_name, function_name = spec.split(":", 1)
    detector = getattr(importlib.import_module(module_name), function_name)
    if not callable(detector):
        raise TypeError(f"Detector {spec!r} is not callable")
    return detector


def qrs_reference(wfdb, record_path: Path, extension: str = "q1c") -> tuple[np.ndarray, np.ndarray]:
    ann = wfdb.rdann(str(record_path), extension=extension)
    onset: list[int] = []
    offset: list[int] = []
    for sample, symbol, num in zip(ann.sample, ann.symbol, ann.num):
        if int(num) != 1:
            continue
        if symbol == "(":
            onset.append(int(sample))
        elif symbol == ")":
            offset.append(int(sample))
    if len(onset) != len(offset):
        raise ValueError(f"{record_path.name}: unmatched QRS onset/offset annotations")
    if onset and (np.any(np.diff(onset) <= 0) or np.any(np.diff(offset) <= 0)):
        raise ValueError(f"{record_path.name}: non-monotonic QRS boundaries")
    if any(o >= f for o, f in zip(onset, offset)):
        raise ValueError(f"{record_path.name}: invalid QRS boundary pair")
    return np.asarray(onset, dtype=int), np.asarray(offset, dtype=int)


def record_hashes(record_path: Path) -> dict[str, str]:
    base = record_path.with_suffix("")
    files = {}
    for suffix in (".hea", ".dat", ".q1c"):
        path = base.with_suffix(suffix)
        if not path.exists():
            raise FileNotFoundError(path)
        files[path.name] = sha256(path)
    return files


def pair_matches(detected: np.ndarray, reference: np.ndarray, fs_hz: float, tolerance_ms: float):
    if len(detected):
        detected = np.asarray(detected, dtype=int)
    reference = np.asarray(reference, dtype=int)
    match_peaks(detected.tolist(), reference.tolist(), fs_hz, tolerance_ms=tolerance_ms)
    tolerance_samples = tolerance_ms * fs_hz / 1000.0
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(detected) and j < len(reference):
        delta = int(detected[i]) - int(reference[j])
        if abs(delta) <= tolerance_samples:
            pairs.append((int(detected[i]), int(reference[j])))
            i += 1
            j += 1
        elif detected[i] < reference[j]:
            i += 1
        else:
            j += 1
    return pairs


def boundary_errors(signal, fs_hz, detected_centers, ref_onset, ref_offset, tolerance_ms):
    ref_centers = ((ref_onset + ref_offset) // 2).astype(int)
    pairs = pair_matches(detected_centers, ref_centers, fs_hz, tolerance_ms)
    by_center = {int(c): delineate_qrs(signal, fs_hz, int(c)) for c in detected_centers}
    onset_err: list[float] = []
    offset_err: list[float] = []
    for det_center, ref_center in pairs:
        candidates = np.flatnonzero(ref_centers == ref_center)
        if candidates.size != 1:
            raise ValueError("Reference QRS centers must be unique")
        idx = int(candidates[0])
        boundary = by_center[det_center]
        onset_err.append((boundary.onset - int(ref_onset[idx])) * 1000.0 / fs_hz)
        offset_err.append((boundary.offset - int(ref_offset[idx])) * 1000.0 / fs_hz)
    return onset_err, offset_err, pairs


def summarize(errors: list[float]) -> dict:
    x = np.asarray(errors, dtype=float)
    a = np.abs(x)
    return {
        "n": int(len(x)),
        "mean_signed_ms": float(np.mean(x)) if len(x) else None,
        "median_signed_ms": float(np.median(x)) if len(x) else None,
        "sd_ms": float(np.std(x, ddof=1)) if len(x) > 1 else None,
        "mean_absolute_ms": float(np.mean(a)) if len(x) else None,
        "median_absolute_ms": float(np.median(a)) if len(x) else None,
        "p95_absolute_ms": float(np.percentile(a, 95)) if len(x) else None,
        "max_absolute_ms": float(np.max(a)) if len(x) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--detector", required=True, help="module:function returning detector sample indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--tolerance-ms", type=float, default=PRIMARY_EVENT_TOLERANCE_MS)
    args = parser.parse_args()

    if args.tolerance_ms <= 0:
        parser.error("--tolerance-ms must be positive")
    wfdb = require_wfdb()
    head = git_head()
    detector = load_detector(args.detector)
    records = sorted(args.dataset_dir.glob("*.hea"))
    if not records:
        raise RuntimeError("No QTDB WFDB headers found")

    per_record = []
    all_onset: list[float] = []
    all_offset: list[float] = []
    all_hashes: dict[str, str] = {}
    for header in records:
        base = header.with_suffix("")
        record = wfdb.rdrecord(str(base), channels=[args.channel], physical=True)
        signal = np.asarray(record.p_signal[:, 0], dtype=float)
        if not np.isfinite(signal).all():
            raise ValueError(f"{base.name}: non-finite signal returned by wfdb")
        ref_on, ref_off = qrs_reference(wfdb, base)
        raw_detected = detector(signal, float(record.fs))
        detected = np.asarray(raw_detected, dtype=int)
        if detected.ndim != 1 or (detected.size and np.any(np.diff(detected) <= 0)):
            raise ValueError(f"{base.name}: detector output must be strictly increasing")
        onset_err, offset_err, pairs = boundary_errors(
            signal, float(record.fs), detected, ref_on, ref_off, args.tolerance_ms
        )
        all_onset.extend(onset_err)
        all_offset.extend(offset_err)
        all_hashes.update(record_hashes(base))
        per_record.append({
            "record": base.name,
            "fs_hz": float(record.fs),
            "reference_qrs": int(len(ref_on)),
            "detected_qrs": int(len(detected)),
            "matched_qrs": int(len(pairs)),
            "onset": summarize(onset_err),
            "offset": summarize(offset_err),
        })

    result = {
        "status": "complete",
        "dataset": DATASET_VERSION,
        "records": len(per_record),
        "channel": args.channel,
        "detector": args.detector,
        "event_tolerance_ms": args.tolerance_ms,
        "delineator_version": DELINEATOR_VERSION,
        "wfdb_version": getattr(wfdb, "__version__", "unknown"),
        "git_head": head,
        "input_file_sha256": dict(sorted(all_hashes.items())),
        "summary": {
            "reference_qrs": int(sum(r["reference_qrs"] for r in per_record)),
            "detected_qrs": int(sum(r["detected_qrs"] for r in per_record)),
            "matched_qrs": int(sum(r["matched_qrs"] for r in per_record)),
            "onset": summarize(all_onset),
            "offset": summarize(all_offset),
        },
        "records_detail": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
