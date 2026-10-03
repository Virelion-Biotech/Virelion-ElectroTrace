#!/usr/bin/env python3
"""QT Database QRS-boundary tolerance-curve validation.

The primary endpoint isolates delineation from event detection by supplying the
frozen signal-only delineator with the midpoint of each manual q1c QRS boundary
pair. A secondary end-to-end analysis uses a frozen signal detector and then
scores boundaries only after one-to-one event matching. The 11 q2c records are
reserved for an inter-observer agreement analysis.

No QTDB annotation is used to tune delineator or detector parameters.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace.qrs_delineation import (
    DEFAULT_AMPLITUDE_THRESHOLD,
    DEFAULT_ENVELOPE_THRESHOLD,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    DEFAULT_SEARCH_S,
    DEFAULT_SMOOTH_S,
    DEFAULT_SUSTAINED_SAMPLES,
    DELINEATOR_VERSION,
    delineate_qrs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "validation_protocols" / "qtdb_qrs_delineation_v1.json"
EXPECTED_SCHEMA = "electrotrace.qtdb_qrs_delineation_protocol/v1"
REPORT_SCHEMA = "electrotrace.qtdb_qrs_delineation_tolerance_curve/v1"


def require_wfdb():
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "QTDB validation requires the official wfdb Python package; "
            "no fallback parser is permitted."
        ) from exc
    return wfdb


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:
        raise RuntimeError("QTDB validation requires a Git checkout") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _same_number(actual, expected, *, atol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def load_locked_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError("Unexpected QTDB protocol schema")
    if protocol["delineator"]["version"] != DELINEATOR_VERSION:
        raise RuntimeError("QTDB protocol delineator version does not match code")

    expected = protocol["delineator"]["parameters"]
    actual = {
        "search_s": DEFAULT_SEARCH_S,
        "low_hz": DEFAULT_LOW_HZ,
        "high_hz": DEFAULT_HIGH_HZ,
        "smooth_s": DEFAULT_SMOOTH_S,
        "envelope_threshold": DEFAULT_ENVELOPE_THRESHOLD,
        "amplitude_threshold": DEFAULT_AMPLITUDE_THRESHOLD,
        "sustained_samples": DEFAULT_SUSTAINED_SAMPLES,
    }
    for key, value in actual.items():
        if isinstance(value, int):
            same = int(expected[key]) == value
        else:
            same = _same_number(expected[key], value)
        if not same:
            raise RuntimeError(f"QTDB protocol/code delineator drift: {key}")

    primary_tolerances = list(protocol["primary_endpoint"]["tolerances_ms"])
    secondary_tolerances = list(protocol["secondary_end_to_end"]["tolerances_ms"])
    interobserver_tolerances = list(
        protocol["confirmatory_interobserver"]["tolerances_ms"]
    )
    if primary_tolerances != [20, 40, 60, 80, 100]:
        raise RuntimeError("Primary QTDB tolerance grid drifted")
    if secondary_tolerances != primary_tolerances:
        raise RuntimeError("Secondary QTDB tolerance grid drifted")
    if interobserver_tolerances != primary_tolerances:
        raise RuntimeError("Inter-observer QTDB tolerance grid drifted")
    return protocol


def load_detector(spec: str):
    module_name, function_name = str(spec).split(":", 1)
    detector = getattr(importlib.import_module(module_name), function_name)
    if not callable(detector):
        raise TypeError(f"Detector {spec!r} is not callable")
    return detector


def load_record_names(dataset_dir: Path, protocol: dict) -> tuple[list[str], list[str]]:
    records_file = dataset_dir / "RECORDS"
    if not records_file.is_file():
        raise FileNotFoundError(records_file)
    records = [
        line.strip()
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_records = int(protocol["dataset"]["expected_records"])
    if len(records) != expected_records or len(set(records)) != expected_records:
        raise RuntimeError(
            f"QTDB RECORDS must contain exactly {expected_records} unique records"
        )

    primary_ext = str(protocol["dataset"]["primary_manual_annotation"])
    second_ext = str(protocol["dataset"]["interobserver_manual_annotation"])
    required_primary = int(protocol["dataset"]["expected_primary_manual_records"])
    expected_second = int(protocol["dataset"]["expected_interobserver_records"])

    missing: list[str] = []
    q1_count = 0
    q2_records: list[str] = []
    for record in records:
        for suffix in ("hea", "dat"):
            path = dataset_dir / f"{record}.{suffix}"
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(path.name)
        q1 = dataset_dir / f"{record}.{primary_ext}"
        if q1.is_file() and q1.stat().st_size > 0:
            q1_count += 1
        else:
            missing.append(q1.name)
        q2 = dataset_dir / f"{record}.{second_ext}"
        if q2.is_file() and q2.stat().st_size > 0:
            q2_records.append(record)

    if missing:
        raise RuntimeError("QTDB required inputs missing: " + ", ".join(missing))
    if q1_count != required_primary:
        raise RuntimeError(
            f"Expected {required_primary} primary manual files, found {q1_count}"
        )
    if len(q2_records) != expected_second:
        raise RuntimeError(
            f"Expected {expected_second} q2c records, found {len(q2_records)}"
        )
    return records, q2_records


def qrs_reference(wfdb, record_path: Path, extension: str) -> tuple[np.ndarray, np.ndarray]:
    """Load strictly paired manual QRS onset/offset boundaries.

    WFDB WFON/WFOFF annotations are '(' and ')'. For QTDB manual waveform
    annotations, num=1 designates QRS complexes.
    """
    ann = wfdb.rdann(str(record_path), extension=extension)
    pairs: list[tuple[int, int]] = []
    open_onset: int | None = None

    for sample, symbol, num in zip(ann.sample, ann.symbol, ann.num):
        if int(num) != 1 or symbol not in {"(", ")"}:
            continue
        sample = int(sample)
        if sample < 0:
            raise ValueError(f"{record_path.name}.{extension}: negative boundary")
        if symbol == "(":
            if open_onset is not None:
                raise ValueError(
                    f"{record_path.name}.{extension}: duplicate/unmatched QRS onset"
                )
            open_onset = sample
        else:
            if open_onset is None:
                raise ValueError(
                    f"{record_path.name}.{extension}: QRS offset without onset"
                )
            if sample <= open_onset:
                raise ValueError(
                    f"{record_path.name}.{extension}: non-positive QRS duration"
                )
            pairs.append((open_onset, sample))
            open_onset = None

    if open_onset is not None:
        raise ValueError(f"{record_path.name}.{extension}: unmatched final QRS onset")
    if not pairs:
        raise ValueError(f"{record_path.name}.{extension}: no manual QRS pairs")

    onset = np.asarray([x[0] for x in pairs], dtype=np.int64)
    offset = np.asarray([x[1] for x in pairs], dtype=np.int64)
    if np.any(np.diff(onset) <= 0) or np.any(np.diff(offset) <= 0):
        raise ValueError(f"{record_path.name}.{extension}: non-monotonic QRS pairs")
    centers = ((onset + offset) // 2).astype(np.int64)
    if np.any(np.diff(centers) <= 0):
        raise ValueError(f"{record_path.name}.{extension}: duplicate QRS centers")
    return onset, offset


def validate_detected(samples, signal_length: int, record: str) -> np.ndarray:
    raw = np.asarray(samples)
    if raw.ndim != 1:
        raise ValueError(f"{record}: detector output must be one-dimensional")
    if raw.size == 0:
        return np.asarray([], dtype=np.int64)
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{record}: detector output must contain finite integers")
    detected = numeric.astype(np.int64)
    if np.any(detected < 0) or np.any(detected >= int(signal_length)):
        raise ValueError(f"{record}: detector output outside signal support")
    if np.any(np.diff(detected) <= 0):
        raise ValueError(f"{record}: detector output must be strictly increasing")
    return detected


def match_indices(
    detected: np.ndarray,
    reference: np.ndarray,
    fs_hz: float,
    tolerance_ms: float,
) -> list[tuple[int, int]]:
    tolerance_samples = float(tolerance_ms) * float(fs_hz) / 1000.0
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(detected) and j < len(reference):
        delta = int(detected[i]) - int(reference[j])
        if abs(delta) <= tolerance_samples:
            pairs.append((i, j))
            i += 1
            j += 1
        elif detected[i] < reference[j]:
            i += 1
        else:
            j += 1
    return pairs


def error_summary(errors: np.ndarray) -> dict:
    x = np.asarray(errors, dtype=float)
    x = x[np.isfinite(x)]
    absolute = np.abs(x)
    return {
        "n": int(x.size),
        "mean_signed_ms": float(np.mean(x)) if x.size else None,
        "median_signed_ms": float(np.median(x)) if x.size else None,
        "sd_ms": float(np.std(x, ddof=1)) if x.size > 1 else None,
        "mean_absolute_ms": float(np.mean(absolute)) if x.size else None,
        "median_absolute_ms": float(np.median(absolute)) if x.size else None,
        "p95_absolute_ms": float(np.percentile(absolute, 95)) if x.size else None,
        "max_absolute_ms": float(np.max(absolute)) if x.size else None,
    }


def delineation_errors(
    signal: np.ndarray,
    fs_hz: float,
    centers: np.ndarray,
    reference_onset: np.ndarray,
    reference_offset: np.ndarray,
) -> dict:
    if not (
        len(centers) == len(reference_onset) == len(reference_offset)
    ):
        raise ValueError("Delineation centers and reference boundaries must align")

    onset_errors: list[float] = []
    offset_errors: list[float] = []
    onset_found: list[bool] = []
    offset_found: list[bool] = []
    for center, ref_on, ref_off in zip(centers, reference_onset, reference_offset):
        boundary = delineate_qrs(signal, fs_hz, int(center))
        onset_errors.append((int(boundary.onset) - int(ref_on)) * 1000.0 / fs_hz)
        offset_errors.append((int(boundary.offset) - int(ref_off)) * 1000.0 / fs_hz)
        onset_found.append(bool(boundary.onset_found))
        offset_found.append(bool(boundary.offset_found))

    return {
        "onset_errors_ms": np.asarray(onset_errors, dtype=float),
        "offset_errors_ms": np.asarray(offset_errors, dtype=float),
        "onset_found": np.asarray(onset_found, dtype=bool),
        "offset_found": np.asarray(offset_found, dtype=bool),
    }


def tolerance_counts(
    onset_errors_ms: np.ndarray,
    offset_errors_ms: np.ndarray,
    onset_found: np.ndarray,
    offset_found: np.ndarray,
    tolerances_ms: list[float],
) -> dict[str, dict]:
    onset_errors = np.asarray(onset_errors_ms, dtype=float)
    offset_errors = np.asarray(offset_errors_ms, dtype=float)
    onset_found = np.asarray(onset_found, dtype=bool)
    offset_found = np.asarray(offset_found, dtype=bool)
    n = len(onset_errors)
    if not (
        len(offset_errors) == len(onset_found) == len(offset_found) == n
    ):
        raise ValueError("Boundary error arrays must have equal lengths")

    out: dict[str, dict] = {}
    for tolerance in tolerances_ms:
        tol = float(tolerance)
        onset_ok = onset_found & (np.abs(onset_errors) <= tol)
        offset_ok = offset_found & (np.abs(offset_errors) <= tol)
        both_ok = onset_ok & offset_ok
        out[f"{tol:g}"] = {
            "eligible": int(n),
            "onset_success": int(np.count_nonzero(onset_ok)),
            "offset_success": int(np.count_nonzero(offset_ok)),
            "both_success": int(np.count_nonzero(both_ok)),
        }
    return out


def _bootstrap_indices(n_records: int, replicates: int, seed: int) -> np.ndarray:
    if n_records < 1:
        raise ValueError("Bootstrap requires at least one record")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n_records, size=(int(replicates), n_records))


def _ratio_summary(
    numerators: np.ndarray,
    denominators: np.ndarray,
    bootstrap_indices: np.ndarray,
    interval: tuple[float, float],
) -> dict:
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    total_den = float(np.sum(denominators))
    estimate = float(np.sum(numerators) / total_den) if total_den > 0 else None

    boot_num = np.sum(numerators[bootstrap_indices], axis=1)
    boot_den = np.sum(denominators[bootstrap_indices], axis=1)
    valid = boot_den > 0
    boot = np.divide(
        boot_num[valid],
        boot_den[valid],
        out=np.zeros(np.count_nonzero(valid), dtype=float),
        where=boot_den[valid] > 0,
    )
    if boot.size:
        ci = [
            float(np.percentile(boot, interval[0])),
            float(np.percentile(boot, interval[1])),
        ]
    else:
        ci = [None, None]
    return {
        "numerator": int(np.sum(numerators)),
        "denominator": int(np.sum(denominators)),
        "estimate": estimate,
        "record_bootstrap_interval": ci,
    }


def aggregate_primary_curve(
    records: list[dict],
    tolerances_ms: list[float],
    *,
    replicates: int,
    seed: int,
    interval: tuple[float, float],
) -> dict:
    indices = _bootstrap_indices(len(records), replicates, seed)
    curve: dict[str, dict] = {}
    for tolerance in tolerances_ms:
        key = f"{float(tolerance):g}"
        eligible = np.asarray(
            [r["primary"]["tolerance_counts"][key]["eligible"] for r in records],
            dtype=float,
        )
        curve[key] = {}
        for criterion in ("onset", "offset", "both"):
            success = np.asarray(
                [
                    r["primary"]["tolerance_counts"][key][f"{criterion}_success"]
                    for r in records
                ],
                dtype=float,
            )
            curve[key][criterion] = _ratio_summary(
                success, eligible, indices, interval
            )
    return curve


def aggregate_secondary(
    records: list[dict],
    tolerances_ms: list[float],
    *,
    replicates: int,
    seed: int,
    interval: tuple[float, float],
) -> dict:
    indices = _bootstrap_indices(len(records), replicates, seed)
    refs = np.asarray([r["secondary"]["reference_qrs"] for r in records], dtype=float)
    dets = np.asarray([r["secondary"]["detected_qrs"] for r in records], dtype=float)
    matched = np.asarray([r["secondary"]["matched_qrs"] for r in records], dtype=float)

    event = {
        "sensitivity": _ratio_summary(matched, refs, indices, interval),
        "positive_predictive_value": _ratio_summary(matched, dets, indices, interval),
    }
    curve: dict[str, dict] = {}
    for tolerance in tolerances_ms:
        key = f"{float(tolerance):g}"
        curve[key] = {}
        for criterion in ("onset", "offset", "both"):
            success = np.asarray(
                [
                    r["secondary"]["tolerance_counts"][key][f"{criterion}_success"]
                    for r in records
                ],
                dtype=float,
            )
            curve[key][criterion] = {
                "conditional_success_fraction": _ratio_summary(
                    success, matched, indices, interval
                ),
                "end_to_end_sensitivity": _ratio_summary(
                    success, refs, indices, interval
                ),
                "end_to_end_positive_predictive_value": _ratio_summary(
                    success, dets, indices, interval
                ),
            }
    return {"event_detection": event, "boundary_tolerance_curve": curve}


def aggregate_interobserver(
    records: list[dict],
    tolerances_ms: list[float],
    *,
    replicates: int,
    seed: int,
    interval: tuple[float, float],
) -> dict:
    if not records:
        raise ValueError("Inter-observer analysis requires q2c records")
    indices = _bootstrap_indices(len(records), replicates, seed)
    paired = np.asarray(
        [r["interobserver"]["paired_qrs"] for r in records], dtype=float
    )
    q1 = np.asarray([r["interobserver"]["q1c_qrs"] for r in records], dtype=float)
    q2 = np.asarray([r["interobserver"]["q2c_qrs"] for r in records], dtype=float)

    curve: dict[str, dict] = {}
    for tolerance in tolerances_ms:
        key = f"{float(tolerance):g}"
        curve[key] = {}
        for criterion in ("onset", "offset", "both"):
            success = np.asarray(
                [
                    r["interobserver"]["tolerance_counts"][key][
                        f"{criterion}_success"
                    ]
                    for r in records
                ],
                dtype=float,
            )
            curve[key][criterion] = _ratio_summary(
                success, paired, indices, interval
            )

    return {
        "paired_event_coverage_vs_q1c": _ratio_summary(
            paired, q1, indices, interval
        ),
        "paired_event_coverage_vs_q2c": _ratio_summary(
            paired, q2, indices, interval
        ),
        "boundary_tolerance_curve": curve,
    }


def record_hashes(record_path: Path, *, q2c: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for suffix in ("hea", "dat", "q1c"):
        path = record_path.with_suffix(f".{suffix}")
        out[path.name] = sha256_file(path)
    if q2c:
        path = record_path.with_suffix(".q2c")
        out[path.name] = sha256_file(path)
    return out


def evaluate_record(
    wfdb,
    dataset_dir: Path,
    record_name: str,
    *,
    protocol: dict,
    detector,
    has_q2c: bool,
) -> dict:
    base = dataset_dir / record_name
    channel = int(protocol["dataset"]["signal_channel"])
    rec = wfdb.rdrecord(str(base), channels=[channel], physical=True)
    signal = np.asarray(rec.p_signal[:, 0], dtype=float)
    fs_hz = float(rec.fs)
    if signal.ndim != 1 or signal.size < 32 or not np.isfinite(signal).all():
        raise ValueError(f"{record_name}: invalid channel-{channel} signal")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(f"{record_name}: invalid sampling frequency")

    q1_ext = str(protocol["dataset"]["primary_manual_annotation"])
    q1_on, q1_off = qrs_reference(wfdb, base, q1_ext)
    if int(q1_off[-1]) >= len(signal):
        raise ValueError(f"{record_name}: q1c boundary outside signal support")
    q1_centers = ((q1_on + q1_off) // 2).astype(np.int64)

    tolerances = [
        float(x) for x in protocol["primary_endpoint"]["tolerances_ms"]
    ]
    primary_errors = delineation_errors(
        signal, fs_hz, q1_centers, q1_on, q1_off
    )
    primary_counts = tolerance_counts(
        primary_errors["onset_errors_ms"],
        primary_errors["offset_errors_ms"],
        primary_errors["onset_found"],
        primary_errors["offset_found"],
        tolerances,
    )
    primary = {
        "reference_qrs": int(len(q1_on)),
        "onset_found": int(np.count_nonzero(primary_errors["onset_found"])),
        "offset_found": int(np.count_nonzero(primary_errors["offset_found"])),
        "both_found": int(
            np.count_nonzero(
                primary_errors["onset_found"] & primary_errors["offset_found"]
            )
        ),
        "onset_error_found_only": error_summary(
            primary_errors["onset_errors_ms"][primary_errors["onset_found"]]
        ),
        "offset_error_found_only": error_summary(
            primary_errors["offset_errors_ms"][primary_errors["offset_found"]]
        ),
        "tolerance_counts": primary_counts,
    }

    detector_kwargs = dict(protocol["secondary_end_to_end"]["detector_kwargs"])
    detected = validate_detected(
        detector(signal, fs_hz, **detector_kwargs), len(signal), record_name
    )
    event_tolerance = float(
        protocol["secondary_end_to_end"]["event_match_tolerance_ms"]
    )
    matched_indices = match_indices(detected, q1_centers, fs_hz, event_tolerance)
    matched_det_centers = np.asarray(
        [detected[i] for i, _ in matched_indices], dtype=np.int64
    )
    matched_ref_on = np.asarray(
        [q1_on[j] for _, j in matched_indices], dtype=np.int64
    )
    matched_ref_off = np.asarray(
        [q1_off[j] for _, j in matched_indices], dtype=np.int64
    )
    secondary_errors = delineation_errors(
        signal,
        fs_hz,
        matched_det_centers,
        matched_ref_on,
        matched_ref_off,
    )
    secondary_counts = tolerance_counts(
        secondary_errors["onset_errors_ms"],
        secondary_errors["offset_errors_ms"],
        secondary_errors["onset_found"],
        secondary_errors["offset_found"],
        tolerances,
    )
    secondary = {
        "reference_qrs": int(len(q1_centers)),
        "detected_qrs": int(len(detected)),
        "matched_qrs": int(len(matched_indices)),
        "event_match_tolerance_ms": event_tolerance,
        "onset_found_among_matched": int(
            np.count_nonzero(secondary_errors["onset_found"])
        ),
        "offset_found_among_matched": int(
            np.count_nonzero(secondary_errors["offset_found"])
        ),
        "onset_error_found_only": error_summary(
            secondary_errors["onset_errors_ms"][secondary_errors["onset_found"]]
        ),
        "offset_error_found_only": error_summary(
            secondary_errors["offset_errors_ms"][secondary_errors["offset_found"]]
        ),
        "tolerance_counts": secondary_counts,
    }

    interobserver = None
    if has_q2c:
        q2_ext = str(protocol["dataset"]["interobserver_manual_annotation"])
        q2_on, q2_off = qrs_reference(wfdb, base, q2_ext)
        if int(q2_off[-1]) >= len(signal):
            raise ValueError(f"{record_name}: q2c boundary outside signal support")
        q2_centers = ((q2_on + q2_off) // 2).astype(np.int64)
        observer_tolerance = float(
            protocol["confirmatory_interobserver"]["event_match_tolerance_ms"]
        )
        observer_pairs = match_indices(
            q2_centers, q1_centers, fs_hz, observer_tolerance
        )
        q2_on_paired = np.asarray(
            [q2_on[i] for i, _ in observer_pairs], dtype=np.int64
        )
        q2_off_paired = np.asarray(
            [q2_off[i] for i, _ in observer_pairs], dtype=np.int64
        )
        q1_on_paired = np.asarray(
            [q1_on[j] for _, j in observer_pairs], dtype=np.int64
        )
        q1_off_paired = np.asarray(
            [q1_off[j] for _, j in observer_pairs], dtype=np.int64
        )
        onset_errors = (q2_on_paired - q1_on_paired) * 1000.0 / fs_hz
        offset_errors = (q2_off_paired - q1_off_paired) * 1000.0 / fs_hz
        found = np.ones(len(observer_pairs), dtype=bool)
        interobserver = {
            "q1c_qrs": int(len(q1_on)),
            "q2c_qrs": int(len(q2_on)),
            "paired_qrs": int(len(observer_pairs)),
            "event_match_tolerance_ms": observer_tolerance,
            "onset_error_q2c_minus_q1c": error_summary(onset_errors),
            "offset_error_q2c_minus_q1c": error_summary(offset_errors),
            "tolerance_counts": tolerance_counts(
                onset_errors, offset_errors, found, found, tolerances
            ),
        }

    return {
        "record": record_name,
        "fs_hz": fs_hz,
        "signal_samples": int(len(signal)),
        "primary": primary,
        "secondary": secondary,
        "interobserver": interobserver,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_locked_protocol()
    wfdb = require_wfdb()
    records, q2_records = load_record_names(args.qtdb_dir, protocol)
    q2_set = set(q2_records)

    detector_spec = str(protocol["secondary_end_to_end"]["detector"])
    detector = load_detector(detector_spec)
    bootstrap = protocol["uncertainty"]
    replicates = int(bootstrap["bootstrap_replicates"])
    seed = int(bootstrap["seed"])
    interval = tuple(float(x) for x in bootstrap["interval_percent"])
    tolerances = [
        float(x) for x in protocol["primary_endpoint"]["tolerances_ms"]
    ]

    per_record: list[dict] = []
    hashes: dict[str, str] = {
        "RECORDS": sha256_file(args.qtdb_dir / "RECORDS")
    }
    for i, record in enumerate(records, start=1):
        result = evaluate_record(
            wfdb,
            args.qtdb_dir,
            record,
            protocol=protocol,
            detector=detector,
            has_q2c=record in q2_set,
        )
        per_record.append(result)
        hashes.update(
            record_hashes(args.qtdb_dir / record, q2c=record in q2_set)
        )
        print(
            f"[{i:03d}/{len(records)}] {record}: "
            f"q1c={result['primary']['reference_qrs']} "
            f"detected={result['secondary']['detected_qrs']} "
            f"matched={result['secondary']['matched_qrs']}",
            flush=True,
        )

    interobserver_records = [
        r for r in per_record if r["interobserver"] is not None
    ]
    if len(interobserver_records) != int(
        protocol["dataset"]["expected_interobserver_records"]
    ):
        raise RuntimeError("Inter-observer record count drift after evaluation")

    primary_curve = aggregate_primary_curve(
        per_record,
        tolerances,
        replicates=replicates,
        seed=seed,
        interval=interval,
    )
    secondary = aggregate_secondary(
        per_record,
        tolerances,
        replicates=replicates,
        seed=seed,
        interval=interval,
    )
    interobserver = aggregate_interobserver(
        interobserver_records,
        tolerances,
        replicates=replicates,
        seed=seed,
        interval=interval,
    )

    primary_onset_found = sum(r["primary"]["onset_found"] for r in per_record)
    primary_offset_found = sum(r["primary"]["offset_found"] for r in per_record)
    primary_reference = sum(r["primary"]["reference_qrs"] for r in per_record)

    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": protocol["evidence_status"],
        "git_head": git_head(),
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(PROTOCOL_PATH),
            "protocol_id": protocol["protocol_id"],
        },
        "runtime": {
            "wfdb_python_version": getattr(wfdb, "__version__", "unknown"),
            "numpy_version": np.__version__,
        },
        "dataset": {
            "name": protocol["dataset"]["name"],
            "version": protocol["dataset"]["version"],
            "records_requested": len(records),
            "records_completed": len(per_record),
            "records_excluded": [],
            "primary_manual_annotation": protocol["dataset"][
                "primary_manual_annotation"
            ],
            "primary_manual_records": len(per_record),
            "interobserver_manual_annotation": protocol["dataset"][
                "interobserver_manual_annotation"
            ],
            "interobserver_records": q2_records,
            "signal_channel": protocol["dataset"]["signal_channel"],
        },
        "delineator": protocol["delineator"],
        "primary_endpoint": {
            "label": "primary",
            "name": protocol["primary_endpoint"]["name"],
            "description": protocol["primary_endpoint"]["description"],
            "reference_qrs": int(primary_reference),
            "onset_found": int(primary_onset_found),
            "offset_found": int(primary_offset_found),
            "tolerance_curve": primary_curve,
        },
        "secondary_end_to_end": {
            "label": "secondary",
            "detector": detector_spec,
            "detector_kwargs": protocol["secondary_end_to_end"][
                "detector_kwargs"
            ],
            "event_match_tolerance_ms": protocol["secondary_end_to_end"][
                "event_match_tolerance_ms"
            ],
            **secondary,
        },
        "confirmatory_interobserver": {
            "label": "confirmatory",
            "annotators": protocol["confirmatory_interobserver"]["annotators"],
            "records": len(interobserver_records),
            **interobserver,
        },
        "uncertainty": {
            "method": "percentile record-bootstrap of ratio estimands",
            "bootstrap_unit": "record",
            "bootstrap_replicates": replicates,
            "seed": seed,
            "interval_percent": list(interval),
        },
        "input_file_sha256": dict(sorted(hashes.items())),
        "record_results": per_record,
        "nonclaims": protocol["nonclaims"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Primary tolerance curve:")
    print(json.dumps(primary_curve, indent=2, sort_keys=True))
    print("Secondary event detection:")
    print(json.dumps(secondary["event_detection"], indent=2, sort_keys=True))
    print("Wrote:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
