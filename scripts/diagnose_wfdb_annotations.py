#!/usr/bin/env python3
"""Why do some records' reference annotations fail validation?

Seven of 75 INCART records (I04, I17, I35, I44, I57, I72, I74) were dropped by
every validation script with "negative annotation sample index".  Before any
repair policy is trusted, look at what is actually in those files.

For each record this prints/records:
  * the raw annotation dump (first entries, where the negative indices sit, symbol
    counts, how many steps are non-increasing);
  * the audit under the historical policy (``error``);
  * the audit under ``drop_edges``, which drops only invalid *edge* annotations and
    accepts the result only if an independent detector (Pan-Tompkins here) agrees
    with what remains;
  * the alignment coverage of the valid remainder even when no repair is possible,
    so a shifted reference is visible as low coverage.

Usage:
  python scripts/diagnose_wfdb_annotations.py \\
      --data-dir .cache/physionet/incartdb --records I04 I17 I35 I44 I57 I72 I74

  # or audit everything and only report records with a problem
  python scripts/diagnose_wfdb_annotations.py --data-dir .cache/physionet/incartdb --only-invalid

Read the output before switching any script to ``--annotation-policy drop_edges``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace import __version__
from electrotrace.baseline_detectors import pan_tompkins_r_peaks
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS
from electrotrace.wfdb_records import (
    RecordExcluded,
    clean_reference_annotations,
    estimate_alignment,
    raw_annotation_dump,
)


def diagnose_record(base: Path, *, tolerance_ms: float) -> dict:
    import wfdb

    name = base.name
    rec = wfdb.rdrecord(str(base), channels=[0], physical=False)
    signal = np.asarray(rec.p_signal[:, 0] if rec.p_signal is not None else rec.d_signal[:, 0], dtype=float)
    fs = float(rec.fs)
    ann = wfdb.rdann(str(base), "atr")
    peaks = pan_tompkins_r_peaks(signal, fs)

    out = {
        "record": name,
        "fs_hz": fs,
        "signal_length": int(signal.size),
        "raw": raw_annotation_dump(base),
        "policies": {},
    }
    for policy in ("error", "drop_edges"):
        try:
            _, audit = clean_reference_annotations(
                ann.sample,
                ann.symbol,
                n_samples=signal.size,
                fs_hz=fs,
                record=name,
                beat_symbols=DEFAULT_BEAT_SYMBOLS,
                policy=policy,
                alignment_peaks=peaks,
                tolerance_ms=tolerance_ms,
            )
        except RecordExcluded as exc:
            audit = exc.audit
        out["policies"][policy] = audit.to_dict()

    sample = np.asarray(ann.sample, dtype=np.int64)
    is_beat = np.array([s in DEFAULT_BEAT_SYMBOLS for s in ann.symbol], dtype=bool)
    valid = sample[is_beat & (sample >= 0) & (sample < signal.size)]
    valid = valid[np.concatenate([[True], np.diff(valid) > 0])] if valid.size else valid
    coverage, offset = estimate_alignment(peaks, valid, fs, tolerance_ms=tolerance_ms)
    out["valid_remainder_alignment"] = {
        "n_valid_beats": int(valid.size),
        "coverage_vs_pan_tompkins": coverage,
        "median_offset_ms": offset,
        "note": "coverage near 1 = aligned; about 2*tolerance/RR (roughly 0.2) = shifted or unrelated",
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--records", nargs="*", default=None, help="Defaults to every local record.")
    parser.add_argument("--only-invalid", action="store_true", help="Report only records with invalid annotations.")
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    names = args.records or sorted(p.stem for p in args.data_dir.glob("*.hea"))
    if not names:
        raise SystemExit(f"No records found under {args.data_dir}")

    reports = []
    failures = []
    print(f"{'record':8} {'neg':>4} {'lead':>4} {'inter':>5} {'trail':>5} {'error':>9} {'drop_edges':>26} {'cover':>6}")
    for name in names:
        try:
            report = diagnose_record(args.data_dir / name, tolerance_ms=args.tolerance_ms)
        except Exception as exc:
            failures.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{name:8} FAILED: {exc}")
            continue
        err = report["policies"]["error"]
        drop = report["policies"]["drop_edges"]
        has_problem = err["action"] != "kept_unchanged" or drop["n_beyond_end_beat"] > 0
        if args.only_invalid and not has_problem:
            continue
        reports.append(report)
        cov = report["valid_remainder_alignment"]["coverage_vs_pan_tompkins"]
        print(
            f"{name:8} {err['n_negative_beat']:>4} {drop['n_leading_invalid']:>4} {drop['n_interior_invalid']:>5} "
            f"{drop['n_trailing_invalid']:>5} {err['action']:>9} {drop['action']:>26} {cov:>6.2f}"
        )
        if has_problem:
            raw = report["raw"]
            print(f"         first annotations: {raw['first'][:6]}")
            print(f"         negative positions: {raw['negative_positions'][:8]}  symbols: {raw['negative_symbols']}")
            if drop["reason"]:
                print(f"         drop_edges: {drop['reason']}")

    payload = {
        "schema": "electrotrace.wfdb_annotation_diagnostics/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tolerance_ms": args.tolerance_ms,
        "records": reports,
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("\nWritten:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
