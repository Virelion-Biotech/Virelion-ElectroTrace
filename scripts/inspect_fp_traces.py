#!/usr/bin/env python3
"""Eyeball actual traces for the "amplitude looks like a real beat" false positives.

incart_fp_offsets.py found that 34% of INCART false positives have amplitude at
least 60% of the matched true beat's height, and 40% of those are opposite
polarity to it. That is too large a fraction to wave off as small T-waves
without looking at a few real segments. This script pulls the actual ECG
window around a sample of those false positives -- plus the nearest true beats
-- so a person can decide whether they are large/inverted T-waves, notched or
split QRS complexes, or something else, before any recalibration or feature
work is scoped.

This is a manual-inspection aid. It draws a sample; it does not compute a new
metric. Look at the figure before drawing a conclusion.

Usage:
  python -u scripts/inspect_fp_traces.py \
      --incart-dir .cache/physionet/incartdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --records I03 I12 I29 I26 I05 --n-samples 12
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
from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.validation import DEFAULT_BEAT_SYMBOLS
from electrotrace.wfdb_records import POLICIES, RecordExcluded, load_annotated_record
from scripts.incart_fp_offsets import (
    _json_default,
    analyze_signal_record,
    git_head,
    package_versions,
    sha256_file,
)


def select_samples(
    pooled: list[dict],
    *,
    n_samples: int,
    seed: int,
    max_per_record: int | None = None,
) -> list[dict]:
    """Spread the sample across records rather than letting one record dominate."""
    if not pooled:
        return []
    rng = np.random.default_rng(seed)
    by_record: dict[str, list[dict]] = {}
    for row in pooled:
        by_record.setdefault(row["record"], []).append(row)
    for rows in by_record.values():
        rng.shuffle(rows)

    per_record_cap = max_per_record or max(1, -(-n_samples // max(1, len(by_record))))
    selected: list[dict] = []
    records = list(by_record)
    rng.shuffle(records)
    cursor = {name: 0 for name in records}
    while len(selected) < n_samples and any(cursor[name] < len(by_record[name]) for name in records):
        for name in records:
            if len(selected) >= n_samples:
                break
            taken_from_record = sum(1 for s in selected if s["record"] == name)
            if cursor[name] >= len(by_record[name]) or taken_from_record >= per_record_cap:
                continue
            selected.append(by_record[name][cursor[name]])
            cursor[name] += 1
    return selected


def build_pool(
    model: CandidateSuppressor,
    incart_dir: Path,
    records: list[str],
    *,
    scale_method: str,
    threshold: float | None,
    tolerance_ms: float,
    policy: str,
    amp_ratio_min: float,
    polarity_filter: str,
) -> tuple[list[dict], dict[str, tuple[np.ndarray, float, np.ndarray]], list[dict]]:
    """Run the forensics analysis per record and collect matching false positives.

    Returns (pooled_fp_rows, signals_by_record, skipped). signals_by_record
    keeps each record's signal/fs/reference in memory for plotting -- INCART
    records are small enough (a few MB each) that holding --n-samples-worth of
    distinct records is fine; this is a diagnostic tool, not a batch job.
    """
    pooled: list[dict] = []
    signals: dict[str, tuple[np.ndarray, float, np.ndarray]] = {}
    skipped: list[dict] = []

    for name in records:
        try:
            annotated = load_annotated_record(
                incart_dir / name,
                beat_symbols=DEFAULT_BEAT_SYMBOLS,
                tolerance_ms=tolerance_ms,
                policy=policy,
            )
        except RecordExcluded as exc:
            skipped.append({"record": name, "reason": exc.reason})
            continue

        op_threshold = float(model.metadata.threshold if threshold is None else threshold)
        result = analyze_signal_record(
            model,
            annotated.signal,
            annotated.fs_hz,
            annotated.reference,
            scale_method=scale_method,
            threshold=op_threshold,
            tolerance_ms=tolerance_ms,
        )
        signals[name] = (annotated.signal, annotated.fs_hz, annotated.reference)
        table = result["false_positive_table"]
        for k in range(len(table["sample"])):
            amp = table["amp_ratio"][k]
            if not np.isfinite(amp) or amp < amp_ratio_min:
                continue
            same = bool(table["same_polarity"][k])
            if polarity_filter == "same" and not same:
                continue
            if polarity_filter == "opposite" and same:
                continue
            pooled.append(
                {
                    "record": name,
                    "sample": int(table["sample"][k]),
                    "category": str(table["category"][k]),
                    "amp_ratio": float(amp),
                    "same_polarity": same,
                    "dt_prev_ms": float(table["dt_prev_ms"][k]),
                    "dt_next_ms": float(table["dt_next_ms"][k]),
                    "nearest_ref_index": int(table["nearest_ref_index"][k]),
                    "probability": float(table["probability"][k]),
                    "operating_threshold": op_threshold,
                }
            )
    return pooled, signals, skipped


def plot_samples(path: Path, samples: list[dict], signals: dict, *, window_s: float) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not samples:
        return False

    cols = 3
    rows = -(-len(samples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for i, s in enumerate(samples):
        ax = axes[i // cols][i % cols]
        signal, fs, reference = signals[s["record"]]
        half = int(round(window_s * fs))
        lo = max(0, s["sample"] - half)
        hi = min(len(signal), s["sample"] + half + 1)
        t = (np.arange(lo, hi) - s["sample"]) / fs
        ax.plot(t, signal[lo:hi], color="tab:blue", linewidth=0.9)
        nearby_ref = reference[(reference >= lo) & (reference < hi)]
        for r in nearby_ref:
            ax.axvline((r - s["sample"]) / fs, color="tab:green", linestyle="--", alpha=0.6)
        ax.axvline(0.0, color="tab:red", linestyle="-", alpha=0.8)
        ax.set_title(
            f"{s['record']} @ {s['sample']} | {s['category']}\n"
            f"amp={s['amp_ratio']:.2f} {'same' if s['same_polarity'] else 'OPPOSITE'} pol, "
            f"p={s['probability']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("s (0 = false positive)")
    for j in range(len(samples), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Red = false positive. Green dashed = reference beat(s) in window.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incart-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--records",
        nargs="*",
        default=["I03", "I12", "I29", "I26", "I05", "I14", "I47", "I30"],
        help="Defaults to the highest false-positive-count records from the 2026-09-22 forensics run.",
    )
    parser.add_argument("--scale-method", default="windowed_std")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--annotation-policy", choices=POLICIES, default="error")
    parser.add_argument("--amp-ratio-min", type=float, default=0.6)
    parser.add_argument("--polarity-filter", choices=["any", "same", "opposite"], default="any")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--max-per-record", type=int, default=None)
    parser.add_argument("--window-s", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation_reports/experiments/2026-09-incart-fp-forensics")
    )
    args = parser.parse_args()

    model = CandidateSuppressor.load(args.model)
    pooled, signals, skipped = build_pool(
        model,
        args.incart_dir,
        args.records,
        scale_method=args.scale_method,
        threshold=args.threshold,
        tolerance_ms=args.tolerance_ms,
        policy=args.annotation_policy,
        amp_ratio_min=args.amp_ratio_min,
        polarity_filter=args.polarity_filter,
    )
    print(f"{len(pooled)} candidate false positives match amp_ratio >= {args.amp_ratio_min} "
          f"(polarity filter: {args.polarity_filter}) across {len(args.records)} records "
          f"({len(skipped)} skipped)")
    if not pooled:
        raise SystemExit("No matching false positives found; nothing to plot.")

    samples = select_samples(pooled, n_samples=args.n_samples, seed=args.seed, max_per_record=args.max_per_record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "incart_fp_trace_samples.png"
    manifest_path = args.output_dir / "incart_fp_trace_samples.json"
    plotted = plot_samples(png_path, samples, signals, window_s=args.window_s)

    manifest = {
        "schema": "electrotrace.incart_fp_trace_samples/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Manual-inspection sample; not a metric. Look at the figure.",
        "git_head": git_head(),
        "software_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model": {"path": str(args.model), "sha256": sha256_file(args.model)},
        "protocol": {
            "records_scanned": args.records,
            "amp_ratio_min": args.amp_ratio_min,
            "polarity_filter": args.polarity_filter,
            "window_s": args.window_s,
            "scale_method": args.scale_method,
            "annotation_policy": args.annotation_policy,
            "n_candidates_found": len(pooled),
        },
        "samples": samples,
        "skipped_records": skipped,
        "figure": png_path.name if plotted else None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )

    print(f"Selected {len(samples)} samples across {len({s['record'] for s in samples})} records:")
    for s in samples:
        print(
            f"  {s['record']} @ {s['sample']} ({s['sample']/signals[s['record']][1]:.1f}s): "
            f"{s['category']}, amp={s['amp_ratio']:.2f}, "
            f"{'same' if s['same_polarity'] else 'OPPOSITE'} polarity, p={s['probability']:.2f}"
        )
    print("\nWritten:", manifest_path)
    if plotted:
        print("Figure:", png_path)
    else:
        print("matplotlib unavailable; figure not drawn. Inspect samples in the manifest instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
