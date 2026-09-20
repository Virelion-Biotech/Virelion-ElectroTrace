#!/usr/bin/env python3
"""Evidence-first diagnostics for the dual-polarity merge experiment.

Runs, for every usable INCART record and with the SAME locked MIT-BIH model
(no retraining), the baseline adaptive detector plus a grid of merge variants:

  pooled_floor0_all          -> reproduces the first (failed) experiment
  per_stream_floor{f}_all    -> RR features per polarity stream (fix #1),
                                minority-stream baseline-blip floor f (fix #2)
  per_stream_floor{f}_gaps   -> opposite-polarity beats only inside long RR gaps
  per_stream_floor1_all_minor{p} -> + higher RF probability bar for the minority stream

and, for --detail-records (default I62), writes the instrumentation the
handoff asked for (direction E): candidate coverage per polarity, RF
probability histograms, and where merge-added false positives sit relative to
true beats (T-wave band vs Q/R/S band) and how high they rise above baseline.

The features/RF are evaluated once per (feature_scope, floor); variants that
only differ in the combine step reuse the scored candidates.

Usage (Colab, after the Drive cache is mounted / symlinked):
  python -u scripts/diagnose_merge_incart.py \
      --incart-dir .cache/physionet/incartdb \
      --model-path validation_reports/incart_mitbih_model_windowed_std_2026-09-09.pkl \
      --scale-method windowed_std --detail-records I62 I27 I31 I53 I64
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor
from electrotrace.validation import (
    DEFAULT_BEAT_SYMBOLS,
    _record_signal,
    load_reference_annotations,
    match_peaks,
)
from electrotrace.validation_detectors import (
    _combine_scored_streams,
    _score_dual_polarity_streams,
    detect_r_peaks_two_stage,
    estimate_stage1_scale,
    select_signal_polarity,
)

TOL_MS = 75.0
# Gaps-only follow-up: dense floor sweep + optional minority RF bars on gaps.
FLOORS = (0.0, 0.25, 0.5, 0.75, 1.0)
MINORITY_THRESHOLDS = (0.90, 0.95)  # applied only on gaps variants below
REGRESSION_DELTA = -0.02
IMPROVEMENT_DELTA = 0.02


# --------------------------------------------------------------------------- #
# variant definitions
# --------------------------------------------------------------------------- #
def variant_specs() -> dict[str, dict]:
    """Gaps-only grid (per-stream features). No *_all / pooled controls."""
    specs: dict[str, dict] = {}
    for f in FLOORS:
        specs[f"per_stream_floor{f:g}_gaps"] = dict(
            feature_scope="per_stream", floor=f, scope="gaps", minor=None
        )
    # Best prior gaps floor (1.0) + mild minority RF bar
    for m in MINORITY_THRESHOLDS:
        specs[f"per_stream_floor1_gaps_minor{m:g}"] = dict(
            feature_scope="per_stream", floor=1.0, scope="gaps", minor=m
        )
    # Floor 0.5 gaps + mild minority RF bar (middle of the earlier tradeoff)
    for m in MINORITY_THRESHOLDS:
        specs[f"per_stream_floor0.5_gaps_minor{m:g}"] = dict(
            feature_scope="per_stream", floor=0.5, scope="gaps", minor=m
        )
    return specs


# --------------------------------------------------------------------------- #
# per-record evaluation
# --------------------------------------------------------------------------- #
def _metrics(peaks: np.ndarray, refs: np.ndarray, fs: float) -> dict:
    m = match_peaks(np.asarray(peaks, dtype=int), refs, fs, tolerance_ms=TOL_MS)
    return {
        "f1": m.f1, "sens": m.sensitivity, "ppv": m.positive_predictive_value,
        "tp": m.true_positive, "fp": m.false_positive, "fn": m.false_negative, "n": m.detected_count,
    }


def _quantiles(values: np.ndarray) -> dict | None:
    if len(values) == 0:
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(len(values)), "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4])}


def _near(peaks: np.ndarray, refs: np.ndarray, tol: int) -> np.ndarray:
    if len(refs) == 0 or len(peaks) == 0:
        return np.zeros(len(peaks), dtype=bool)
    idx = np.searchsorted(refs, peaks)
    lo = refs[np.clip(idx - 1, 0, len(refs) - 1)]
    hi = refs[np.clip(idx, 0, len(refs) - 1)]
    return np.minimum(np.abs(peaks - lo), np.abs(peaks - hi)) <= tol


def detail_for_record(signal, fs, refs, model, adaptive_peaks, merge_peaks, scored_floor0, major_id, scale_method) -> dict:
    """Direction-E instrumentation for one record (see module docstring)."""
    tol = int(round(TOL_MS * fs / 1000.0))
    thr = float(model.metadata.threshold)
    z = signal - np.median(signal)
    scale = estimate_stage1_scale(z, fs, method=scale_method)
    peaks, prob, stream = scored_floor0["peaks"], scored_floor0["probabilities"], scored_floor0["stream"]
    is_major = stream == major_id

    out: dict = {
        "majority_stream": "positive" if major_id == 0 else "negative",
        "rf_threshold": thr,
        "candidates": {"majority": int(is_major.sum()), "minority": int((~is_major).sum())},
        "reference_count": int(len(refs)),
    }

    def coverage(mask):
        p = peaks[mask]
        q = prob[mask]
        covered = np.array([bool(np.any(np.abs(p - r) <= tol)) for r in refs]) if len(refs) else np.array([], bool)
        accepted = np.array([bool(np.any((np.abs(p - r) <= tol) & (q >= thr))) for r in refs]) if len(refs) else np.array([], bool)
        return covered, accepted

    cov_maj, acc_maj = coverage(is_major)
    cov_min, acc_min = coverage(~is_major)
    n_ref = max(len(refs), 1)
    out["reference_coverage"] = {
        "majority_has_candidate": float(cov_maj.mean()) if len(refs) else 0.0,
        "minority_has_candidate": float(cov_min.mean()) if len(refs) else 0.0,
        "either_has_candidate": float((cov_maj | cov_min).mean()) if len(refs) else 0.0,
        "majority_rf_accepted": float(acc_maj.mean()) if len(refs) else 0.0,
        "minority_rf_accepted": float(acc_min.mean()) if len(refs) else 0.0,
        "either_rf_accepted": float((acc_maj | acc_min).mean()) if len(refs) else 0.0,
    }

    # Beats the adaptive detector misses
    adaptive_hit = np.array([bool(np.any(np.abs(adaptive_peaks - r) <= tol)) for r in refs]) if len(refs) else np.array([], bool)
    missed = ~adaptive_hit
    out["adaptive_missed_refs"] = {
        "count": int(missed.sum()),
        "of_which_no_candidate_in_majority_stream": int((missed & ~cov_maj).sum()),
        "of_which_minority_candidate_exists": int((missed & cov_min).sum()),
        "of_which_minority_candidate_rf_accepted": int((missed & acc_min).sum()),
    }

    # RF probability distributions (minority stream)
    near_ref = _near(peaks, refs, tol)
    missed_refs = refs[missed] if len(refs) else refs
    near_missed = _near(peaks, missed_refs, tol)
    out["rf_probability_minority_stream"] = {
        "near_a_reference_beat": _quantiles(prob[(~is_major) & near_ref]),
        "near_an_adaptive_missed_beat": _quantiles(prob[(~is_major) & near_missed]),
        "far_from_every_reference_beat": _quantiles(prob[(~is_major) & ~near_ref]),
    }
    out["rf_probability_majority_stream"] = {
        "near_a_reference_beat": _quantiles(prob[is_major & near_ref]),
        "far_from_every_reference_beat": _quantiles(prob[is_major & ~near_ref]),
    }

    # Where do merge-added false positives sit?
    added = np.setdiff1d(merge_peaks, adaptive_peaks)
    added_fp = added[~_near(added, refs, tol)] if len(added) else added
    out["merge_added_peaks"] = {"count": int(len(added)), "false_positives": int(len(added_fp))}
    if len(added_fp) and len(refs):
        idx = np.searchsorted(refs, added_fp)
        prev_ref = refs[np.clip(idx - 1, 0, len(refs) - 1)]
        dt_ms = (added_fp - prev_ref) * 1000.0 / fs   # time after the nearest preceding true beat
        bands = {
            "0_100ms_after_beat (Q/R/S band; dedup should catch)": (0, 100),
            "100_120ms": (100, 120),
            "120_300ms_after_beat (T-wave band)": (120, 300),
            "300ms_plus": (300, 1e9),
        }
        n = len(added_fp)
        out["merge_added_peaks"]["fp_by_time_after_preceding_true_beat"] = {
            k: float(np.mean((dt_ms >= lo) & (dt_ms < hi))) for k, (lo, hi) in bands.items()
        }
        sig = z if major_id == 1 else -z   # minority-stream orientation
        height_sigma = sig[added_fp] / max(scale, 1e-12)
        out["merge_added_peaks"]["fp_peak_height_above_baseline_in_scale_units"] = _quantiles(height_sigma)
        out["merge_added_peaks"]["fp_fraction_below_1_scale_unit_(baseline_blips)"] = float(np.mean(height_sigma < 1.0))
        out["merge_added_peaks"]["fp_rf_probability"] = _quantiles(
            np.array([prob[np.searchsorted(peaks, p)] for p in added_fp if np.any(peaks == p)])
        )
    return out


def evaluate_record(base: Path, model, scale_method: str, specs: dict, want_detail: bool) -> dict:
    signal, fs = _record_signal(base, channel=0)
    refs = load_reference_annotations(base, extension="atr", symbols=sorted(DEFAULT_BEAT_SYMBOLS))
    if refs.size and np.any(refs < 0):
        raise ValueError("negative annotation sample index")

    adaptive, _ = detect_r_peaks_two_stage(signal, fs, model, polarity="adaptive", scale_method=scale_method)
    majority = select_signal_polarity(signal, fs, scale_method=scale_method).polarity
    major_id = 0 if majority != "negative" else 1

    results = {"adaptive": _metrics(adaptive, refs, fs)}
    scored_cache: dict[tuple, dict] = {}
    thr = float(model.metadata.threshold)
    for name, sp in specs.items():
        key = (sp["feature_scope"], sp["floor"])
        if key not in scored_cache:
            scored_cache[key] = _score_dual_polarity_streams(
                signal, fs, model, scale_method=scale_method, feature_scope=sp["feature_scope"],
                minority_stream=1 - major_id, minority_min_peak_scale=sp["floor"],
            )
        peaks, _ = _combine_scored_streams(
            scored_cache[key], major_id, thr, fs, merge_scope=sp["scope"], minority_threshold=sp["minor"]
        )
        results[name] = _metrics(peaks, refs, fs)

    payload = {"record": base.name, "fs_hz": fs, "selected_polarity": majority, "variants": results}
    if want_detail:
        merge_peaks, _ = _combine_scored_streams(scored_cache[("per_stream", 0.0)], major_id, thr, fs, merge_scope="all")
        payload["detail"] = detail_for_record(
            signal, fs, refs, model, adaptive, merge_peaks, scored_cache[("per_stream", 0.0)], major_id, scale_method
        )
    return payload


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict], variants: list[str], gate_record: str) -> dict:
    out = {}
    base = {r["record"]: r["variants"]["adaptive"] for r in records}
    for v in variants:
        tp = sum(r["variants"][v]["tp"] for r in records)
        fp = sum(r["variants"][v]["fp"] for r in records)
        fn = sum(r["variants"][v]["fn"] for r in records)
        sens = tp / max(tp + fn, 1)
        ppv = tp / max(tp + fp, 1)
        f1 = 2 * sens * ppv / max(sens + ppv, 1e-12)
        deltas = {r["record"]: r["variants"][v]["f1"] - base[r["record"]]["f1"] for r in records}
        gate_delta = deltas.get(gate_record)
        regressed = sorted([k for k, d in deltas.items() if d < REGRESSION_DELTA], key=lambda k: deltas[k])
        improved = sorted([k for k, d in deltas.items() if d > IMPROVEMENT_DELTA], key=lambda k: -deltas[k])
        out[v] = {
            "micro_f1": f1, "micro_sens": sens, "micro_ppv": ppv,
            "mean_record_f1": float(np.mean([r["variants"][v]["f1"] for r in records])),
            f"{gate_record}_delta_f1": gate_delta,
            "n_regressed": len(regressed), "n_improved": len(improved),
            "worst_regressions": [(k, round(deltas[k], 4)) for k in regressed[:5]],
            "best_improvements": [(k, round(deltas[k], 4)) for k in improved[:5]],
            "passes_ship_gate": bool(
                v != "adaptive" and gate_delta is not None and gate_delta > 0.05 and len(regressed) == 0
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incart-dir", default=".cache/physionet/incartdb")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--scale-method", default="windowed_std",
                    choices=["std", "mad", "windowed_mad", "windowed_std", "adaptive"])
    ap.add_argument("--records", nargs="*", default=None, help="Default: every record with a .hea in --incart-dir")
    ap.add_argument("--detail-records", nargs="*", default=["I62"])
    ap.add_argument("--gate-record", default="I62")
    ap.add_argument("--output-dir", default="validation_reports")
    args = ap.parse_args()

    incart = Path(args.incart_dir)
    if not incart.exists():
        raise SystemExit(f"Missing INCART directory: {incart}")
    model = CandidateSuppressor.load(args.model_path)
    if not model.fitted:
        raise SystemExit("model is not fitted")
    names = args.records or sorted(p.stem for p in incart.glob("*.hea"))
    specs = variant_specs()

    records, skipped = [], []
    for i, name in enumerate(names, 1):
        try:
            rec = evaluate_record(incart / name, model, args.scale_method, specs, name in set(args.detail_records))
            records.append(rec)
            v = rec["variants"]
            print(f"[{i}/{len(names)}] {name}: adaptive F1={v['adaptive']['f1']:.4f}  "
                  f"pooled={v['pooled_floor0_all']['f1']:.4f}  ps_floor0={v['per_stream_floor0_all']['f1']:.4f}  "
                  f"ps_floor1={v['per_stream_floor1_all']['f1']:.4f}  gaps1={v['per_stream_floor1_gaps']['f1']:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the sweep going, record why
            skipped.append({"record": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{i}/{len(names)}] {name}: SKIPPED: {exc}", flush=True)
    if not records:
        raise SystemExit("No usable records.")

    variants = ["adaptive"] + list(specs)
    agg = aggregate(records, variants, args.gate_record)

    print("\n=== AGGREGATE (vs adaptive; ship gate: gate-record dF1 > +0.05 AND zero records with dF1 < -0.02) ===")
    print(f"{'variant':34s} {'microF1':>8s} {'sens':>7s} {'ppv':>7s} {args.gate_record + ' dF1':>9s} {'regr':>5s} {'impr':>5s} gate")
    for v in variants:
        a = agg[v]
        gd = a[f"{args.gate_record}_delta_f1"]
        print(f"{v:34s} {a['micro_f1']:8.4f} {a['micro_sens']:7.4f} {a['micro_ppv']:7.4f} "
              f"{'' if gd is None else format(gd, '+9.4f'):>9s} {a['n_regressed']:5d} {a['n_improved']:5d} "
              f"{'PASS' if a['passes_ship_gate'] else ''}")
    for rec in records:
        if "detail" in rec:
            print(f"\n=== DETAIL {rec['record']} ===")
            print(json.dumps(rec["detail"], indent=2))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"merge_diagnostics_{args.scale_method}_{datetime.now(timezone.utc).date().isoformat()}.json"
    path.write_text(json.dumps({
        "schema": "electrotrace.merge_diagnostics/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path), "scale_method": args.scale_method,
        "rf_threshold": float(model.metadata.threshold), "tolerance_ms": TOL_MS,
        "aggregate": agg, "records": records, "skipped": skipped,
    }, indent=2) + "\n", encoding="utf-8")
    print("\nReport written to:", path, "| skipped:", len(skipped))


if __name__ == "__main__":
    main()
