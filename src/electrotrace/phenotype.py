"""Beat-level electrophysiology phenotype extraction."""
from __future__ import annotations

import numpy as np


def _validate_inputs(time: np.ndarray, signal: np.ndarray) -> float:
    if time.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and signal must be one-dimensional arrays")
    if len(time) != len(signal) or len(time) < 2:
        raise ValueError("time and signal must have equal length with at least two samples")
    if not np.isfinite(time).all():
        raise ValueError("time contains NaN or infinite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or infinite values")
    dt = float(np.median(np.diff(time)))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("could not infer a valid sampling interval")
    return dt


def beat_phenotypes(time: np.ndarray, signal: np.ndarray, r_indices: np.ndarray) -> list[dict]:
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    r_indices = np.asarray(r_indices, dtype=int)
    dt = _validate_inputs(time, signal)
    peaks = np.unique(r_indices[(r_indices >= 0) & (r_indices < len(time))])
    out = []
    r_times = time[peaks]
    for i, idx in enumerate(peaks):
        prev_rr = float(r_times[i] - r_times[i - 1]) if i else None
        next_rr = float(r_times[i + 1] - r_times[i]) if i + 1 < len(r_times) else None
        rr = prev_rr if prev_rr is not None and prev_rr > 0 else next_rr
        hr = 60.0 / rr if rr is not None and rr > 0 else None
        left = max(0, idx - 2)
        right = min(len(signal), idx + 3)
        r_amp = float(np.mean(signal[left:right]))
        radius = max(1, int(round(0.20 / dt)))
        before = max(0, idx - radius)
        after = min(len(signal), idx + radius)
        local = signal[before:after]
        baseline_parts = [signal[before:idx], signal[idx:after]]
        baseline_values = np.concatenate([part for part in baseline_parts if len(part)]) if any(len(part) for part in baseline_parts) else np.array([], dtype=float)
        baseline = float(np.median(baseline_values)) if len(baseline_values) else 0.0
        qrs_width_proxy = None
        if len(local) > 5:
            centered = local - baseline
            peak_abs = float(np.max(np.abs(centered)))
            threshold = 0.5 * peak_abs
            if threshold > 0:
                crossing = np.flatnonzero(np.abs(centered) >= threshold)
                if len(crossing) >= 2:
                    qrs_width_proxy = float(time[before + crossing[-1]] - time[before + crossing[0]])
        out.append({
            "r_index": int(idx),
            "r_time_s": float(time[idx]),
            "rr_prev_s": prev_rr,
            "rr_next_s": next_rr,
            "heart_rate_bpm": hr,
            "r_amplitude": r_amp,
            "qrs_width_proxy_s": qrs_width_proxy,
        })
    return out


def summary_statistics(phenotypes: list[dict]) -> dict:
    def vals(key: str):
        return np.asarray([x[key] for x in phenotypes if x.get(key) is not None and np.isfinite(x[key])], dtype=float)
    summary = {"n_beats": len(phenotypes)}
    for key in ("heart_rate_bpm", "rr_prev_s", "r_amplitude", "qrs_width_proxy_s"):
        x = vals(key)
        summary[key] = {
            "mean": float(np.mean(x)) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "p25": float(np.percentile(x, 25)) if len(x) else None,
            "p75": float(np.percentile(x, 75)) if len(x) else None,
        }
    hr = vals("heart_rate_bpm")
    summary["tachycardia_fraction"] = float(np.mean(hr > 100)) if len(hr) else None
    summary["bradycardia_fraction"] = float(np.mean(hr < 60)) if len(hr) else None
    return summary
