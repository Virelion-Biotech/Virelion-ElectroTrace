"""Beat-level electrophysiology phenotype extraction."""
from __future__ import annotations

import numpy as np


def beat_phenotypes(time: np.ndarray, signal: np.ndarray, r_indices: np.ndarray) -> list[dict]:
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    r_indices = np.asarray(r_indices, dtype=int)
    if len(time) != len(signal):
        raise ValueError("time and signal must have equal length")
    peaks = np.unique(r_indices[(r_indices >= 0) & (r_indices < len(time))])
    out = []
    r_times = time[peaks]
    for i, idx in enumerate(peaks):
        prev_rr = float(r_times[i] - r_times[i - 1]) if i else None
        next_rr = float(r_times[i + 1] - r_times[i]) if i + 1 < len(r_times) else None
        rr = prev_rr if prev_rr and prev_rr > 0 else next_rr
        hr = 60.0 / rr if rr else None
        left = max(0, idx - 2)
        right = min(len(signal), idx + 3)
        r_amp = float(np.nanmean(signal[left:right]))
        before = max(0, idx - (int(0.20 / max(np.median(np.diff(time)), 1e-9))))
        after = min(len(signal), idx + (int(0.20 / max(np.median(np.diff(time)), 1e-9))))
        local = signal[before:after]
        baseline = float(np.nanmedian(np.concatenate([signal[before:idx], signal[idx:after]]))) if after > before else 0.0
        qrs_width_proxy = None
        if local.size > 5:
            centered = local - baseline
            threshold = 0.5 * np.nanmax(np.abs(centered))
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
        return np.asarray([x[key] for x in phenotypes if x.get(key) is not None], dtype=float)
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
