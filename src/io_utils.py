"""
File loading helpers: CSV signal parsing + column/sampling-rate inference.
"""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

TIME_COL_CANDIDATES = ["time", "t", "timestamp", "time_s", "seconds"]


def guess_time_column(columns: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in TIME_COL_CANDIDATES:
        if cand in lower:
            return lower[cand]
    return None


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def infer_sampling_rate(time_values: np.ndarray) -> Optional[float]:
    if len(time_values) < 2:
        return None
    diffs = np.diff(time_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return round(1.0 / median_dt, 3)


def build_synthetic_time(n_samples: int, sampling_rate: float) -> np.ndarray:
    return np.arange(n_samples) / sampling_rate


def summarize(df: pd.DataFrame, time_col: str, signal_cols: List[str]) -> dict:
    time_vals = df[time_col].to_numpy(dtype=float)
    fs = infer_sampling_rate(time_vals)
    duration = float(time_vals[-1] - time_vals[0]) if len(time_vals) else 0.0
    return {
        "n_samples": len(df),
        "channels": signal_cols,
        "sampling_rate_hz": fs,
        "duration_s": round(duration, 3),
    }
