"""Robust ECG CSV loading, validation, and metadata inference."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TIME_COL_CANDIDATES = ["time", "t", "timestamp", "time_s", "seconds"]

@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    time_col: str | None
    signal_cols: list[str]
    sampling_rate_hz: float | None
    duration_s: float | None
    n_samples: int

    @property
    def valid(self) -> bool:
        return not self.errors

def guess_time_column(columns: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in TIME_COL_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None

def infer_sampling_rate(time_values: np.ndarray) -> float | None:
    if len(time_values) < 2:
        return None
    diffs = np.diff(time_values)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive) == 0:
        return None
    dt = float(np.median(positive))
    return round(1.0 / dt, 6) if dt > 0 else None

def validate_dataframe(df: pd.DataFrame, time_col: str | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    time_col = time_col or guess_time_column(df.columns)
    if time_col is None:
        errors.append("No time column found. Use one of: time, t, timestamp, time_s, seconds.")
        return ValidationResult(errors, warnings, None, [], None, None, len(df))
    if time_col not in df.columns:
        errors.append(f"Time column '{time_col}' does not exist.")
        return ValidationResult(errors, warnings, time_col, [], None, None, len(df))
    if len(df) < 2:
        errors.append("Recording must contain at least two samples.")
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(time).any() or np.isinf(time).any():
        errors.append("Time column contains non-numeric, NaN, or infinite values.")
    if len(time) >= 2:
        diffs = np.diff(time)
        if np.any(diffs <= 0):
            errors.append("Time values must be strictly increasing with no duplicate timestamps.")
    signal_cols = [str(c) for c in df.columns if str(c) != str(time_col)]
    if not signal_cols:
        errors.append("At least one signal channel is required.")
    for col in signal_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            errors.append(f"Signal channel '{col}' contains no numeric values.")
        elif numeric.isna().any():
            warnings.append(f"Signal channel '{col}' contains missing/non-numeric values.")
        if numeric.isin([np.inf, -np.inf]).any():
            errors.append(f"Signal channel '{col}' contains infinite values.")
    fs = infer_sampling_rate(time) if len(time) >= 2 and not np.any(np.diff(time) <= 0) else None
    duration = float(time[-1] - time[0]) if len(time) else None
    if duration is not None and duration <= 0:
        errors.append("Recording duration must be positive.")
    if fs is not None and not math.isfinite(fs):
        errors.append("Sampling rate could not be inferred reliably.")
    if fs is not None and fs < 10:
        warnings.append(f"Very low inferred sampling rate: {fs:g} Hz.")
    if fs is not None:
        diffs = np.diff(time)
        positive = diffs[diffs > 0]
        if len(positive) > 2 and np.std(positive) > max(np.median(positive) * 0.01, 1e-12):
            warnings.append("Sampling intervals are irregular; inferred rate uses the median interval.")
    return ValidationResult(errors, warnings, time_col, signal_cols, fs, duration, len(df))

def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def load_and_validate(file_bytes: bytes, time_col: str | None = None) -> tuple[pd.DataFrame, ValidationResult]:
    try:
        df = load_csv(file_bytes)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    result = validate_dataframe(df, time_col)
    if not result.valid:
        raise ValueError("CSV validation failed: " + " ".join(result.errors))
    return df, result


def summarize(df: pd.DataFrame, result: ValidationResult) -> dict:
    return {
        "n_samples": result.n_samples,
        "channels": result.signal_cols,
        "sampling_rate_hz": result.sampling_rate_hz,
        "duration_s": round(result.duration_s or 0.0, 6),
    }
