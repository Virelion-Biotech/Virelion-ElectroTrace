"""Cheap recording metadata discovery without materializing full signals."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formats import _unique_labels


def _csv_metadata(path: Path) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    if header.empty:
        raise ValueError("CSV has no columns")
    from .io import guess_time_column
    time_col = guess_time_column(header.columns)
    if time_col is None:
        raise ValueError("No time column found")
    signal_cols = [str(c) for c in header.columns if str(c) != str(time_col)]
    if not signal_cols:
        raise ValueError("At least one signal channel is required")
    first_value = None
    last_value = None
    previous = None
    count = 0
    dt_values: list[float] = []
    for chunk in pd.read_csv(path, usecols=[time_col], chunksize=100_000):
        values = pd.to_numeric(chunk[time_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("CSV time values must be finite")
        if len(values):
            if first_value is None:
                first_value = float(values[0])
            if previous is not None:
                boundary_dt = float(values[0] - previous)
                if boundary_dt <= 0:
                    raise ValueError("CSV time values must be strictly increasing")
                dt_values.append(boundary_dt)
            if len(values) > 1:
                diffs = np.diff(values)
                if np.any(diffs <= 0):
                    raise ValueError("CSV time values must be strictly increasing")
                dt_values.extend(diffs.tolist())
            previous = float(values[-1])
            last_value = previous
            count += len(values)
    if count < 2 or first_value is None or last_value is None:
        raise ValueError("Recording must contain at least two samples")
    median_dt = float(np.median(np.asarray(dt_values, dtype=float)))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Could not infer a valid sampling interval")
    fs = round(1.0 / median_dt, 6)
    return {"source_format": "csv", "sampling_rate_hz": fs, "n_samples": count, "time_start_s": first_value, "time_end_s": last_value, "duration_s": float(last_value - first_value), "channels": signal_cols}


def _edf_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("EDF support requires pyedflib") from exc
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = _unique_labels([str(x).strip() for x in reader.getSignalLabels()])
        if not reader.signals_in_file:
            raise ValueError("EDF contains no signal channels")
        all_rates = [float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
        if any(not np.isfinite(r) or r <= 0 for r in all_rates):
            raise ValueError("EDF contains an invalid sampling rate")
        if any(abs(r - all_rates[0]) > 1e-9 for r in all_rates):
            raise ValueError("EDF channels have different sampling rates")
        counts = [int(x) for x in reader.getNSamples()]
        n = min(counts)
        if n < 2:
            raise ValueError("EDF contains fewer than two samples")
        fs = all_rates[0]
        return {"source_format": "edf", "sampling_rate_hz": fs, "n_samples": n, "time_start_s": 0.0, "time_end_s": float((n - 1) / fs), "duration_s": float((n - 1) / fs), "channels": labels}
    finally:
        reader.close()


def _wfdb_metadata(path: Path) -> dict[str, Any]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB support requires wfdb") from exc
    header = wfdb.rdheader(str(path))
    fs = float(header.fs)
    n = int(header.sig_len)
    if not np.isfinite(fs) or fs <= 0 or n < 2:
        raise ValueError("WFDB record contains invalid metadata")
    return {"source_format": "wfdb", "sampling_rate_hz": fs, "n_samples": n, "time_start_s": 0.0, "time_end_s": float((n - 1) / fs), "duration_s": float((n - 1) / fs), "channels": _unique_labels([str(x) for x in header.sig_name])}


def recording_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_metadata(path)
    if suffix == ".edf":
        return _edf_metadata(path)
    if suffix in {".hea", ".dat", ".atr"}:
        return _wfdb_metadata(path.with_suffix(""))
    raise ValueError(f"Unsupported recording format: {suffix}")
