"""ECG loading, validation, and normalized multi-format recording access."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
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
    format: str = "csv"
    @property
    def valid(self) -> bool:
        return not self.errors

@dataclass
class Recording:
    time: np.ndarray
    signals: dict[str, np.ndarray]
    sampling_rate_hz: float
    source_format: str
    channel_units: dict[str, str] | None = None
    metadata: dict | None = None


def _unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out=[]
    for i, raw in enumerate(labels):
        base=str(raw).strip() or f"channel_{i+1}"
        count=seen.get(base,0)+1
        seen[base]=count
        out.append(base if count==1 else f"{base}_{count}")
    return out


def guess_time_column(columns: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in TIME_COL_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None


def infer_sampling_rate(time_values: np.ndarray) -> float | None:
    if len(time_values) < 2:
        return None
    positive = np.diff(time_values)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) == 0:
        return None
    dt = float(np.median(positive))
    return round(1.0 / dt, 6) if dt > 0 else None


def validate_dataframe(df: pd.DataFrame, time_col: str | None = None, source_format: str = "csv") -> ValidationResult:
    errors=[]; warnings=[]
    time_col=time_col or guess_time_column(df.columns)
    if time_col is None:
        errors.append("No time column found. Use one of: time, t, timestamp, time_s, seconds.")
        return ValidationResult(errors,warnings,None,[],None,None,len(df),source_format)
    if time_col not in df.columns:
        errors.append(f"Time column '{time_col}' does not exist.")
        return ValidationResult(errors,warnings,time_col,[],None,None,len(df),source_format)
    if len(df)<2: errors.append("Recording must contain at least two samples.")
    time=pd.to_numeric(df[time_col],errors="coerce").to_numpy(dtype=float)
    if np.isnan(time).any() or np.isinf(time).any(): errors.append("Time column contains non-numeric, NaN, or infinite values.")
    if len(time)>=2 and np.any(np.diff(time)<=0): errors.append("Time values must be strictly increasing with no duplicate timestamps.")
    signal_cols=_unique_labels([str(c) for c in df.columns if str(c)!=str(time_col)])
    original_signal_cols=[str(c) for c in df.columns if str(c)!=str(time_col)]
    if not signal_cols: errors.append("At least one signal channel is required.")
    for col in original_signal_cols:
        numeric=pd.to_numeric(df[col],errors="coerce")
        if numeric.isna().all(): errors.append(f"Signal channel '{col}' contains no numeric values.")
        elif numeric.isna().any(): errors.append(f"Signal channel '{col}' contains missing/non-numeric values; impute or remove them before import.")
        if numeric.isin([np.inf,-np.inf]).any(): errors.append(f"Signal channel '{col}' contains infinite values.")
    fs=infer_sampling_rate(time) if len(time)>=2 and not np.any(np.diff(time)<=0) else None
    duration=float(time[-1]-time[0]) if len(time) else None
    if duration is not None and duration<=0: errors.append("Recording duration must be positive.")
    if fs is not None and not math.isfinite(fs): errors.append("Sampling rate could not be inferred reliably.")
    if fs is not None and fs<10: warnings.append(f"Very low inferred sampling rate: {fs:g} Hz.")
    if fs is not None:
        diffs=np.diff(time); positive=diffs[diffs>0]
        if len(positive)>2 and np.std(positive)>max(np.median(positive)*0.01,1e-12): warnings.append("Sampling intervals are irregular; inferred rate uses the median interval.")
    return ValidationResult(errors,warnings,time_col,signal_cols,fs,duration,len(df),source_format)


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def load_edf(file_path: str | Path) -> Recording:
    try: import pyedflib
    except ImportError as exc: raise RuntimeError("EDF support requires pyedflib") from exc
    reader=pyedflib.EdfReader(str(file_path))
    try:
        labels=_unique_labels([str(x).strip() for x in reader.getSignalLabels()])
        rates=[float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
        if not rates or not all(np.isfinite(r) and r>0 for r in rates): raise ValueError("EDF contains an invalid sampling rate")
        if len(set(rates))!=1: raise ValueError("EDF contains channels with different sampling rates; select/resample explicitly before analysis")
        fs=rates[0]; arrays=[np.asarray(reader.readSignal(i),dtype=float) for i in range(reader.signals_in_file)]
        n=min(map(len,arrays))
        if n<2: raise ValueError("EDF contains fewer than two samples")
        signals={label:arr[:n] for label,arr in zip(labels,arrays)}
        if not all(np.isfinite(v).all() for v in signals.values()): raise ValueError("EDF contains NaN or infinite signal values")
        time=np.arange(n,dtype=float)/fs
        units={label:str(reader.getPhysicalDimension(i) or "") for i,label in enumerate(labels)}
        return Recording(time,signals,fs,"edf",units,{"duration_s":float(reader.getFileDuration()),"time_start_s":0.0,"time_end_s":float(time[-1])})
    finally: reader.close()


def load_wfdb(record_path: str | Path) -> Recording:
    try: import wfdb
    except ImportError as exc: raise RuntimeError("WFDB support requires wfdb") from exc
    record=wfdb.rdrecord(str(record_path)); fs=float(record.fs)
    if not np.isfinite(fs) or fs<=0: raise ValueError("WFDB record contains an invalid sampling rate")
    labels=_unique_labels([str(name) for name in record.sig_name])
    signals={label:np.asarray(record.p_signal[:,i],dtype=float) for i,label in enumerate(labels)}
    if not signals or min(map(len,signals.values()))<2: raise ValueError("WFDB record contains insufficient signal data")
    if not all(np.isfinite(v).all() for v in signals.values()): raise ValueError("WFDB record contains NaN or infinite signal values")
    time=np.arange(record.sig_len,dtype=float)/fs
    return Recording(time,signals,fs,"wfdb",metadata={"record_name":record.record_name,"units":record.units,"time_start_s":0.0,"time_end_s":float(time[-1])})


def load_recording(file_path: str | Path) -> Recording:
    path=Path(file_path); suffix=path.suffix.lower()
    if suffix==".edf": return load_edf(path)
    if suffix in {".hea",".dat",".atr"}: return load_wfdb(path.with_suffix(""))
    if suffix==".csv":
        df=load_csv(path.read_bytes()); result=validate_dataframe(df)
        if not result.valid: raise ValueError("CSV validation failed: "+" ".join(result.errors))
        if result.sampling_rate_hz is None: raise ValueError("Could not infer a valid sampling rate from the CSV time axis")
        t=df[result.time_col].to_numpy(dtype=float)
        channels=_unique_labels([c for c in result.signal_cols])
        original=[c for c in df.columns if str(c)!=str(result.time_col)]
        signals={label:pd.to_numeric(df[col],errors="coerce").to_numpy(dtype=float) for label,col in zip(channels,original)}
        return Recording(t,signals,result.sampling_rate_hz,"csv",metadata={"time_start_s":float(t[0]),"time_end_s":float(t[-1])})
    raise ValueError(f"Unsupported recording format: {suffix}")


def load_and_validate(file_bytes: bytes, time_col: str | None = None) -> tuple[pd.DataFrame, ValidationResult]:
    try: df=load_csv(file_bytes)
    except Exception as exc: raise ValueError(f"Could not read CSV: {exc}") from exc
    result=validate_dataframe(df,time_col)
    if not result.valid: raise ValueError("CSV validation failed: "+" ".join(result.errors))
    return df,result


def summarize(df: pd.DataFrame, result: ValidationResult) -> dict:
    return {"n_samples":result.n_samples,"channels":result.signal_cols,"sampling_rate_hz":result.sampling_rate_hz,"duration_s":round(result.duration_s or 0.0,6),"format":result.format}
