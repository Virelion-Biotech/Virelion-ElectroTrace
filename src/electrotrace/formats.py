"""Native electrophysiology import for EDF and zipped WFDB records."""
from __future__ import annotations

from io import BytesIO
import os
import tempfile
import zipfile

import numpy as np

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 256
MAX_MEMBER_BYTES = 512 * 1024 * 1024


def _unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, raw in enumerate(labels):
        base = str(raw).strip() or f"channel_{i + 1}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        out.append(base if count == 1 else f"{base}_{count}")
    return out


def _payload(times, signals: np.ndarray, channel_names: list[str], source_format: str) -> dict:
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, None]
    channel_names = _unique_labels(channel_names)
    if signals.ndim != 2 or signals.shape[1] != len(channel_names):
        raise ValueError("signal/channel dimensions do not match")
    if len(times) and (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("recording time axis must be finite and strictly increasing")
    fs = None
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
        if dt > 0:
            fs = round(1.0 / dt, 6)
    return {"time": np.asarray(times, dtype=float).tolist(), "signals": {name: signals[:, i].tolist() for i, name in enumerate(channel_names)}, "signal_cols": channel_names, "sampling_rate_hz": fs, "duration_s": float(times[-1] - times[0]) if len(times) else 0.0, "n_samples": int(len(times)), "source_format": source_format}


def load_edf(file_bytes: bytes) -> dict:
    if len(file_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("EDF file exceeds the 512 MB upload limit")
    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("EDF import requires pyedflib") from exc
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        reader = pyedflib.EdfReader(path)
        try:
            channels = _unique_labels([str(x).strip() for x in reader.getSignalLabels()])
            arrays = [reader.readSignal(i) for i in range(reader.signals_in_file)]
            rates = [float(reader.getSampleFrequency(i)) for i in range(reader.signals_in_file)]
            if not arrays:
                raise ValueError("EDF contains no signal channels")
            fs = rates[0]
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError("EDF contains an invalid sampling rate")
            if any(abs(r - fs) > 1e-9 for r in rates):
                raise ValueError("EDF channels have different sampling rates; choose/export a common-rate channel first")
            n = min(len(x) for x in arrays)
            if n < 2:
                raise ValueError("EDF contains fewer than two samples")
            signals = np.column_stack([np.asarray(x[:n], dtype=float) for x in arrays])
            if not np.isfinite(signals).all():
                raise ValueError("EDF contains NaN or infinite signal values")
            time = np.arange(n, dtype=float) / fs
            return _payload(time, signals, channels, "EDF")
        finally:
            reader.close()
    finally:
        os.unlink(path)


def _safe_extract(archive: zipfile.ZipFile, root: str) -> None:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"WFDB ZIP contains too many files (max {MAX_ARCHIVE_MEMBERS})")
    total = 0
    root_path = os.path.realpath(root)
    for member in members:
        name = member.filename
        if not name or name.endswith("/"):
            continue
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise ValueError("WFDB ZIP contains an oversized member")
        total += member.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("WFDB ZIP uncompressed size exceeds the 512 MB limit")
        target = os.path.realpath(os.path.join(root, name))
        if not (target == root_path or target.startswith(root_path + os.sep)):
            raise ValueError("Unsafe archive path")
        mode = (member.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError("WFDB ZIP symlinks are not supported")
    archive.extractall(root)


def load_wfdb_zip(file_bytes: bytes) -> dict:
    if len(file_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("WFDB ZIP exceeds the 512 MB upload limit")
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB import requires wfdb") from exc
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            if archive.testzip() is not None:
                raise ValueError("WFDB ZIP is corrupted")
            _safe_extract(archive, tmp)
        headers = []
        for root, _, files in os.walk(tmp):
            headers.extend(os.path.join(root, f) for f in files if f.lower().endswith(".hea"))
        if not headers:
            raise ValueError("WFDB ZIP must contain a .hea header and matching signal files")
        if len(headers) > 1:
            raise ValueError("WFDB ZIP must contain exactly one record header")
        record_path = os.path.splitext(headers[0])[0]
        record = wfdb.rdrecord(record_path)
        fs = float(record.fs)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("WFDB record contains an invalid sampling rate")
        signals = np.asarray(record.p_signal if record.p_signal is not None else record.d_signal, dtype=float)
        if signals.ndim != 2 or signals.shape[0] < 2:
            raise ValueError("WFDB record contains insufficient signal data")
        if not np.isfinite(signals).all():
            raise ValueError("WFDB record contains NaN or infinite signal values")
        channels = _unique_labels([str(x) for x in record.sig_name])
        if len(channels) != signals.shape[1]:
            raise ValueError("WFDB channel metadata does not match signal dimensions")
        time = np.arange(signals.shape[0], dtype=float) / fs
        return _payload(time, signals, channels, "WFDB")


def load_electrophysiology(file_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename.lower())[1]
    if suffix == ".edf":
        return load_edf(file_bytes)
    if suffix in {".zip", ".wfdb"}:
        return load_wfdb_zip(file_bytes)
    raise ValueError("Supported native formats are .edf and WFDB .zip archives")
