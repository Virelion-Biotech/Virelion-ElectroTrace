"""ElectroTrace research API server."""
from __future__ import annotations

import os
import re
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy import signal as sps

from electrotrace.beats import segment_beats
from electrotrace.benchmark import benchmark_models
from electrotrace.formats import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_MEMBERS, MAX_MEMBER_BYTES, load_electrophysiology
from electrotrace.io import load_csv, load_recording, validate_dataframe
from electrotrace.ml import rank_uncertain, train_classifier
from electrotrace.phenotype import beat_phenotypes, summary_statistics
from electrotrace.project_store import ProjectStore, RecordingRef
from electrotrace.signal import FilterConfigurationError, apply_pipeline
from electrotrace.statistics import benjamini_hochberg, compare_groups

WEB = os.path.join(ROOT, "web")
SAMPLE_DATA = os.path.join(ROOT, "sample_data")
UPLOAD_ROOT = Path(os.getenv("ELECTROTRACE_UPLOAD_ROOT", os.path.join(tempfile.gettempdir(), "electrotrace_uploads")))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(os.getenv("ELECTROTRACE_PROJECT_ROOT", os.path.join(ROOT, "projects"))).resolve()
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=WEB, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVE_BYTES


def _json() -> dict:
    return request.get_json(silent=True) or {}


def _project_store(name: str) -> ProjectStore:
    clean = str(name).strip()
    if not clean or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", clean):
        raise ValueError("project name must use only letters, numbers, '.', '_' or '-' and be at most 64 characters")
    path = (PROJECT_ROOT / clean).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise ValueError("invalid project path")
    return ProjectStore(path)


def _safe_extract(archive: zipfile.ZipFile, root: str) -> None:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"WFDB ZIP contains too many files (max {MAX_ARCHIVE_MEMBERS})")
    total = 0
    root_path = os.path.realpath(root)
    for member in members:
        if not member.filename or member.filename.endswith("/"):
            continue
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise ValueError("WFDB ZIP contains an oversized member")
        total += member.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("WFDB ZIP uncompressed size exceeds the 512 MB limit")
        target = os.path.realpath(os.path.join(root, member.filename))
        if not (target == root_path or target.startswith(root_path + os.sep)):
            raise ValueError("Unsafe archive path")
        mode = (member.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError("WFDB ZIP symlinks are not supported")
    archive.extractall(root)


def _save_upload(uploaded) -> tuple[str, Path]:
    name = Path(uploaded.filename or "recording.bin")
    suffix = name.suffix.lower()
    token = uuid.uuid4().hex
    destination = UPLOAD_ROOT / f"{token}{suffix}"
    uploaded.save(destination)
    if destination.stat().st_size > MAX_ARCHIVE_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError("Uploaded file exceeds the 512 MB limit")
    if suffix != ".zip":
        return token, destination
    extract_dir = UPLOAD_ROOT / token
    extract_dir.mkdir()
    try:
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError("Uploaded ZIP is corrupted")
            _safe_extract(archive, str(extract_dir))
        headers = list(extract_dir.rglob("*.hea"))
        if len(headers) != 1:
            raise ValueError("WFDB ZIP must contain exactly one .hea record header")
        return token, headers[0].with_suffix("")
    except Exception:
        for path in sorted(extract_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        extract_dir.rmdir()
        destination.unlink(missing_ok=True)
        raise


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Uploaded request exceeds the 512 MB limit."}), 413


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/sample_data/<path:filename>")
def sample_data(filename: str):
    return send_from_directory(SAMPLE_DATA, filename)


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "No recording uploaded."}), 400
    filename = uploaded.filename or "recording.csv"
    suffix = Path(filename).suffix.lower()
    try:
        raw = uploaded.read()
        if len(raw) > MAX_ARCHIVE_BYTES:
            raise ValueError("Uploaded file exceeds the 512 MB limit")
        if suffix == ".csv":
            df = load_csv(raw)
            result = validate_dataframe(df, request.form.get("time_col") or None)
            payload = {
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "time_col": result.time_col,
                "signal_cols": result.signal_cols,
                "sampling_rate_hz": result.sampling_rate_hz,
                "duration_s": result.duration_s,
                "n_samples": result.n_samples,
                "format": "csv",
                "source_format": "CSV",
                "filename": filename,
            }
            if result.valid:
                payload["time"] = df[result.time_col].astype(float).tolist()
                payload["signals"] = {c: df[c].astype(float).fillna(0).tolist() for c in result.signal_cols}
            return jsonify(payload)
        native = load_electrophysiology(raw, filename)
        native.update({"valid": True, "errors": [], "warnings": [], "format": native["source_format"].lower(), "filename": filename})
        return jsonify(native)
    except Exception as exc:
        return jsonify({"error": f"Could not read recording: {exc}"}), 400


@app.post("/api/recording")
def recording_upload():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "No recording uploaded."}), 400
    token = None
    path = None
    try:
        token, path = _save_upload(uploaded)
        recording = load_recording(path)
        return jsonify({
            "recording_id": token,
            "filename": uploaded.filename,
            "format": recording.source_format,
            "sampling_rate_hz": recording.sampling_rate_hz,
            "duration_s": float(recording.time[-1]) if len(recording.time) else 0.0,
            "n_samples": len(recording.time),
            "channels": list(recording.signals.keys()),
        })
    except Exception as exc:
        if path is not None and path.exists() and path.is_file():
            path.unlink(missing_ok=True)
        if token and (UPLOAD_ROOT / token).is_dir():
            for item in sorted((UPLOAD_ROOT / token).rglob("*"), reverse=True):
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    item.rmdir()
            (UPLOAD_ROOT / token).rmdir()
        return jsonify({"error": str(exc)}), 400


@app.get("/api/recording/<recording_id>/window")
def recording_window(recording_id: str):
    if len(recording_id) != 32 or any(c not in "0123456789abcdef" for c in recording_id.lower()):
        return jsonify({"error": "invalid recording id"}), 400
    try:
        matches = list(UPLOAD_ROOT.glob(recording_id)) + list(UPLOAD_ROOT.glob(recording_id + ".*"))
        path = matches[0] if matches else None
        if path is None and (UPLOAD_ROOT / recording_id).is_dir():
            headers = list((UPLOAD_ROOT / recording_id).rglob("*.hea"))
            path = headers[0].with_suffix("") if len(headers) == 1 else None
        if path is None:
            return jsonify({"error": "recording not found"}), 404
        recording = load_recording(path)
        start = max(0, int(request.args.get("start", 0)))
        stop = min(len(recording.time), int(request.args.get("stop", min(start + 5000, len(recording.time)))))
        if stop <= start:
            raise ValueError("invalid window")
        return jsonify({"start": start, "stop": stop, "time": recording.time[start:stop].tolist(), "signals": {c: y[start:stop].tolist() for c, y in recording.signals.items()}})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/filter")
def filter_signal():
    data = _json()
    try:
        fs = float(data["sampling_rate_hz"])
        raw = np.asarray(data["signal"], dtype=float)
        filtered = apply_pipeline(raw, fs, baseline=bool(data.get("baseline", False)), highpass_hz=float(data["highpass_hz"]) if data.get("highpass_hz") is not None else None, lowpass_hz=float(data["lowpass_hz"]) if data.get("lowpass_hz") is not None else None, notch_hz=float(data["notch_hz"]) if data.get("notch_hz") is not None else None)
        return jsonify({"signal": filtered.tolist()})
    except (KeyError, TypeError, ValueError, FilterConfigurationError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/detect/r-peaks")
def detect_r_peaks():
    data = _json()
    try:
        fs = float(data["sampling_rate_hz"])
        y = np.asarray(data["signal"], dtype=float)
        distance_ms = float(data.get("min_distance_ms", 250))
        prominence_factor = float(data.get("prominence_factor", 0.5))
        if distance_ms <= 0 or prominence_factor < 0:
            raise ValueError("peak detector settings must be valid positive values")
        distance = max(1, int(round(fs * distance_ms / 1000.0)))
        centered = np.nan_to_num(y - np.nanmedian(y))
        scale = float(np.nanstd(centered)) or 1.0
        peaks, props = sps.find_peaks(centered, distance=distance, prominence=scale * prominence_factor)
        return jsonify({"peaks": peaks.astype(int).tolist(), "prominences": props.get("prominences", np.zeros(len(peaks))).astype(float).tolist()})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/beats")
@app.post("/api/segment")
def beats():
    data = _json()
    try:
        time = np.asarray(data["time"], dtype=float)
        peaks = np.asarray(data.get("peaks") or [], dtype=int)
        if len(peaks) == 0 and data.get("signal") is not None:
            fs = float(data["sampling_rate_hz"])
            y = np.asarray(data["signal"], dtype=float)
            centered = np.nan_to_num(y - np.nanmedian(y))
            scale = float(np.nanstd(centered)) or 1.0
            peaks, _ = sps.find_peaks(centered, distance=max(1, int(round(fs * 0.25))), prominence=scale * 0.5)
        result = segment_beats(time, peaks, float(data.get("pre_s", 0.35)), float(data.get("post_s", 0.55)))
        return jsonify({"beats": [b.to_dict() for b in result], "peaks": peaks.tolist(), "n_beats": len(result)})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/phenotype")
def phenotype():
    data = _json()
    try:
        phenotypes = beat_phenotypes(np.asarray(data["time"], dtype=float), np.asarray(data["signal"], dtype=float), np.asarray(data["r_indices"], dtype=int))
        return jsonify({"beats": phenotypes, "summary": summary_statistics(phenotypes)})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/statistics/compare")
def statistics_compare():
    data = _json()
    try:
        return jsonify(compare_groups(data["group_a"], data["group_b"]))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/statistics/fdr")
def statistics_fdr():
    data = _json()
    try:
        return jsonify({"adjusted_p": benjamini_hochberg(data["p_values"])})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/benchmark")
def benchmark():
    data = _json()
    try:
        return jsonify(benchmark_models(np.asarray(data["X"], dtype=float), np.asarray(data["y"]), np.asarray(data["groups"]), int(data.get("folds", 5))))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/ml/train")
def ml_train():
    data = _json()
    try:
        model, metrics = train_classifier(np.asarray(data["signal"], dtype=float), float(data["sampling_rate_hz"]), np.asarray(data["peaks"], dtype=int), list(data.get("annotations", [])), time=np.asarray(data["time"], dtype=float) if data.get("time") is not None else None)
        return jsonify({"trained": True, "metrics": metrics, "model": model.__class__.__name__})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"trained": False, "error": str(exc)}), 400


@app.post("/api/ml/suggest")
def ml_suggest():
    data = _json()
    try:
        signal = np.asarray(data["signal"], dtype=float)
        fs = float(data["sampling_rate_hz"])
        peaks = np.asarray(data["peaks"], dtype=int)
        time = np.asarray(data["time"], dtype=float) if data.get("time") is not None else None
        model, metrics = train_classifier(signal, fs, peaks, list(data.get("annotations", [])), time=time)
        return jsonify({"trained": True, "metrics": metrics, "suggestions": rank_uncertain(signal, fs, peaks, model, int(data.get("top_n", 20)), time=time)})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"trained": False, "error": str(exc)}), 400


@app.get("/api/project")
def project_get():
    try:
        return jsonify(_project_store(request.args.get("name", "default")).load().to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/project/recording")
def project_recording():
    data = _json()
    try:
        store = _project_store(str(data.get("project", "default")))
        ref = RecordingRef(recording_id=str(data["recording_id"]), subject_id=str(data["subject_id"]), group=str(data.get("group", "")), visit=str(data.get("visit", "")), source_path=str(data.get("source_path", "")), format=str(data.get("format", "")), sampling_rate_hz=data.get("sampling_rate_hz"), duration_s=data.get("duration_s"), channels=list(data.get("channels", [])), metadata=dict(data.get("metadata", {})))
        return jsonify(store.add_recording(ref).to_dict())
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/<path:path>")
def static_file(path: str):
    return send_from_directory(WEB, path)


if __name__ == "__main__":
    app.run(host=os.getenv("ELECTROTRACE_HOST", "127.0.0.1"), port=int(os.getenv("ELECTROTRACE_PORT", "5000")), debug=False)
