"""ElectroTrace Flask application with deployment safeguards."""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from scipy import signal as sps

from .beats import segment_beats
from .benchmark import benchmark_models
from .formats import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_MEMBERS, MAX_MEMBER_BYTES, load_electrophysiology
from .io import load_csv, validate_dataframe
from .metadata import recording_metadata
from .ml import rank_uncertain, train_classifier
from .phenotype import beat_phenotypes, summary_statistics
from .project_store import ProjectStore, RecordingRef
from .security import ApiKeyMiddleware, configured_api_key, validate_bind
from .signal import FilterConfigurationError, apply_pipeline
from .statistics import benjamini_hochberg, compare_groups
from .window import read_recording_window

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
SAMPLE_DATA = ROOT / "sample_data"
UPLOAD_ROOT = Path(os.getenv("ELECTROTRACE_UPLOAD_ROOT", Path(tempfile.gettempdir()) / "electrotrace_uploads"))
PROJECT_ROOT = Path(os.getenv("ELECTROTRACE_PROJECT_ROOT", ROOT / "projects")).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_JSON_BODY_BYTES = 64 * 1024 * 1024
UPLOAD_TTL_S = float(os.getenv("ELECTROTRACE_UPLOAD_TTL_S", 24 * 3600))
API_KEY = configured_api_key()

app = Flask(__name__, static_folder=str(WEB), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVE_BYTES
if API_KEY:
    app.wsgi_app = ApiKeyMiddleware(app.wsgi_app, API_KEY)


def _cleanup_uploads() -> None:
    if not UPLOAD_TTL_S > 0:
        return
    cutoff = time.time() - UPLOAD_TTL_S
    for path in list(UPLOAD_ROOT.iterdir()):
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _json():
    length = request.content_length
    if length is not None and length > MAX_JSON_BODY_BYTES:
        raise ValueError("JSON request exceeds the 64 MB limit; use recording window endpoints for large signals")
    return request.get_json(silent=True) or {}


def _project_store(name):
    name = str(name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError("invalid project name")
    return ProjectStore((PROJECT_ROOT / name).resolve())


def _safe_extract(archive, root):
    members = archive.infolist()
    root = Path(root).resolve()
    total = 0
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("WFDB ZIP contains too many files")
    for member in members:
        if not member.filename or member.filename.endswith("/"):
            continue
        if member.file_size < 0 or member.file_size > MAX_MEMBER_BYTES:
            raise ValueError("WFDB ZIP contains an oversized member")
        total += member.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("WFDB ZIP uncompressed size exceeds the limit")
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Unsafe archive path")
        if ((member.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError("WFDB ZIP symlinks are not supported")
    archive.extractall(root)


def _save_upload(file):
    token = uuid.uuid4().hex
    suffix = Path(file.filename or "recording.bin").suffix.lower()
    destination = UPLOAD_ROOT / f"{token}{suffix}"
    file.save(destination)
    if destination.stat().st_size > MAX_ARCHIVE_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError("uploaded file exceeds the limit")
    if suffix not in {".zip", ".wfdb"}:
        return token, destination
    extracted = UPLOAD_ROOT / token
    extracted.mkdir()
    try:
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError("WFDB ZIP is corrupted")
            _safe_extract(archive, extracted)
        headers = list(extracted.rglob("*.hea"))
        if len(headers) != 1:
            raise ValueError("WFDB ZIP must contain exactly one .hea header")
        destination.unlink(missing_ok=True)
        return token, headers[0].with_suffix("")
    except Exception:
        for path in sorted(extracted.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        extracted.rmdir()
        destination.unlink(missing_ok=True)
        raise


@app.before_request
def _before_request():
    if request.path.startswith("/api/"):
        _cleanup_uploads()


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "request exceeds the 512 MB limit"}), 413


@app.errorhandler(ValueError)
def bad_value(error):
    return jsonify({"error": str(error)}), 400


@app.get("/")
def index():
    return send_from_directory(str(WEB), "index.html")


@app.get("/sample_data/<path:name>")
def sample_data(name):
    return send_from_directory(str(SAMPLE_DATA), name)


@app.post("/api/analyze")
def analyze():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No recording uploaded."}), 400
    filename = file.filename or "recording.csv"
    try:
        raw = file.read()
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("analyze endpoint is limited to 64 MB; use /api/recording and window access for larger recordings")
        if Path(filename).suffix.lower() == ".csv":
            dataframe = load_csv(raw)
            result = validate_dataframe(dataframe, request.form.get("time_col") or None)
            output = {"valid": result.valid, "errors": result.errors, "warnings": result.warnings, "time_col": result.time_col, "signal_cols": result.signal_cols, "sampling_rate_hz": result.sampling_rate_hz, "duration_s": result.duration_s, "n_samples": result.n_samples, "time_start_s": float(dataframe[result.time_col].iloc[0]) if result.valid else None, "time_end_s": float(dataframe[result.time_col].iloc[-1]) if result.valid else None, "format": "csv", "source_format": "CSV", "filename": filename}
            if result.valid:
                output["time"] = dataframe[result.time_col].astype(float).tolist()
                output["signals"] = {channel: dataframe[channel].astype(float).to_numpy().tolist() for channel in result.signal_cols}
            return jsonify(output)
        output = load_electrophysiology(raw, filename)
        output.update(valid=True, errors=[], warnings=[], format=output["source_format"].lower(), filename=filename, time_start_s=output["time"][0], time_end_s=output["time"][-1])
        return jsonify(output)
    except Exception as error:
        return jsonify({"error": f"Could not read recording: {error}"}), 400


@app.post("/api/recording")
def recording_upload():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No recording uploaded."}), 400
    token = path = None
    try:
        token, path = _save_upload(file)
        metadata = recording_metadata(path)
        return jsonify({"recording_id": token, "filename": file.filename, "format": metadata["source_format"], "sampling_rate_hz": metadata["sampling_rate_hz"], "duration_s": metadata["duration_s"], "time_start_s": metadata["time_start_s"], "time_end_s": metadata["time_end_s"], "n_samples": metadata["n_samples"], "channels": metadata["channels"]})
    except Exception as error:
        if path and path.is_file():
            path.unlink(missing_ok=True)
        if token and (UPLOAD_ROOT / token).is_dir():
            for child in sorted((UPLOAD_ROOT / token).rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            (UPLOAD_ROOT / token).rmdir()
        return jsonify({"error": str(error)}), 400


@app.get("/api/recording/<recording_id>/window")
def recording_window(recording_id):
    if not re.fullmatch(r"[0-9a-f]{32}", recording_id):
        return jsonify({"error": "invalid recording id"}), 400
    try:
        candidates = [UPLOAD_ROOT / recording_id, *UPLOAD_ROOT.glob(recording_id + ".*")]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None and (UPLOAD_ROOT / recording_id).is_dir():
            headers = list((UPLOAD_ROOT / recording_id).rglob("*.hea"))
            path = headers[0].with_suffix("") if len(headers) == 1 else None
        if path is None:
            return jsonify({"error": "recording not found"}), 404
        start = int(request.args.get("start", 0))
        stop = int(request.args.get("stop", start + 5000))
        window = read_recording_window(path, start, stop)
        return jsonify({"start": window["start"], "stop": window["stop"], "n_samples": window["n_samples"], "total_samples": window.get("total_samples"), "sampling_rate_hz": window["sampling_rate_hz"], "time_start_s": window["time_start_s"], "time_end_s": window["time_end_s"], "time": window["time"].tolist(), "signals": {key: np.asarray(value).tolist() for key, value in window["signals"].items()}})
    except (TypeError, ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/filter")
def filter_signal():
    data = _json()
    try:
        filtered = apply_pipeline(np.asarray(data["signal"], float), float(data["sampling_rate_hz"]), baseline=bool(data.get("baseline")), highpass_hz=float(data["highpass_hz"]) if data.get("highpass_hz") is not None else None, lowpass_hz=float(data["lowpass_hz"]) if data.get("lowpass_hz") is not None else None, notch_hz=float(data["notch_hz"]) if data.get("notch_hz") is not None else None)
        return jsonify({"signal": filtered.tolist()})
    except (KeyError, TypeError, ValueError, FilterConfigurationError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/detect/r-peaks")
def detect_r_peaks():
    data = _json()
    try:
        fs = float(data["sampling_rate_hz"])
        signal = np.asarray(data["signal"], float)
        min_distance_ms = float(data.get("min_distance_ms", 250))
        prominence_factor = float(data.get("prominence_factor", 0.5))
        if signal.ndim != 1 or len(signal) < 8 or not np.isfinite(signal).all():
            raise ValueError("signal must be one-dimensional, contain at least eight finite samples, and have no NaN or infinite values")
        if not np.isfinite(fs) or fs <= 0 or min_distance_ms <= 0 or prominence_factor < 0:
            raise ValueError("invalid peak detector settings")
        centered = signal - np.median(signal)
        scale = float(np.std(centered)) or 1.0
        peaks, properties = sps.find_peaks(centered, distance=max(1, int(round(fs * min_distance_ms / 1000))), prominence=scale * prominence_factor)
        return jsonify({"peaks": peaks.astype(int).tolist(), "prominences": properties.get("prominences", np.zeros(len(peaks))).tolist()})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/beats")
@app.post("/api/segment")
def beats():
    data = _json()
    try:
        time_axis = np.asarray(data["time"], float)
        if time_axis.ndim != 1 or len(time_axis) < 2 or not np.isfinite(time_axis).all() or np.any(np.diff(time_axis) <= 0):
            raise ValueError("time must be one-dimensional, finite, and strictly increasing")
        peaks = np.asarray(data.get("peaks") or [], int)
        if len(peaks) == 0 and data.get("signal") is not None:
            fs = float(data["sampling_rate_hz"])
            signal = np.asarray(data["signal"], float)
            if signal.ndim != 1 or len(signal) != len(time_axis) or not np.isfinite(signal).all():
                raise ValueError("signal must match time length and contain only finite samples")
            centered = signal - np.median(signal)
            scale = float(np.std(centered)) or 1.0
            peaks, _ = sps.find_peaks(centered, distance=max(1, int(round(fs * 0.25))), prominence=scale * 0.5)
        beat_windows = segment_beats(time_axis, peaks, float(data.get("pre_s", 0.35)), float(data.get("post_s", 0.55)))
        return jsonify({"beats": [beat.to_dict() for beat in beat_windows], "peaks": peaks.tolist(), "n_beats": len(beat_windows)})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/phenotype")
def phenotype():
    try:
        data = _json()
        phenotypes = beat_phenotypes(np.asarray(data["time"], float), np.asarray(data["signal"], float), np.asarray(data["r_indices"], int))
        return jsonify({"beats": phenotypes, "summary": summary_statistics(phenotypes)})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/statistics/compare")
def stats_compare():
    try:
        data = _json()
        return jsonify(compare_groups(data["group_a"], data["group_b"], data.get("unit_ids_a"), data.get("unit_ids_b")))
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/statistics/fdr")
def stats_fdr():
    try:
        return jsonify({"adjusted_p": benjamini_hochberg(_json()["p_values"])})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/benchmark")
def benchmark():
    try:
        data = _json()
        return jsonify(benchmark_models(np.asarray(data["X"], float), np.asarray(data["y"]), np.asarray(data["groups"]), int(data.get("folds", 5))))
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/ml/train")
def ml_train():
    try:
        data = _json()
        model, metrics = train_classifier(np.asarray(data["signal"], float), float(data["sampling_rate_hz"]), np.asarray(data["peaks"], int), list(data.get("annotations", [])), time=np.asarray(data["time"], float) if data.get("time") is not None else None)
        return jsonify({"trained": True, "metrics": metrics, "model": model.__class__.__name__})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"trained": False, "error": str(error)}), 400


@app.post("/api/ml/suggest")
def ml_suggest():
    try:
        data = _json()
        signal = np.asarray(data["signal"], float)
        fs = float(data["sampling_rate_hz"])
        peaks = np.asarray(data["peaks"], int)
        time_axis = np.asarray(data["time"], float) if data.get("time") is not None else None
        annotations = list(data.get("annotations", []))
        model, metrics = train_classifier(signal, fs, peaks, annotations, time=time_axis)
        suggestions = rank_uncertain(signal, fs, peaks, model, int(data.get("top_n", 20)), time=time_axis, annotations=annotations, min_spacing_s=float(data.get("min_spacing_s", 0.25)))
        return jsonify({"trained": True, "metrics": metrics, "suggestions": suggestions})
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"trained": False, "error": str(error)}), 400


@app.get("/api/project")
def project_get():
    try:
        return jsonify(_project_store(request.args.get("name", "default")).load().to_dict())
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/project/recording")
def project_recording():
    try:
        data = _json()
        store = _project_store(data.get("project", "default"))
        reference = RecordingRef(recording_id=str(data["recording_id"]), subject_id=str(data["subject_id"]), group=str(data.get("group", "")), visit=str(data.get("visit", "")), source_path=str(data.get("source_path", "")), format=str(data.get("format", "")), sampling_rate_hz=data.get("sampling_rate_hz"), duration_s=data.get("duration_s"), channels=list(data.get("channels", [])), metadata=dict(data.get("metadata", {})))
        return jsonify(store.add_recording(reference).to_dict())
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/<path:path>")
def static_file(path):
    return send_from_directory(str(WEB), path)


def run(host: str | None = None, port: int | None = None) -> None:
    host = host or os.getenv("ELECTROTRACE_HOST", "127.0.0.1")
    port = int(port or os.getenv("ELECTROTRACE_PORT", "5000"))
    validate_bind(host, API_KEY)
    app.run(host=host, port=port, debug=False)
