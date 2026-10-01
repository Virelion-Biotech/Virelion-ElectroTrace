"""Calibration-grade electrical observation handoff for CardiEP/CardiInfer.

This module converts ElectroTrace measurements into small, provenance-linked
artifacts intended for inverse-model calibration.  It deliberately separates
measurement preparation from the EP forward model and inference engine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .detectors import get_detector
from .io import Recording, load_recording
from .qrs_delineation import DELINEATOR_VERSION, delineate_qrs

CALIBRATION_SCHEMA_VERSION = "electrotrace-ep-calibration-v1"
SUPPORTED_MAP_KINDS = {"eam_activation", "activation_map", "repolarization_map"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_sigma(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def _channel(record: Recording, channel: int | str) -> tuple[int, str, np.ndarray]:
    names = list(record.signals)
    if isinstance(channel, str) and not channel.strip().lstrip("+-").isdigit():
        if channel not in record.signals:
            raise ValueError(f"Unknown calibration channel {channel!r}; available: {names}")
        index = names.index(channel)
    else:
        index = int(channel)
        if index < 0 or index >= len(names):
            raise ValueError(
                f"Calibration channel {channel!r} is out of range; available: {names}"
            )
    name = names[index]
    return index, name, np.asarray(record.signals[name], dtype=float)


def _peak_indices(
    signal: np.ndarray,
    fs_hz: float,
    *,
    detector: str,
    polarity: str,
    scale_method: str,
    supplied: Sequence[int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if supplied is not None:
        peaks = np.asarray(list(supplied), dtype=int)
        peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(signal))])
        return peaks, {
            "name": "supplied",
            "version": "external",
            "polarity": None,
            "scale_method": None,
        }
    spec = get_detector(detector, polarity=polarity, scale_method=scale_method)
    peaks = np.asarray(spec.detector(signal, fs_hz), dtype=int)
    peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(signal))])
    return peaks, {
        "name": spec.name,
        "version": spec.version,
        "polarity": polarity,
        "scale_method": scale_method,
        "citation": spec.citation,
    }


def build_ecg_calibration_bundle(
    record: Recording,
    *,
    primary_channel: int | str = 0,
    detector: str = "stage1",
    polarity: str = "adaptive",
    scale_method: str = "windowed_std",
    r_indices: Sequence[int] | None = None,
    pre_s: float = 0.25,
    post_s: float = 0.45,
    max_beats: int = 256,
    min_beats: int = 3,
) -> dict[str, Any]:
    """Build an aligned multi-lead median-beat observation with uncertainty.

    The exported waveform is a measured beat template, not a clinical
    interpretation.  Lead-quality values are transparent SNR-like heuristics
    intended for weighting/QC, not calibrated probabilities.
    """
    fs = float(record.sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("recording sampling_rate_hz must be positive and finite")
    if pre_s <= 0 or post_s <= 0:
        raise ValueError("pre_s and post_s must be positive")
    if max_beats < 1 or min_beats < 1:
        raise ValueError("max_beats and min_beats must be >= 1")

    channel_index, channel_name, primary = _channel(record, primary_channel)
    lead_names = list(record.signals)
    arrays = [np.asarray(record.signals[name], dtype=float) for name in lead_names]
    n_samples = len(primary)
    if any(arr.ndim != 1 or len(arr) != n_samples for arr in arrays):
        raise ValueError("all ECG channels must be one-dimensional and have equal length")
    if any(not np.isfinite(arr).all() for arr in arrays):
        raise ValueError("ECG channels must contain only finite values")

    peaks, detector_meta = _peak_indices(
        primary,
        fs,
        detector=detector,
        polarity=polarity,
        scale_method=scale_method,
        supplied=r_indices,
    )
    pre_n = max(1, int(round(pre_s * fs)))
    post_n = max(1, int(round(post_s * fs)))
    usable = peaks[(peaks - pre_n >= 0) & (peaks + post_n < n_samples)]
    if usable.size > max_beats:
        positions = np.linspace(0, usable.size - 1, max_beats).round().astype(int)
        usable = usable[np.unique(positions)]
    if usable.size < min_beats:
        raise ValueError(
            f"Need at least {min_beats} complete beats for calibration; found {usable.size}"
        )

    beat_stack = np.stack(
        [
            np.stack(
                [arr[int(peak) - pre_n : int(peak) + post_n + 1] for arr in arrays],
                axis=0,
            )
            for peak in usable
        ],
        axis=0,
    )
    template = np.median(beat_stack, axis=0)
    relative_time = (np.arange(template.shape[1], dtype=float) - pre_n) / fs

    residual = beat_stack - template[None, :, :]
    lead_quality: dict[str, dict[str, float]] = {}
    noise_sigma: dict[str, float] = {}
    for i, name in enumerate(lead_names):
        sigma = _robust_sigma(residual[:, i, :])
        amplitude = float(np.ptp(template[i]))
        snr_proxy = amplitude / max(2.0 * sigma, np.finfo(float).eps)
        noise_sigma[name] = sigma
        lead_quality[name] = {
            "residual_sigma": sigma,
            "template_peak_to_peak": amplitude,
            "snr_proxy": float(snr_proxy),
            "quality_score": float(snr_proxy / (1.0 + snr_proxy)),
        }

    qrs_rows: list[dict[str, float]] = []
    for peak in usable:
        try:
            boundary = delineate_qrs(primary, fs, int(peak))
        except Exception:
            continue
        if not boundary.onset_found or not boundary.offset_found:
            continue
        onset_ms = float((boundary.onset - int(peak)) * 1000.0 / fs)
        offset_ms = float((boundary.offset - int(peak)) * 1000.0 / fs)
        qrs_rows.append(
            {
                "onset_relative_ms": onset_ms,
                "offset_relative_ms": offset_ms,
                "duration_ms": offset_ms - onset_ms,
            }
        )

    qrs_summary: dict[str, Any] = {
        "delineator": DELINEATOR_VERSION,
        "n_valid": len(qrs_rows),
        "median_onset_relative_ms": None,
        "median_offset_relative_ms": None,
        "median_duration_ms": None,
        "duration_robust_sigma_ms": None,
    }
    if qrs_rows:
        onset = np.asarray([row["onset_relative_ms"] for row in qrs_rows])
        offset = np.asarray([row["offset_relative_ms"] for row in qrs_rows])
        widths = np.asarray([row["duration_ms"] for row in qrs_rows])
        qrs_summary.update(
            median_onset_relative_ms=float(np.median(onset)),
            median_offset_relative_ms=float(np.median(offset)),
            median_duration_ms=float(np.median(widths)),
            duration_robust_sigma_ms=_robust_sigma(widths),
        )

    rr = np.diff(usable.astype(float)) / fs
    rr_summary = {
        "n_intervals": int(rr.size),
        "median_s": float(np.median(rr)) if rr.size else None,
        "robust_sigma_s": _robust_sigma(rr) if rr.size else None,
    }

    units = dict(record.channel_units or {})
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "ecg",
        "representation": "median_beat_template",
        "sampling_rate_hz": fs,
        "relative_time_s": relative_time.tolist(),
        "lead_names": lead_names,
        "lead_units": {name: units.get(name) or None for name in lead_names},
        "primary_channel": {"index": channel_index, "name": channel_name},
        "beat_template": {
            name: template[i].astype(float).tolist() for i, name in enumerate(lead_names)
        },
        "n_detected_beats": int(peaks.size),
        "n_template_beats": int(usable.size),
        "template_peak_indices": usable.astype(int).tolist(),
        "detector": detector_meta,
        "qrs": qrs_summary,
        "rr": rr_summary,
        "uncertainty": {
            "method": "median-template residual MAD",
            "noise_sigma_by_lead": noise_sigma,
        },
        "quality": {
            "method": "heuristic residual-SNR proxy",
            "lead_quality": lead_quality,
        },
        "preprocessing": {
            "beat_window_pre_s": float(pre_s),
            "beat_window_post_s": float(post_s),
            "aggregation": "pointwise_median",
            "resampling": "none",
        },
    }


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def prepare_ecg_calibration(
    input_path: str | Path,
    *,
    entity_id: str,
    output_dir: str | Path,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(params or {})
    source = Path(input_path)
    record = load_recording(source)
    bundle = build_ecg_calibration_bundle(
        record,
        primary_channel=params.get("channel", params.get("primary_channel", 0)),
        detector=str(params.get("detector", "stage1")),
        polarity=str(params.get("polarity", "adaptive")),
        scale_method=str(params.get("scale_method", "windowed_std")),
        r_indices=params.get("r_indices"),
        pre_s=float(params.get("pre_s", 0.25)),
        post_s=float(params.get("post_s", 0.45)),
        max_beats=int(params.get("max_beats", 256)),
        min_beats=int(params.get("min_beats", 3)),
    )
    bundle["source"] = {
        "filename": source.name,
        "sha256": _sha256_file(source),
        "source_format": record.source_format,
    }

    out_dir = Path(output_dir)
    observation_id = str(
        params.get("observation_id") or f"{entity_id}-electrotrace-ecg-calibration"
    )
    artifact_path = out_dir / f"{observation_id}.json"
    digest, byte_count = _write_json_artifact(artifact_path, bundle)

    lead_units = {value for value in bundle["lead_units"].values() if value}
    units = str(params.get("units")) if params.get("units") else (
        next(iter(lead_units)) if len(lead_units) == 1 else None
    )
    artifact_id = f"electrotrace-{digest[:20]}"
    observation = {
        "observation_id": observation_id,
        "kind": "ecg",
        "artifact": {
            "artifact_id": artifact_id,
            "kind": "electrotrace_ecg_calibration",
            "uri": artifact_path.resolve().as_uri(),
            "sha256": digest,
            "metadata": {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "representation": bundle["representation"],
                "sampling_rate_hz": bundle["sampling_rate_hz"],
                "lead_names": bundle["lead_names"],
                "primary_channel": bundle["primary_channel"],
                "n_template_beats": bundle["n_template_beats"],
                "qrs": bundle["qrs"],
                "rr": bundle["rr"],
                "uncertainty": bundle["uncertainty"],
                "quality": bundle["quality"],
                "source_sha256": bundle["source"]["sha256"],
                "byte_count": byte_count,
            },
        },
        "coordinate_frame": str(params.get("coordinate_frame", "clinical_ecg")),
        "units": units,
        "acquired_at": params.get("acquired_at"),
    }

    hints = [
        {
            "term_id": f"{observation_id}:morphology",
            "model_output": "ecg",
            "discrepancy": "correlation",
            "weight": float(params.get("ecg_morphology_weight", 1.0)),
            "noise_parameters": {},
            "metadata": {
                "artifact_field": "beat_template",
                "alignment": "r_peak_relative",
                "scale_sensitive": False,
            },
        }
    ]
    qrs_duration = bundle["qrs"].get("median_duration_ms")
    if qrs_duration is not None:
        sigma_ms = bundle["qrs"].get("duration_robust_sigma_ms")
        hints.append(
            {
                "term_id": f"{observation_id}:qrs_duration",
                "model_output": "qrs_duration_ms",
                "discrepancy": "gaussian",
                "weight": float(params.get("qrs_weight", 0.25)),
                "noise_parameters": {
                    "sigma_ms": max(float(sigma_ms or 0.0), 1.0)
                },
                "metadata": {
                    "artifact_field": "qrs.median_duration_ms",
                    "observed_value_ms": float(qrs_duration),
                },
            }
        )

    return {
        "contract_version": "1.0",
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "entity_id": entity_id,
        "observations": [observation],
        "likelihood_hints": hints,
        "summary": {
            "kind": "ecg",
            "artifact_id": artifact_id,
            "artifact_uri": observation["artifact"]["uri"],
            "n_template_beats": bundle["n_template_beats"],
            "lead_names": bundle["lead_names"],
            "qrs_median_duration_ms": qrs_duration,
        },
    }


def register_map_calibration(
    input_path: str | Path,
    *,
    entity_id: str,
    kind: str,
    coordinate_frame: str,
    units: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a pre-aligned EAM/activation/repolarization artifact for calibration.

    ElectroTrace does not invent spatial registration.  The caller must provide
    the coordinate frame and a calibration-ready map file.
    """
    params = dict(params or {})
    normalized = str(kind)
    if normalized not in SUPPORTED_MAP_KINDS:
        raise ValueError(
            f"Unsupported map observation kind {kind!r}; expected one of "
            f"{sorted(SUPPORTED_MAP_KINDS)}"
        )
    if not str(coordinate_frame).strip():
        raise ValueError("coordinate_frame is required for map calibration observations")
    if not str(units).strip():
        raise ValueError("units is required for map calibration observations")

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _sha256_file(source)
    observation_id = str(
        params.get("observation_id") or f"{entity_id}-electrotrace-{normalized}"
    )
    artifact = {
        "artifact_id": f"electrotrace-{digest[:20]}",
        "kind": f"electrotrace_{normalized}",
        "uri": source.resolve().as_uri(),
        "sha256": digest,
        "metadata": {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "registration_status": "caller_declared",
            "source_filename": source.name,
            **dict(params.get("artifact_metadata") or {}),
        },
    }
    model_output = "activation_map" if normalized in {"eam_activation", "activation_map"} else "repolarization_map"
    hint = {
        "term_id": f"{observation_id}:map",
        "model_output": model_output,
        "discrepancy": str(params.get("discrepancy", "student_t")),
        "weight": float(params.get("weight", 2.0)),
        "noise_parameters": dict(params.get("noise_parameters") or {}),
        "metadata": {
            "coordinate_frame": coordinate_frame,
            "registration_status": "caller_declared",
        },
    }
    return {
        "contract_version": "1.0",
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "entity_id": entity_id,
        "observations": [
            {
                "observation_id": observation_id,
                "kind": normalized,
                "artifact": artifact,
                "coordinate_frame": coordinate_frame,
                "units": units,
                "acquired_at": params.get("acquired_at"),
            }
        ],
        "likelihood_hints": [hint],
        "summary": {
            "kind": normalized,
            "artifact_id": artifact["artifact_id"],
            "artifact_uri": artifact["uri"],
        },
    }


def prepare_calibration(
    input_path: str | Path,
    *,
    entity_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one electrical observation for downstream EP calibration."""
    params = dict(params or {})
    kind = str(params.get("observation_kind", "ecg"))
    if kind == "ecg":
        default_dir = Path(input_path).parent / f"{Path(input_path).stem}.electrotrace-calibration"
        return prepare_ecg_calibration(
            input_path,
            entity_id=entity_id,
            output_dir=params.get("calibration_output_dir", default_dir),
            params=params,
        )
    if kind in SUPPORTED_MAP_KINDS:
        return register_map_calibration(
            input_path,
            entity_id=entity_id,
            kind=kind,
            coordinate_frame=str(params.get("coordinate_frame") or ""),
            units=str(params.get("units") or ""),
            params=params,
        )
    raise ValueError(
        f"Unsupported observation_kind {kind!r}; expected 'ecg' or one of "
        f"{sorted(SUPPORTED_MAP_KINDS)}"
    )
