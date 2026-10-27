"""Versioned ECG measurement bundle for patient-calibration workflows.

This module preserves decoded waveforms and beat-level observations. It does not
claim that heuristic QC scores are probabilities or that exploratory P/ST/T
proxies are clinically validated measurements.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .detectors import get_detector
from .io import Recording
from .qrs_delineation import DELINEATOR_VERSION, delineate_qrs

ECG_MEASUREMENT_SCHEMA_VERSION = "electrotrace-ecg-measurement-v2"
RHYTHM_SCREEN_VERSION = "rr-variability-screen-v1"
MORPHOLOGY_PROXY_VERSION = "p-st-t-window-proxy-v1"
NOISE_ESTIMATOR_VERSION = "pre-qrs-difference-mad-v1"


def _robust_sigma(values: np.ndarray) -> float | None:
    """Return MAD-derived scale, or None when no finite samples exist."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    median = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - median)))


def _normalise_voltage_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    raw = str(unit).strip()
    key = raw.casefold().replace("μ", "µ")
    aliases = {
        "v": "V", "volt": "V", "volts": "V",
        "mv": "mV", "millivolt": "mV", "millivolts": "mV",
        "uv": "µV", "µv": "µV", "microvolt": "µV", "microvolts": "µV",
    }
    return aliases.get(key)


def _resolve_primary_channel(record: Recording, primary_channel: int | str) -> tuple[int, str]:
    names = list(record.signals)
    if isinstance(primary_channel, str) and primary_channel not in names:
        if not primary_channel.strip().lstrip("+-").isdigit():
            raise ValueError(f"Unknown primary ECG lead {primary_channel!r}; available leads: {names}")
    index = names.index(primary_channel) if isinstance(primary_channel, str) and primary_channel in names else int(primary_channel)
    if index < 0 or index >= len(names):
        raise ValueError(f"Primary ECG channel {primary_channel!r} is out of range; available leads: {names}")
    return index, names[index]


def _detect_peaks(
    signal: np.ndarray,
    fs_hz: float,
    *,
    detector: str,
    polarity: str,
    scale_method: str,
    supplied: Sequence[int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if supplied is not None:
        raw = np.asarray(list(supplied), dtype=int)
        peaks = np.unique(raw[(raw >= 0) & (raw < len(signal))])
        return peaks, {
            "name": "supplied",
            "version": "caller-supplied",
            "polarity": None,
            "scale_method": None,
        }
    spec = get_detector(detector, polarity=polarity, scale_method=scale_method)
    raw = np.asarray(spec.detector(signal, fs_hz), dtype=int)
    peaks = np.unique(raw[(raw >= 0) & (raw < len(signal))])
    return peaks, {
        "name": spec.name,
        "version": spec.version,
        "polarity": polarity,
        "scale_method": scale_method,
        "citation": spec.citation,
    }


def _annotation_for_peak(annotations: Mapping[Any, Any] | None, peak: int) -> dict[str, Any]:
    if not annotations:
        return {}
    value = annotations.get(peak, annotations.get(str(peak), {}))
    return dict(value) if isinstance(value, Mapping) else {}


def _source_units(
    record: Recording, params: Mapping[str, Any], lead_names: list[str]
) -> tuple[dict[str, str | None], dict[str, str]]:
    existing: dict[str, Any] = dict(record.channel_units or {})
    metadata_units = (record.metadata or {}).get("units")
    metadata_names: set[str] = set(record.channel_units or {})
    if isinstance(metadata_units, Mapping):
        for name, unit in metadata_units.items():
            existing.setdefault(str(name), unit)
            metadata_names.add(str(name))
    elif isinstance(metadata_units, (list, tuple)) and len(metadata_units) == len(record.signals):
        for name, unit in zip(list(record.signals), metadata_units):
            existing.setdefault(name, unit)
            metadata_names.add(name)

    explicit = params.get("lead_units")
    explicit_names: set[str] = set()
    if isinstance(explicit, Mapping):
        for name, unit in explicit.items():
            if name in lead_names and unit is not None and str(unit).strip():
                existing[str(name)] = str(unit).strip()
                explicit_names.add(str(name))
    global_unit = params.get("voltage_unit", params.get("units"))
    units: dict[str, str | None] = {}
    unit_source: dict[str, str] = {}
    for name in lead_names:
        if name in explicit_names:
            units[name] = str(existing[name]).strip()
            unit_source[name] = "caller_declared"
        elif existing.get(name) is not None and str(existing[name]).strip():
            units[name] = str(existing[name]).strip()
            unit_source[name] = "recording_metadata" if name in metadata_names else "caller_declared"
        elif global_unit is not None and str(global_unit).strip():
            units[name] = str(global_unit).strip()
            unit_source[name] = "caller_declared"
        else:
            units[name] = None
            unit_source[name] = "unknown"
    return units, unit_source


def _morphology_proxies(
    signal: np.ndarray,
    fs: float,
    qrs_onset: int,
    qrs_offset: int,
    next_r: int | None,
) -> dict[str, Any]:
    """Create explicitly exploratory P/ST/T window measurements.

    These simple windows are measurement proxies, not validated delineations.
    Values are withheld when windows are truncated or overlap a nearby beat.
    """
    x = np.asarray(signal, dtype=float)
    baseline_lo = max(0, qrs_onset - int(round(0.20 * fs)))
    baseline_hi = max(baseline_lo, qrs_onset - int(round(0.10 * fs)))
    if baseline_hi - baseline_lo < max(3, int(round(0.03 * fs))):
        return {
            "algorithm": MORPHOLOGY_PROXY_VERSION,
            "status": "not_estimable",
            "reason": "insufficient_pre_qrs_baseline",
            "validation_status": "not_independently_validated",
        }
    baseline = float(np.median(x[baseline_lo:baseline_hi]))
    p_lo = max(0, qrs_onset - int(round(0.28 * fs)))
    p_hi = max(p_lo, qrs_onset - int(round(0.06 * fs)))
    if p_hi - p_lo >= max(3, int(round(0.04 * fs))):
        p_idx = p_lo + int(np.argmax(np.abs(x[p_lo:p_hi] - baseline)))
        p_measurement = {
            "peak_time_s_offset_from_record_start": float(p_idx / fs),
            "peak_sample_index": int(p_idx),
            "amplitude_relative_to_pre_qrs_baseline": float(x[p_idx] - baseline),
            "window_s_relative_to_qrs_onset": [-0.28, -0.06],
            "status": "exploratory_proxy",
        }
    else:
        p_measurement = {"status": "not_estimable", "reason": "p_window_too_short"}

    st_idx = qrs_offset + int(round(0.08 * fs))
    st_limit = (next_r - int(round(0.08 * fs))) if next_r is not None else len(x) - 1
    if st_idx < len(x) and st_idx < st_limit:
        st_measurement = {
            "sample_index": int(st_idx),
            "time_s_offset_from_record_start": float(st_idx / fs),
            "amplitude_relative_to_pre_qrs_baseline": float(x[st_idx] - baseline),
            "sample_delay_after_qrs_offset_ms": 80.0,
            "status": "exploratory_proxy",
        }
    else:
        st_measurement = {"status": "not_estimable", "reason": "st_sample_overlaps_next_beat_or_recording_end"}

    t_lo = qrs_offset + int(round(0.10 * fs))
    t_hi = min(len(x), qrs_offset + int(round(0.40 * fs)), st_limit)
    if t_hi - t_lo >= max(4, int(round(0.05 * fs))):
        t_idx = t_lo + int(np.argmax(np.abs(x[t_lo:t_hi] - baseline)))
        t_measurement = {
            "peak_time_s_offset_from_record_start": float(t_idx / fs),
            "peak_sample_index": int(t_idx),
            "amplitude_relative_to_pre_qrs_baseline": float(x[t_idx] - baseline),
            "window_s_after_qrs_offset": [0.10, min(0.40, (t_hi - qrs_offset) / fs)],
            "status": "exploratory_proxy",
        }
    else:
        t_measurement = {"status": "not_estimable", "reason": "t_window_too_short_or_overlaps_next_beat"}

    return {
        "algorithm": MORPHOLOGY_PROXY_VERSION,
        "isoelectric_reference": {
            "median_amplitude": baseline,
            "unit": "same_as_lead",
            "window_s_before_qrs_onset": [-0.20, -0.10],
        },
        "p_wave": p_measurement,
        "st_level": st_measurement,
        "t_wave": t_measurement,
        "validation_status": "exploratory_proxy_not_independently_validated",
    }


def build_ecg_measurement_bundle(
    record: Recording,
    *,
    primary_channel: int | str = 0,
    detector: str = "stage1",
    polarity: str = "adaptive",
    scale_method: str = "windowed_std",
    r_indices: Sequence[int] | None = None,
    pre_s: float = 0.25,
    post_s: float = 0.45,
    max_beats: int | None = None,
    min_beats: int = 3,
    source_sha256: str | None = None,
    source_uri: str | None = None,
    source_filename: str | None = None,
    source_format: str | None = None,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    entity_id: str | None = None,
    lead_order: Sequence[str] | None = None,
    expected_leads: Sequence[str] | None = None,
    acquisition_metadata: Mapping[str, Any] | None = None,
    beat_annotations: Mapping[Any, Any] | None = None,
    preprocessing_steps: Sequence[Any] | None = None,
    irregular_rr_threshold: float = 0.20,
) -> dict[str, Any]:
    """Build a bundle preserving decoded raw samples and each selected beat.

    All sample arrays are retained in their source numeric scale; no filtering,
    resampling, unit conversion, or amplitude normalization is performed here.
    If metadata/annotations are absent, they remain explicitly unknown.
    """
    fs = float(record.sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("recording sampling_rate_hz must be positive and finite")
    if not np.isfinite(pre_s) or not np.isfinite(post_s) or pre_s <= 0 or post_s <= 0:
        raise ValueError("pre_s and post_s must be positive and finite")
    if max_beats is not None and (
        isinstance(max_beats, bool) or not isinstance(max_beats, (int, np.integer)) or max_beats < 1
    ):
        raise ValueError("max_beats must be None or a positive integer")
    if isinstance(min_beats, bool) or not isinstance(min_beats, (int, np.integer)) or min_beats < 1:
        raise ValueError("min_beats must be a positive integer")
    if not 0 <= irregular_rr_threshold <= 1:
        raise ValueError("irregular_rr_threshold must be between 0 and 1")

    observed_order = list(record.signals)
    if not observed_order:
        raise ValueError("recording must contain at least one ECG lead")
    time = np.asarray(record.time, dtype=float)
    arrays = {name: np.asarray(record.signals[name], dtype=float) for name in observed_order}
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("recording time axis must be finite, one-dimensional, and strictly increasing")
    n_samples = len(time)
    if any(values.ndim != 1 or len(values) != n_samples for values in arrays.values()):
        raise ValueError("all ECG leads must be one-dimensional and match the time axis")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("raw ECG values must be finite; repair or explicitly reject missing/non-finite segments before bundling")

    acquisition = dict(acquisition_metadata or {})
    source_metadata = dict(record.metadata or {})
    raw_device_metadata = acquisition.get("device_metadata") or source_metadata.get("device_metadata")
    if not raw_device_metadata and source_metadata.get("device") is not None:
        raw_device_metadata = {"device": source_metadata.get("device")}
    if isinstance(raw_device_metadata, Mapping):
        device_metadata = dict(raw_device_metadata)
    elif raw_device_metadata:
        device_metadata = {"description": str(raw_device_metadata)}
    else:
        device_metadata = {}
    raw_filter_metadata = acquisition.get("filter_metadata") or source_metadata.get("filter_metadata") or source_metadata.get("filters")
    if isinstance(raw_filter_metadata, Mapping):
        filter_metadata = dict(raw_filter_metadata)
    elif raw_filter_metadata:
        filter_metadata = {"description": str(raw_filter_metadata)}
    else:
        filter_metadata = {}
    primary_index, primary_name = _resolve_primary_channel(record, primary_channel)
    requested_order = [str(name) for name in (lead_order or observed_order)]
    if len(requested_order) != len(set(requested_order)) or any(not x.strip() for x in requested_order):
        raise ValueError("lead_order must contain unique non-empty lead names")
    ordered_leads = [name for name in requested_order if name in arrays]
    ordered_leads.extend(name for name in observed_order if name not in ordered_leads)
    expected = [str(name) for name in (expected_leads or ())]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_leads must be unique")
    missing_leads = [name for name in expected if name not in arrays]
    units, unit_sources = _source_units(record, acquisition, ordered_leads)
    normalized_units = {name: _normalise_voltage_unit(units.get(name)) for name in ordered_leads}

    time_diffs = np.diff(time)
    median_dt = float(np.median(time_diffs))
    time_fs = 1.0 / median_dt
    fs_relative_error = abs(time_fs - fs) / fs
    sampling_rate_agrees = bool(fs_relative_error <= 0.01)
    interval_relative_deviation = np.abs(time_diffs - median_dt) / median_dt
    irregular_sampling_fraction = float(np.mean(interval_relative_deviation > 0.01))
    irregular_sampling = irregular_sampling_fraction > 0.01

    peaks, detector_meta = _detect_peaks(
        arrays[primary_name], fs, detector=detector, polarity=polarity,
        scale_method=scale_method, supplied=r_indices
    )
    pre_n = max(1, int(round(pre_s * fs)))
    post_n = max(1, int(round(post_s * fs)))
    full_window_mask = (peaks - pre_n >= 0) & (peaks + post_n < n_samples)
    complete_peaks = peaks[full_window_mask]
    rejected_segments: list[dict[str, Any]] = [
        {"sample_index": int(peak), "time_s": float(time[int(peak)]),
         "reason": "beat_window_truncated_at_recording_boundary"}
        for peak in peaks[~full_window_mask]
    ]
    selected_peaks = complete_peaks
    if max_beats is not None and len(selected_peaks) > max_beats:
        chosen = np.unique(np.linspace(0, len(selected_peaks) - 1, max_beats).round().astype(int))
        chosen_peaks = selected_peaks[chosen]
        selected_set = set(map(int, chosen_peaks))
        rejected_segments.extend(
            {"sample_index": int(peak), "time_s": float(time[int(peak)]),
             "reason": "omitted_by_max_beats_sampling_cap"}
            for peak in selected_peaks if int(peak) not in selected_set
        )
        selected_peaks = chosen_peaks

    all_rr = np.diff(time[peaks]) if len(peaks) > 1 else np.asarray([], dtype=float)
    rr_median = float(np.median(all_rr)) if len(all_rr) else None
    rr_scale = _robust_sigma(all_rr)
    rr_variability_ratio = (float(rr_scale) / rr_median) if rr_scale is not None and rr_median and rr_median > 0 else None
    has_rr_outlier_interval = bool(
        rr_median is not None
        and rr_median > 0
        and len(all_rr) > 0
        and np.any(np.abs(all_rr - rr_median) / rr_median > irregular_rr_threshold)
    )
    rr_pattern = (
        "insufficient_beats" if rr_variability_ratio is None
        else "regular_rr_pattern"
        if rr_variability_ratio <= 0.05 and not has_rr_outlier_interval
        else "variable_rr_pattern"
    )
    rr_outlier_samples: set[int] = set()
    if rr_median is not None and rr_median > 0:
        for i, peak in enumerate(peaks):
            local_intervals = []
            if i > 0:
                local_intervals.append(float(time[int(peak)] - time[int(peaks[i - 1])]))
            if i + 1 < len(peaks):
                local_intervals.append(float(time[int(peaks[i + 1])] - time[int(peak)]))
            if any(abs(interval - rr_median) / rr_median > irregular_rr_threshold for interval in local_intervals):
                rr_outlier_samples.add(int(peak))

    relative_time = (np.arange(-pre_n, post_n + 1, dtype=float) / fs).tolist()
    annotations = beat_annotations or {}
    beat_records: list[dict[str, Any]] = []
    baseline_noise_by_lead: dict[str, float | None] = {}
    for name in ordered_leads:
        baseline_parts = []
        for peak in selected_peaks:
            left = int(peak) - int(round(0.20 * fs))
            right = int(peak) - int(round(0.08 * fs))
            if left >= 0 and right - left >= 4:
                baseline_parts.append(np.diff(arrays[name][left:right]))
        baseline_noise_by_lead[name] = (
            _robust_sigma(np.concatenate(baseline_parts)) / np.sqrt(2.0)
            if baseline_parts and _robust_sigma(np.concatenate(baseline_parts)) is not None
            else None
        )

    for peak_value in selected_peaks:
        peak = int(peak_value)
        start, stop = peak - pre_n, peak + post_n + 1
        annotation = _annotation_for_peak(annotations, peak)
        ectopic_raw = annotation.get("ectopic", annotation.get("is_ectopic"))
        paced_raw = annotation.get("paced", annotation.get("is_paced"))
        ectopic = bool(ectopic_raw) if isinstance(ectopic_raw, (bool, np.bool_)) else None
        paced = bool(paced_raw) if isinstance(paced_raw, (bool, np.bool_)) else None
        is_rr_outlier = peak in rr_outlier_samples
        waveform_by_lead = {name: arrays[name][start:stop].astype(float).tolist() for name in ordered_leads}
        peak_pos = int(np.searchsorted(peaks, peak))
        previous = int(peaks[peak_pos - 1]) if peak_pos > 0 else None
        following = int(peaks[peak_pos + 1]) if peak_pos + 1 < len(peaks) else None
        rr_prev = float(time[peak] - time[previous]) if previous is not None else None
        rr_next = float(time[following] - time[peak]) if following is not None else None
        quality_flags = []
        if is_rr_outlier:
            quality_flags.append("rr_interval_outlier_screen")
        if ectopic is True:
            quality_flags.append("adjudicated_ectopic")
        if paced is True:
            quality_flags.append("adjudicated_paced")
        if ectopic is None:
            quality_flags.append("ectopy_status_unknown")
        if paced is None:
            quality_flags.append("pacing_status_unknown")
        beat = {
            "beat_id": f"beat-{peak:08d}",
            "sample_index": peak,
            "timestamp_s_from_record_start": float(time[peak] - time[0]),
            "timestamp_s_on_source_time_axis": float(time[peak]),
            "r_peak_index": peak,
            "r_peak_time_s": float(time[peak]),
            "window_sample_range_half_open": [start, stop],
            "relative_time_s": relative_time,
            "waveform_by_lead_original_units": waveform_by_lead,
            "rr_previous_s": rr_prev,
            "rr_next_s": rr_next,
            "beat_annotation": {
                "ectopic": ectopic, "paced": paced,
                "rhythm_label": annotation.get("rhythm_label"),
                "source": annotation.get("source", "unannotated"),
                "adjudicated": bool(annotation.get("adjudicated", False)),
            },
            "rr_outlier_screen": {
                "flagged": is_rr_outlier,
                "threshold_fraction_from_median_rr": float(irregular_rr_threshold),
                "is_diagnostic": False,
            },
            "quality": {
                "status": "qualified" if is_rr_outlier or ectopic is True or paced is True else "unreviewed",
                "flags": quality_flags,
                "score_semantics": "flags_only_no_probability_claim",
            },
        }
        boundary = None
        try:
            if len(arrays[primary_name]) >= 32:
                boundary = delineate_qrs(arrays[primary_name], fs, peak)
        except (ValueError, FloatingPointError):
            boundary = None
        if boundary is None:
            beat["morphology"] = {"qrs": {"status": "not_estimable", "reason": "delineation_failed"}}
        else:
            onset = int(boundary.onset)
            offset = int(boundary.offset)
            duration_ms = float((offset - onset) * 1000.0 / fs)
            qrs = {
                "algorithm": DELINEATOR_VERSION,
                "onset_sample_index": onset, "offset_sample_index": offset,
                "onset_time_s": float(time[onset]), "offset_time_s": float(time[offset]),
                "duration_ms": duration_ms,
                "onset_found": bool(boundary.onset_found), "offset_found": bool(boundary.offset_found),
                "segmentation_uncertainty_ms": None,
                "segmentation_uncertainty_status": "not_calibrated_against_independent_manual_annotations",
            }
            beat["morphology"] = {
                "qrs": qrs,
                "single_primary_lead_proxies": _morphology_proxies(arrays[primary_name], fs, onset, offset, following),
            }
        beat_records.append(beat)

    template_candidates = []
    for peak in selected_peaks:
        annotation = _annotation_for_peak(annotations, int(peak))
        ectopic = annotation.get("ectopic", annotation.get("is_ectopic"))
        paced = annotation.get("paced", annotation.get("is_paced"))
        if ectopic is True or paced is True or int(peak) in rr_outlier_samples:
            continue
        template_candidates.append(int(peak))
    template_eligible = len(template_candidates) >= min_beats
    # When the stable subset is too small, keep a descriptive median but prohibit automatic fitting.
    template_peaks = template_candidates if template_eligible else list(map(int, selected_peaks))
    template: dict[str, list[float]] = {}
    beat_to_beat_variability: dict[str, dict[str, Any]] = {}
    residual_sigma_by_lead: dict[str, float | None] = {}
    lead_quality: dict[str, dict[str, Any]] = {}
    for name in ordered_leads:
        all_stack = (
            np.stack([arrays[name][int(peak) - pre_n:int(peak) + post_n + 1] for peak in selected_peaks], axis=0)
            if len(selected_peaks) else np.empty((0, len(relative_time)), dtype=float)
        )
        if len(all_stack):
            all_median = np.median(all_stack, axis=0)
            all_abs_dev = np.abs(all_stack - all_median[None, :])
            all_samplewise_mad = 1.4826 * np.median(all_abs_dev, axis=0)
            beat_to_beat_variability[name] = {
                "method": "1.4826 * pointwise median absolute deviation across all retained beat windows",
                "samplewise_mad": all_samplewise_mad.astype(float).tolist(),
                "median_samplewise_mad": float(np.median(all_samplewise_mad)),
                "units": units.get(name),
                "beat_population": "all selected complete beats, including flagged/annotated beats",
                "not_a_probability": True,
            }
        else:
            all_median = np.asarray([], dtype=float)
            beat_to_beat_variability[name] = {
                "method": "not_estimable_no_complete_beats", "samplewise_mad": [],
                "median_samplewise_mad": None, "units": units.get(name),
                "beat_population": "no complete beats", "not_a_probability": True,
            }

        template_stack = (
            np.stack([arrays[name][int(peak) - pre_n:int(peak) + post_n + 1] for peak in template_peaks], axis=0)
            if template_peaks else np.empty((0, len(relative_time)), dtype=float)
        )
        if len(template_stack):
            template_median = np.median(template_stack, axis=0)
            template[name] = template_median.astype(float).tolist()
            template_residual_scale = _robust_sigma(template_stack - template_median[None, :])
            residual_sigma_by_lead[name] = template_residual_scale
            amplitude = float(np.ptp(template_median))
        else:
            template_median = np.asarray([], dtype=float)
            template_residual_scale = None
            residual_sigma_by_lead[name] = None
            amplitude = 0.0
        noise = baseline_noise_by_lead.get(name)
        snr = amplitude / max(float(noise or 0.0), np.finfo(float).eps) if len(template_median) else None
        flags = []
        if normalized_units.get(name) is None:
            flags.append("voltage_unit_unknown_or_unrecognized")
        elif unit_sources.get(name) == "caller_declared":
            flags.append("voltage_unit_is_caller_declared_not_source_verified")
        if noise is None:
            flags.append("baseline_noise_not_estimable")
        if noise is not None and amplitude < 5.0 * noise:
            flags.append("low_template_amplitude_relative_to_baseline_noise")
        lead_quality[name] = {
            "baseline_noise_sigma_proxy": noise,
            "baseline_noise_method": NOISE_ESTIMATOR_VERSION,
            "beat_to_template_residual_scale": template_residual_scale,
            "template_peak_to_peak_original_units": amplitude,
            "heuristic_snr_ratio": float(snr) if snr is not None else None,
            "flags": flags,
            "score_semantics": "heuristic_ratio_not_a_probability_of_correctness",
        }

    qrs_rows = []
    for beat in beat_records:
        qrs = beat.get("morphology", {}).get("qrs", {})
        if qrs.get("onset_found") and qrs.get("offset_found"):
            peak = int(beat["r_peak_index"])
            qrs_rows.append({
                "beat_id": beat["beat_id"],
                "onset_relative_ms": float((qrs["onset_sample_index"] - peak) * 1000.0 / fs),
                "offset_relative_ms": float((qrs["offset_sample_index"] - peak) * 1000.0 / fs),
                "duration_ms": float(qrs["duration_ms"]),
            })
    qrs_durations = np.asarray([row["duration_ms"] for row in qrs_rows], dtype=float)
    qrs_summary = {
        "algorithm": DELINEATOR_VERSION, "n_valid": int(len(qrs_rows)),
        "median_onset_relative_ms": float(np.median([row["onset_relative_ms"] for row in qrs_rows])) if qrs_rows else None,
        "median_offset_relative_ms": float(np.median([row["offset_relative_ms"] for row in qrs_rows])) if qrs_rows else None,
        "median_duration_ms": float(np.median(qrs_durations)) if qrs_rows else None,
        "duration_robust_sigma_ms": _robust_sigma(qrs_durations) if qrs_rows else None,
        "segmentation_uncertainty_ms": None,
        "segmentation_uncertainty_status": "requires_independent_manual_adjudication",
        "beat_measurements": qrs_rows,
    }

    reasons: list[str] = []
    if len(selected_peaks) < min_beats:
        reasons.append("fewer_than_minimum_complete_beats")
    if not source_sha256 or len(str(source_sha256)) != 64:
        reasons.append("missing_or_invalid_immutable_source_sha256")
    if not source_uri and not source_sha256:
        reasons.append("missing_immutable_source_reference")
    if not patient_id:
        reasons.append("patient_linkage_missing")
    if not encounter_id:
        reasons.append("encounter_linkage_missing")
    if not device_metadata:
        reasons.append("device_metadata_missing")
    if not filter_metadata:
        reasons.append("filter_metadata_missing")
    if any(normalized_units.get(name) is None for name in ordered_leads):
        reasons.append("one_or_more_voltage_units_unknown_or_unrecognized")
    if not sampling_rate_agrees:
        reasons.append("sampling_rate_disagrees_with_time_axis")
    if irregular_sampling:
        reasons.append("irregular_sampling_intervals")
    if missing_leads:
        reasons.append("expected_leads_missing")
    if rr_pattern == "variable_rr_pattern":
        reasons.append("variable_rr_pattern_requires_rhythm_aware_fit_selection")
    if not template_eligible:
        reasons.append("insufficient_stable_beats_for_a_calibration_template")
    if any(lead_quality[name]["flags"] for name in ordered_leads):
        reasons.append("one_or_more_lead_quality_flags")
    status = "rejected" if len(selected_peaks) < min_beats else "qualified" if reasons else "provisional_pass"

    try:
        package_version = version("electrotrace")
    except PackageNotFoundError:
        package_version = "source-checkout"
    source_digest = str(source_sha256).lower() if source_sha256 else None
    if source_digest and (len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest)):
        raise ValueError("source_sha256 must be a 64-character hexadecimal SHA-256 digest")
    source_reference = {
        "uri": source_uri or (f"urn:sha256:{source_digest}" if source_digest else None),
        "uri_kind": "caller_supplied" if source_uri else "content_hash" if source_digest else "missing",
        "sha256": source_digest,
        "filename": source_filename,
        "format": source_format or record.source_format,
        "hash_semantics": "SHA-256 of original source file bytes; separate from decoded waveform arrays",
    }
    bundle = {
        "schema_version": ECG_MEASUREMENT_SCHEMA_VERSION,
        "kind": "ecg",
        "representation": "raw_waveform_plus_beat_level_measurements_and_robust_template",
        "acquisition": {
            "sampling_frequency_hz": fs,
            "sampling_frequency_inferred_from_time_axis_hz": float(time_fs),
            "sampling_frequency_relative_disagreement": float(fs_relative_error),
            "sampling_rate_agrees_with_time_axis_within_1_percent": sampling_rate_agrees,
            "physical_voltage_unit_by_lead": units,
            "normalized_voltage_unit_by_lead": normalized_units,
            "unit_provenance_by_lead": unit_sources,
            "lead_names_and_order": ordered_leads,
            "observed_source_lead_order": observed_order,
            "requested_lead_order": requested_order,
            "expected_leads": expected,
            "missing_expected_leads": missing_leads,
            "primary_lead": {"index_in_original_recording": primary_index, "name": primary_name},
            "sample_count": int(n_samples),
            "recording_duration_s": float(time[-1] - time[0]),
            "source_time_start_s": float(time[0]),
            "source_time_end_s": float(time[-1]),
            "device_metadata": device_metadata,
            "filter_metadata": filter_metadata,
            "source_metadata": source_metadata,
            "patient_id": patient_id, "encounter_id": encounter_id, "linked_entity_id": entity_id,
            "start_datetime": acquisition.get("start_datetime"), "timezone": acquisition.get("timezone"),
        },
        "raw_measurements": {
            "source_reference": source_reference,
            "sample_count": int(n_samples),
            "sample_time_s_on_source_time_axis": time.astype(float).tolist(),
            "samples_by_lead_original_units": {name: arrays[name].astype(float).tolist() for name in ordered_leads},
            "waveform_representation": "decoded numeric samples without filtering, resampling, normalization, or unit conversion",
        },
        "beat_analysis": {
            "r_peak_timestamps_s_on_source_time_axis": [float(time[int(p)]) for p in peaks],
            "r_peak_sample_indices": [int(p) for p in peaks],
            "n_detected_beats": int(len(peaks)),
            "n_complete_beats": int(len(complete_peaks)),
            "n_selected_beats_with_waveforms": int(len(selected_peaks)),
            "beat_window_s": {"pre_r": float(pre_s), "post_r": float(post_s)},
            "relative_time_s": relative_time,
            "beats": beat_records,
            "rr_intervals_s": all_rr.astype(float).tolist(),
            "rhythm_characterization": {
                "rr_interval_count": int(len(all_rr)), "median_rr_s": rr_median,
                "rr_robust_sigma_s": rr_scale, "robust_rr_variability_ratio": rr_variability_ratio,
                "pattern_screen": rr_pattern, "regularity_threshold": 0.05,
                "rr_outlier_threshold_fraction_from_median": float(irregular_rr_threshold),
                "adjudicated_ectopic_count": sum(1 for b in beat_records if b["beat_annotation"]["ectopic"] is True),
                "adjudicated_paced_count": sum(1 for b in beat_records if b["beat_annotation"]["paced"] is True),
                "ectopy_or_pacing_unclassified_count": sum(
                    1 for b in beat_records if b["beat_annotation"]["ectopic"] is None or b["beat_annotation"]["paced"] is None
                ),
                "interpretation": "descriptive screening only; not an automated rhythm diagnosis",
            },
        },
        "morphology": {
            "qrs": qrs_summary,
            "robust_template": {
                "aggregation": "pointwise_median", "relative_time_s": relative_time,
                "samples_by_lead_original_units": template, "n_template_beats": int(len(template_peaks)),
                "template_beat_sample_indices": [int(p) for p in template_peaks],
                "eligible_for_calibration_fit": bool(template_eligible),
                "eligibility_rule": "exclude adjudicated ectopic/paced beats and RR-outlier-screened beats, then require minimum beat count",
                "status": "eligible" if template_eligible else "descriptive_only_not_recommended_for_fit",
            },
            "beat_to_beat_variability_by_lead": beat_to_beat_variability,
            "p_st_t_measurements": {
                "algorithm": MORPHOLOGY_PROXY_VERSION, "measurement_status": "exploratory_per_beat_proxies",
                "primary_lead_only": primary_name, "validation_status": "not_independently_validated",
            },
        },
        "measurement_uncertainty": {
            "baseline_noise_sigma_proxy_by_lead": baseline_noise_by_lead,
            "baseline_noise_method": NOISE_ESTIMATOR_VERSION,
            "beat_to_template_residual_scale_by_lead": residual_sigma_by_lead,
            "qrs_segmentation_uncertainty_ms": None,
            "qrs_segmentation_uncertainty_status": "not_estimable_without_independent_manual_adjudication",
            "beat_to_beat_variability_by_lead": beat_to_beat_variability,
            "interpretation": "scale estimates and variability summaries are not probabilities; residual scale may include real physiological beat variability",
            "independent_adjudication_calibration": {
                "status": "not_run", "required_before_probabilistic_or_clinical_interpretation": True,
            },
        },
        "quality": {
            "status": status, "reasons": reasons, "lead_quality": lead_quality,
            "heuristic_method": "baseline high-frequency difference scale and beat-to-template summaries",
            "not_a_probability_of_measurement_correctness": True, "independent_adjudication_required": True,
        },
        "provenance": {
            "software": "electrotrace", "software_version": package_version,
            "schema_version": ECG_MEASUREMENT_SCHEMA_VERSION,
            "algorithm_versions": {
                "r_peak_detector": detector_meta, "qrs_delineator": DELINEATOR_VERSION,
                "rhythm_screen": RHYTHM_SCREEN_VERSION, "p_st_t_proxy": MORPHOLOGY_PROXY_VERSION,
                "baseline_noise_estimator": NOISE_ESTIMATOR_VERSION,
            },
            "detector_settings": {
                "requested_detector": detector, "polarity": polarity, "scale_method": scale_method,
                "primary_channel": primary_name, "supplied_r_indices": r_indices is not None,
            },
            "preprocessing_steps": list(preprocessing_steps or ["none; decoded source values retained unchanged"]),
            "preprocessing_applied_by_bundle_builder": [],
            "rejected_or_omitted_segments": rejected_segments,
            "source_sha256": source_digest,
            "unit_declarations": {"unit_source_by_lead": unit_sources, "original_units_preserved": True},
            "deterministic_serialization": "JSON sorted keys; no generation timestamp included",
        },
        "calibration_recommendation": {
            "template_fit_eligible": bool(template_eligible and len(selected_peaks) >= min_beats),
            "fit_beat_ids": [f"beat-{int(p):08d}" for p in template_peaks] if template_eligible else [],
            "excluded_from_template_sample_indices": [int(p) for p in selected_peaks if int(p) not in set(template_peaks)],
            "reason": "stable beat subset is retained separately; confirm units, lead mapping, linkage, and rhythm suitability before fitting",
        },
        "acceptance_gate": {
            "status": "not_demonstrated",
            "reason": "the independent, manually adjudicated frozen acceptance set has not been run by this function",
        },
    }
    # The flat fields are a deliberate read-compatibility layer for consumers of
    # electrotrace-ep-calibration-v1. New code should prefer the nested v2 sections.
    bundle.update({
        "sampling_rate_hz": fs,
        "relative_time_s": relative_time,
        "lead_names": ordered_leads,
        "lead_units": units,
        "primary_channel": {"index": primary_index, "name": primary_name},
        "beat_template": template,
        "n_detected_beats": int(len(peaks)),
        "n_template_beats": int(len(template_peaks)),
        "template_peak_indices": [int(p) for p in template_peaks],
        "detector": detector_meta,
        "qrs": qrs_summary,
        "rr": {
            "n_intervals": int(len(all_rr)),
            "median_s": rr_median,
            "robust_sigma_s": rr_scale,
        },
        "uncertainty": {
            "method": "empirical beat-to-template scale; not measurement-error probability",
            "noise_sigma_by_lead": residual_sigma_by_lead,
            "baseline_noise_sigma_by_lead": baseline_noise_by_lead,
        },
        "preprocessing": {
            "beat_window_pre_s": float(pre_s),
            "beat_window_post_s": float(post_s),
            "aggregation": "pointwise_median",
            "resampling": "none",
        },
    })
    bundle["quality"]["method"] = "heuristic baseline-noise and template-residual summaries; not a probability"
    validate_ecg_measurement_bundle(bundle)
    return bundle


def validate_ecg_measurement_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate core v2 invariants without claiming scientific validity."""
    if bundle.get("schema_version") != ECG_MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported ECG measurement schema: {bundle.get('schema_version')!r}")
    for section in ("acquisition", "raw_measurements", "beat_analysis", "morphology", "measurement_uncertainty", "provenance", "quality"):
        if not isinstance(bundle.get(section), Mapping):
            raise ValueError(f"ECG measurement bundle section {section!r} must be an object")
    acquisition = bundle["acquisition"]
    fs = acquisition.get("sampling_frequency_hz")
    if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_frequency_hz must be positive and finite")
    raw = bundle["raw_measurements"]
    times = np.asarray(raw.get("sample_time_s_on_source_time_axis", []), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("raw sample time axis must be finite and strictly increasing")
    leads = raw.get("samples_by_lead_original_units", {})
    if not isinstance(leads, Mapping) or not leads:
        raise ValueError("raw waveform must include at least one lead")
    for name, values in leads.items():
        signal = np.asarray(values, dtype=float)
        if signal.ndim != 1 or len(signal) != len(times) or not np.isfinite(signal).all():
            raise ValueError(f"raw waveform for lead {name!r} does not match the sample axis")
    for beat in bundle["beat_analysis"].get("beats", []):
        if not isinstance(beat.get("waveform_by_lead_original_units"), Mapping):
            raise ValueError("every selected beat must preserve per-lead waveforms")
        for name, values in beat["waveform_by_lead_original_units"].items():
            waveform = np.asarray(values, dtype=float)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError(f"beat waveform for lead {name!r} must be a finite vector")


def evaluate_uncertainty_calibration(
    predicted_uncertainty: Sequence[float],
    adjudicated_absolute_error: Sequence[float],
    *,
    adjudication_source: str,
    reference_split: str = "independent_test",
    independent_reference: bool = True,
    bins: int = 5,
    minimum_samples: int = 20,
) -> dict[str, Any]:
    """Measure correspondence between uncertainty scales and errors on manual labels.

    This reports empirical scale-to-error correspondence, not event correctness
    probabilities or confidence coverage. The caller must ensure patient/encounter
    separation and genuinely independent adjudication.
    """
    if not independent_reference:
        raise ValueError("uncertainty calibration requires an independent adjudicated reference subset")
    if reference_split not in {"independent_test", "test", "validation"}:
        raise ValueError("reference_split must identify an independent validation/test subset")
    if not str(adjudication_source).strip():
        raise ValueError("adjudication_source must identify the independent annotation source")
    if isinstance(bins, bool) or bins < 2 or isinstance(minimum_samples, bool) or minimum_samples < 2:
        raise ValueError("bins and minimum_samples must be integers >= 2")
    predicted = np.asarray(predicted_uncertainty, dtype=float)
    error = np.asarray(adjudicated_absolute_error, dtype=float)
    if predicted.ndim != 1 or error.ndim != 1 or len(predicted) != len(error) or len(predicted) < 2:
        raise ValueError("predicted uncertainty and adjudicated errors must be paired one-dimensional arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(error).all() or np.any(predicted < 0) or np.any(error < 0):
        raise ValueError("uncertainty scales and absolute errors must be finite and nonnegative")

    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    curve = []
    for index, members in enumerate(groups):
        if len(members) == 0:
            continue
        curve.append({
            "bin": index + 1, "n": int(len(members)),
            "median_predicted_uncertainty_scale": float(np.median(predicted[members])),
            "median_adjudicated_absolute_error": float(np.median(error[members])),
            "mean_adjudicated_absolute_error": float(np.mean(error[members])),
        })
    if len(np.unique(predicted)) > 1 and len(np.unique(error)) > 1:
        corr = float(spearmanr(predicted, error).statistic)
    else:
        corr = None
    binned_scale_mae = float(np.mean([
        abs(row["median_predicted_uncertainty_scale"] - row["median_adjudicated_absolute_error"])
        for row in curve
    ]))
    return {
        "method": "equal-count empirical predicted-scale versus adjudicated-absolute-error curve",
        "status": "empirical_reference_available" if len(predicted) >= minimum_samples else "insufficient_reference_data",
        "n_measurements": int(len(predicted)), "minimum_samples_for_status": int(minimum_samples),
        "adjudication_source": str(adjudication_source), "reference_split": reference_split,
        "independent_reference_declared_by_caller": True,
        "spearman_scale_error_correlation": corr,
        "binned_median_scale_absolute_error_difference": binned_scale_mae,
        "curve": curve,
        "interpretation": (
            "Empirical correspondence of an uncertainty scale to absolute measurement error. "
            "This is not a probability of correctness and does not establish clinical validity."
        ),
        "limitations": [
            "Independence and patient/encounter separation are caller-declared and must be audited.",
            "Freeze the scale-to-error mapping on an adjudicated calibration subset and evaluate it on a separate untouched subset before deployment.",
            "QRS segmentation and other morphology algorithms must be compared with expert annotations using prespecified tolerances.",
        ],
    }
