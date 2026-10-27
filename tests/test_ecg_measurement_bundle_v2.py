import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from electrotrace.calibration import CALIBRATION_SCHEMA_VERSION, prepare_ecg_calibration
from electrotrace.ecg_measurement import (
    ECG_MEASUREMENT_SCHEMA_VERSION,
    apply_uncertainty_calibration_map,
    build_ecg_measurement_bundle,
    evaluate_uncertainty_calibration,
    fit_uncertainty_calibration_map,
    validate_ecg_measurement_bundle,
)
from electrotrace.io import Recording


def _recording(fs=200.0, duration_s=10.0, noise=0.0, peaks=None, units=True):
    time = np.arange(int(fs * duration_s), dtype=float) / fs
    peaks = np.asarray(peaks if peaks is not None else [200, 400, 600, 800, 1000, 1200, 1400, 1600], dtype=int)
    rng = np.random.default_rng(231)
    lead_i = rng.normal(0.0, noise, len(time))
    lead_ii = rng.normal(0.0, noise, len(time))
    for peak in peaks:
        x = np.arange(len(time)) - peak
        lead_i += np.exp(-0.5 * (x / 2.5) ** 2)
        lead_ii += 0.7 * np.exp(-0.5 * ((x - 1.0) / 3.0) ** 2)
        lead_i += 0.12 * np.exp(-0.5 * ((x + 28.0) / 5.0) ** 2)
        lead_i += 0.25 * np.exp(-0.5 * ((x - 35.0) / 9.0) ** 2)
    return Recording(
        time=time,
        signals={"I": lead_i, "II": lead_ii},
        sampling_rate_hz=fs,
        source_format="synthetic",
        channel_units={"I": "mV", "II": "mV"} if units else None,
        metadata={"device": "synthetic", "filters": {"status": "known"}})


def _build(record, peaks, **kwargs):
    return build_ecg_measurement_bundle(
        record,
        primary_channel="I",
        r_indices=peaks,
        pre_s=0.20,
        post_s=0.40,
        source_sha256="a" * 64,
        patient_id="patient-01",
        encounter_id="encounter-01",
        entity_id="case-01",
        **kwargs,
    )


def test_v2_preserves_raw_units_beat_distribution_and_is_deterministic():
    record = _recording()
    peaks = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    first = _build(record, peaks)
    second = _build(record, peaks)

    assert first["schema_version"] == ECG_MEASUREMENT_SCHEMA_VERSION
    assert first["raw_measurements"]["samples_by_lead_original_units"]["I"] == record.signals["I"].tolist()
    assert first["acquisition"]["lead_names_and_order"] == ["I", "II"]
    assert first["acquisition"]["physical_voltage_unit_by_lead"]["I"] == "mV"
    assert first["acquisition"]["sampling_rate_agrees_with_time_axis_within_1_percent"] is True
    assert first["beat_analysis"]["r_peak_sample_indices"] == peaks
    assert len(first["beat_analysis"]["beats"]) == len(peaks)
    assert all(len(beat["waveform_by_lead_original_units"]["I"]) == len(first["relative_time_s"])
               for beat in first["beat_analysis"]["beats"])
    assert first["morphology"]["robust_template"]["eligible_for_calibration_fit"] is True
    assert first["quality"]["not_a_probability_of_measurement_correctness"] is True
    assert first["acceptance_gate"]["status"] == "not_demonstrated"
    validate_ecg_measurement_bundle(first)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)


def test_explicit_lead_mapping_keeps_source_labels_and_emits_model_order():
    source = _recording()
    record = type(source)(
        time=source.time,
        signals={"MLII": source.signals["I"], "V1": source.signals["II"]},
        sampling_rate_hz=source.sampling_rate_hz,
        source_format=source.source_format,
        channel_units={"MLII": "mV", "V1": "mV"},
        metadata=source.metadata,
    )
    peaks = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    bundle = build_ecg_measurement_bundle(
        record,
        primary_channel="MLII",
        r_indices=peaks,
        pre_s=0.20,
        post_s=0.40,
        source_sha256="b" * 64,
        patient_id="patient-02",
        encounter_id="encounter-02",
        entity_id="case-02",
        lead_mapping={"MLII": "II"},
        lead_order=["V1", "II", "V2"],
        expected_leads=["II", "V1", "V2"],
    )
    assert bundle["acquisition"]["lead_names_and_order"] == ["V1", "II"]
    assert bundle["acquisition"]["lead_mapping"]["mapping_status"] == "caller_declared"
    assert bundle["acquisition"]["lead_mapping"]["source_to_output_lead"] == {"MLII": "II", "V1": "V1"}
    assert bundle["acquisition"]["missing_expected_leads"] == ["V2"]
    assert bundle["raw_measurements"]["source_lead_names_and_order"] == ["MLII", "V1"]
    assert "MLII" in bundle["raw_measurements"]["samples_by_lead_original_units"]
    beat = bundle["beat_analysis"]["beats"][0]
    assert set(beat["waveform_by_lead_original_units"]) == {"II", "V1"}
    assert bundle["acquisition"]["physical_voltage_unit_by_lead"]["II"] == "mV"


def test_invalid_or_non_unique_lead_mapping_fails_closed():
    record = _recording()
    peaks = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    with pytest.raises(ValueError, match="unknown source leads"):
        _build(record, peaks, lead_mapping={"III": "III"})
    with pytest.raises(ValueError, match="duplicate output lead"):
        _build(record, peaks, lead_mapping={"I": "II"})


def test_annotations_are_preserved_and_excluded_from_fit_template():
    record = _recording()
    peaks = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    bundle = _build(
        record,
        peaks,
        beat_annotations={
            800: {"ectopic": True, "paced": False, "source": "manual", "adjudicated": True},
            1000: {"ectopic": False, "paced": True, "source": "device", "adjudicated": True},
        },
    )
    by_peak = {beat["sample_index"]: beat for beat in bundle["beat_analysis"]["beats"]}
    assert by_peak[800]["beat_annotation"]["ectopic"] is True
    assert by_peak[1000]["beat_annotation"]["paced"] is True
    fit_peaks = set(bundle["morphology"]["robust_template"]["template_beat_sample_indices"])
    assert 800 not in fit_peaks
    assert 1000 not in fit_peaks
    assert len(bundle["beat_analysis"]["beats"]) == len(peaks)


def test_noisy_unknown_units_missing_lead_and_irregular_rhythm_are_qualified():
    peaks = [200, 400, 600, 900, 1100, 1300, 1500, 1700]
    record = _recording(noise=0.8, peaks=peaks, units=False)
    bundle = _build(
        record, peaks, expected_leads=["I", "II", "III"], lead_order=["II", "I", "III"],
    )
    assert bundle["acquisition"]["missing_expected_leads"] == ["III"]
    assert bundle["acquisition"]["lead_names_and_order"] == ["II", "I"]
    assert bundle["quality"]["status"] in {"qualified", "rejected"}
    assert "one_or_more_voltage_units_unknown_or_unrecognized" in bundle["quality"]["reasons"]
    assert "expected_leads_missing" in bundle["quality"]["reasons"]
    assert bundle["beat_analysis"]["rhythm_characterization"]["pattern_screen"] == "variable_rr_pattern"
    assert any(beat["rr_outlier_screen"]["flagged"] for beat in bundle["beat_analysis"]["beats"])


def test_short_recording_is_explicitly_rejected_not_silently_averaged():
    record = _recording(peaks=[200, 400])
    bundle = _build(record, [200, 400])
    assert bundle["quality"]["status"] == "rejected"
    assert "fewer_than_minimum_complete_beats" in bundle["quality"]["reasons"]
    assert bundle["calibration_recommendation"]["template_fit_eligible"] is False


def test_nonfinite_raw_signal_fails_closed():
    record = _recording()
    record.signals["I"][100] = np.nan
    with pytest.raises(ValueError, match="raw ECG values must be finite"):
        _build(record, [200, 400, 600, 800, 1000, 1200, 1400, 1600])


def test_uncertainty_evaluation_requires_independent_manual_reference():
    with pytest.raises(ValueError, match="independent adjudicated"):
        evaluate_uncertainty_calibration(
            [1.0, 2.0], [1.0, 2.0],
            adjudication_source="manual-review",
            measurement_unit="ms",
            subject_ids=["s1", "s2"],
            encounter_ids=["e1", "e2"],
            independent_reference=False,
        )
    with pytest.raises(ValueError, match="independent validation/test"):
        evaluate_uncertainty_calibration(
            [1.0, 2.0], [1.0, 2.0],
            adjudication_source="manual-review",
            measurement_unit="ms",
            subject_ids=["s1", "s2"],
            encounter_ids=["e1", "e2"],
            reference_split="training",
            independent_reference=True,
        )


def test_uncertainty_evaluation_reports_empirical_error_curve_without_probability_claim():
    uncertainty = np.linspace(0.1, 2.0, 30)
    errors = uncertainty * 1.4 + 0.05
    subjects = [f"subject-{i % 10}" for i in range(30)]
    encounters = [f"encounter-{i}" for i in range(30)]
    report = evaluate_uncertainty_calibration(
        uncertainty,
        errors,
        adjudication_source="independent-manual-adjudication-v1",
        measurement_unit="ms",
        subject_ids=subjects,
        encounter_ids=encounters,
        independent_reference=True,
        reference_split="independent_test",
    )
    assert report["status"] == "empirical_reference_available"
    assert report["n_measurements"] == 30
    assert report["n_distinct_subjects"] == 10
    assert report["n_distinct_encounters"] == 30
    assert report["spearman_scale_error_correlation"] > 0.99
    assert "not a probability" in report["interpretation"]


def test_uncertainty_map_is_fit_only_on_declared_adjudicated_calibration_split():
    uncertainty = np.linspace(0.1, 2.0, 30)
    errors = uncertainty * 1.4 + 0.05
    subjects = [f"subject-{i % 10}" for i in range(30)]
    encounters = [f"encounter-{i}" for i in range(30)]
    with pytest.raises(ValueError, match="independent adjudicated"):
        fit_uncertainty_calibration_map(
            uncertainty, errors,
            adjudication_source="manual-review-v1",
            measurement_unit="ms",
            subject_ids=subjects,
            encounter_ids=encounters,
        )
    calibration_map = fit_uncertainty_calibration_map(
        uncertainty, errors,
        adjudication_source="manual-review-v1",
        measurement_unit="ms",
        subject_ids=subjects,
        encounter_ids=encounters,
        independent_reference=True,
        reference_split="adjudicated_calibration",
    )
    calibrated = apply_uncertainty_calibration_map([0.2, 1.0, 1.8], calibration_map)
    assert len(calibrated) == 3
    assert calibrated[0] <= calibrated[1] <= calibrated[2]
    assert calibration_map["measurement_unit"] == "ms"
    with pytest.raises(ValueError, match="declared adjudicated calibration split"):
        fit_uncertainty_calibration_map(
            uncertainty, errors,
            adjudication_source="manual-review-v1",
            measurement_unit="ms",
            subject_ids=subjects,
            encounter_ids=encounters,
            independent_reference=True,
            reference_split="independent_test",
        )


def test_prepare_calibration_writes_v2_artifact_with_source_hash_and_preserves_hint(tmp_path: Path):
    record = _recording()
    source = tmp_path / "ecg.csv"
    rows = ["time,I,II"]
    rows.extend(
        f"{t:.12g},{record.signals['I'][i]:.12g},{record.signals['II'][i]:.12g}"
        for i, t in enumerate(record.time)
    )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    handoff = prepare_ecg_calibration(
        source,
        entity_id="case-01",
        output_dir=tmp_path / "calibration",
        params={
            "patient_id": "patient-01",
            "encounter_id": "encounter-01",
            "primary_channel": "I",
            "r_indices": [200, 400, 600, 800, 1000, 1200, 1400, 1600],
            "pre_s": 0.20,
            "post_s": 0.40,
            "units": "mV",
        },
    )
    observation = handoff["observations"][0]
    artifact = Path(observation["artifact"]["uri"].replace("file://", ""))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ECG_MEASUREMENT_SCHEMA_VERSION
    assert payload["raw_measurements"]["source_reference"]["sha256"] == expected_hash
    assert handoff["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert handoff["summary"]["schema_version"] == ECG_MEASUREMENT_SCHEMA_VERSION
    assert handoff["likelihood_hints"][0]["model_output"] == "ecg"
    assert handoff["summary"]["acceptance_gate_status"] == "not_demonstrated"
    # QRS repeatability across beats is not silently reused as measurement error.
    assert all(h["model_output"] != "qrs_duration_ms" for h in handoff["likelihood_hints"])
    if payload["qrs"]["median_duration_ms"] is not None:
        assert any(term["model_output"] == "qrs_duration_ms" for term in handoff["summary"]["suppressed_likelihood_terms"])
