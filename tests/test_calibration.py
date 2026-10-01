from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from electrotrace.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    build_ecg_calibration_bundle,
    prepare_calibration,
    register_map_calibration,
)
from electrotrace.io import Recording


def _synthetic_recording(fs: float = 100.0, duration_s: float = 10.0):
    time = np.arange(int(fs * duration_s), dtype=float) / fs
    peaks = np.asarray([100, 300, 500, 700, 900], dtype=int)
    lead_i = np.zeros_like(time)
    lead_ii = np.zeros_like(time)
    for peak in peaks:
        x = np.arange(len(time)) - peak
        lead_i += np.exp(-0.5 * (x / 2.0) ** 2)
        lead_ii += 0.7 * np.exp(-0.5 * ((x - 1.0) / 2.5) ** 2)
    record = Recording(
        time=time,
        signals={"I": lead_i, "II": lead_ii},
        sampling_rate_hz=fs,
        source_format="synthetic",
        channel_units={"I": "mV", "II": "mV"},
    )
    return record, peaks


def test_build_ecg_calibration_bundle_is_simulation_comparable() -> None:
    record, peaks = _synthetic_recording()
    bundle = build_ecg_calibration_bundle(
        record,
        primary_channel="I",
        r_indices=peaks,
        pre_s=0.10,
        post_s=0.15,
        min_beats=3,
    )
    assert bundle["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert bundle["representation"] == "median_beat_template"
    assert bundle["lead_names"] == ["I", "II"]
    assert bundle["n_template_beats"] == 5
    assert set(bundle["beat_template"]) == {"I", "II"}
    assert len(bundle["relative_time_s"]) == len(bundle["beat_template"]["I"])
    assert bundle["uncertainty"]["noise_sigma_by_lead"].keys() == {"I", "II"}


def test_prepare_calibration_writes_ep_observation_artifact(tmp_path: Path) -> None:
    record, peaks = _synthetic_recording()
    source = tmp_path / "ecg.csv"
    rows = ["time,I,II"]
    rows.extend(
        f"{t:.6f},{record.signals['I'][i]:.12g},{record.signals['II'][i]:.12g}"
        for i, t in enumerate(record.time)
    )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    handoff = prepare_calibration(
        source,
        entity_id="S1",
        params={
            "observation_kind": "ecg",
            "r_indices": peaks.tolist(),
            "pre_s": 0.10,
            "post_s": 0.15,
            "units": "mV",
            "calibration_output_dir": tmp_path / "calibration",
        },
    )
    observation = handoff["observations"][0]
    assert observation["kind"] == "ecg"
    assert observation["artifact"]["kind"] == "electrotrace_ecg_calibration"
    assert observation["artifact"]["sha256"]
    artifact_path = Path(urlparse(observation["artifact"]["uri"]).path)
    assert artifact_path.is_file()
    assert handoff["likelihood_hints"][0]["model_output"] == "ecg"


def test_map_calibration_requires_explicit_coordinate_frame(tmp_path: Path) -> None:
    source = tmp_path / "activation.csv"
    source.write_text("x,y,z,lat_ms\n0,0,0,1.0\n", encoding="utf-8")
    try:
        register_map_calibration(
            source,
            entity_id="S1",
            kind="activation_map",
            coordinate_frame="",
            units="ms",
        )
    except ValueError as exc:
        assert "coordinate_frame" in str(exc)
    else:
        raise AssertionError("missing coordinate frame should fail closed")
