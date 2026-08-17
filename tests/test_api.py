import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from server import app


def test_healthless_index_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"ElectroTrace" in response.data


def test_analyze_csv_endpoint():
    client = app.test_client()
    csv = b"time,Lead_II\n0.000,0.1\n0.002,0.2\n0.004,0.3\n"
    response = client.post("/api/analyze", data={"file": (io.BytesIO(csv), "test.csv")}, content_type="multipart/form-data")
    assert response.status_code == 200
    data = response.get_json()
    assert data["valid"] is True
    assert data["sampling_rate_hz"] == 500
    assert data["signal_cols"] == ["Lead_II"]


def test_filter_endpoint():
    client = app.test_client()
    fs = 500
    x = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * 2 * x).tolist()
    response = client.post("/api/filter", json={"sampling_rate_hz": fs, "signal": signal, "lowpass_hz": 40})
    assert response.status_code == 200
    assert len(response.get_json()["signal"]) == len(signal)


def test_segment_endpoint():
    client = app.test_client()
    fs = 500
    x = np.arange(2000) / fs
    signal = np.zeros_like(x)
    signal[[250, 750, 1250]] = 1
    response = client.post("/api/segment", json={"sampling_rate_hz": fs, "time": x.tolist(), "signal": signal.tolist(), "peaks": [250, 750, 1250]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["n_beats"] == 3


def test_statistics_endpoint_marks_observation_level_comparisons():
    client = app.test_client()
    response = client.post("/api/statistics/compare", json={"group_a": [1, 2, 3], "group_b": [4, 5, 6]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["unit_of_analysis"] == "observation"
    assert data["pseudoreplication_warning"] is True


def test_statistics_endpoint_aggregates_by_subject():
    client = app.test_client()
    response = client.post(
        "/api/statistics/compare",
        json={
            "group_a": [1, 3, 10, 12],
            "group_b": [4, 6, 14, 16],
            "unit_ids_a": ["s1", "s1", "s2", "s2"],
            "unit_ids_b": ["s3", "s3", "s4", "s4"],
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["unit_of_analysis"] == "experimental_unit_mean"
    assert data["n_units_a"] == 2
    assert data["n_observations_a"] == 4
    assert data["pseudoreplication_warning"] is False


def test_native_import_requires_supported_extension():
    client = app.test_client()
    response = client.post("/api/analyze", data={"file": (io.BytesIO(b"not a recording"), "recording.txt")}, content_type="multipart/form-data")
    assert response.status_code == 400
