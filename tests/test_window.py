import numpy as np
import pandas as pd

from electrotrace.window import read_recording_window


def test_csv_window_returns_only_requested_rows(tmp_path):
    path = tmp_path / "trace.csv"
    pd.DataFrame({"time": np.arange(1000) / 500, "II": np.arange(1000, dtype=float)}).to_csv(path, index=False)
    result = read_recording_window(path, 100, 125)
    assert result["start"] == 100
    assert result["stop"] == 125
    assert len(result["time"]) == 25
    assert result["signals"]["II"].tolist() == list(np.arange(100, 125, dtype=float))


def test_window_rejects_invalid_bounds(tmp_path):
    path = tmp_path / "trace.csv"
    pd.DataFrame({"time": [0.0, 0.01], "II": [1.0, 2.0]}).to_csv(path, index=False)
    try:
        read_recording_window(path, 10, 20)
    except ValueError as exc:
        assert "beyond" in str(exc)
    else:
        raise AssertionError("expected an out-of-range window to fail")
