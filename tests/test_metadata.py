from pathlib import Path

import numpy as np
import pandas as pd

from electrotrace.metadata import recording_metadata


def test_csv_metadata_does_not_materialize_signal_columns(tmp_path: Path):
    path = tmp_path / "large.csv"
    frame = pd.DataFrame({
        "time": np.arange(100_000, dtype=float) / 500.0,
        "Lead_I": np.arange(100_000, dtype=float),
        "Lead_II": np.arange(100_000, dtype=float) * 2,
    })
    frame.to_csv(path, index=False)
    meta = recording_metadata(path)
    assert meta["n_samples"] == 100_000
    assert meta["channels"] == ["Lead_I", "Lead_II"]
    assert meta["sampling_rate_hz"] == 500.0
    assert meta["duration_s"] == frame["time"].iloc[-1] - frame["time"].iloc[0]


def test_csv_metadata_rejects_non_monotonic_time(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": [0.0, 0.01, 0.005], "II": [1, 2, 3]}).to_csv(path, index=False)
    try:
        recording_metadata(path)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("non-monotonic timestamps should be rejected")
