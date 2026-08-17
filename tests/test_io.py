import numpy as np
import pandas as pd
import pytest

from electrotrace.io import infer_sampling_rate, validate_dataframe


def test_validate_standard_ecg():
    df = pd.DataFrame({"time": np.arange(0, 1, 0.002), "Lead_II": np.sin(np.arange(0, 1, 0.002))})
    result = validate_dataframe(df)
    assert result.valid
    assert result.sampling_rate_hz == pytest.approx(500)
    assert result.signal_cols == ["Lead_II"]


def test_reject_non_monotonic_time():
    df = pd.DataFrame({"time": [0, 0.01, 0.005], "II": [1, 2, 3]})
    result = validate_dataframe(df)
    assert not result.valid
    assert any("strictly increasing" in e for e in result.errors)


def test_warn_irregular_sampling():
    df = pd.DataFrame({"time": [0.0, 0.01, 0.02, 0.035, 0.045], "II": [1, 2, 3, 4, 5]})
    result = validate_dataframe(df)
    assert result.valid
    assert any("irregular" in w for w in result.warnings)


def test_sampling_rate_empty():
    assert infer_sampling_rate(np.array([0.0])) is None
