import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from electrotrace.beats import segment_beats
from electrotrace.formats import _safe_extract
from electrotrace.statistics import benjamini_hochberg, compare_groups


def test_segment_beats_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        segment_beats(np.array([0.0, 0.1, 0.05]), np.array([1]))


def test_statistics_reject_invalid_p_values():
    with pytest.raises(ValueError, match="between 0 and 1"):
        benjamini_hochberg([0.01, 1.5])


def test_statistics_zero_variance_effect_size_is_explicit():
    result = compare_groups([1.0, 1.0], [1.0, 1.0])
    assert result["cohens_d"] is None


def test_safe_extract_rejects_zip_slip():
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as zf:
        zf.writestr("../../escape.txt", "owned")
    archive_bytes.seek(0)
    with zipfile.ZipFile(archive_bytes) as zf:
        with pytest.raises(ValueError, match="Unsafe archive path"):
            _safe_extract(zf, "/tmp/electrotrace-test-root")


def test_validation_handles_missing_signal_values_without_marking_all_data_validated():
    from electrotrace.io import validate_dataframe

    df = pd.DataFrame({"time": [0.0, 0.01, 0.02], "II": [1.0, np.nan, 2.0]})
    result = validate_dataframe(df)
    assert result.valid
    assert any("missing/non-numeric" in warning for warning in result.warnings)
