from pathlib import Path

from electrotrace.hearttwin_adapter import _analyze_file


def test_hearttwin_adapter_analyzes_csv_without_flask(tmp_path: Path) -> None:
    path = tmp_path / "ecg.csv"
    path.write_text(
        "time,lead_i\n"
        "0.00,0.0\n"
        "0.01,0.2\n"
        "0.02,0.0\n"
        "0.03,-0.2\n"
        "0.04,0.0\n"
        "0.05,0.2\n"
        "0.06,0.0\n"
        "0.07,-0.2\n",
        encoding="utf-8",
    )
    result = _analyze_file(path, {"time_col": "time"})
    assert result["valid"] is True
    assert result["source_format"] == "CSV"
    assert result["signal_cols"] == ["lead_i"]
