import subprocess
import sys

import electrotrace


def test_documented_public_api_imports():
    expected = [
        "detect_r_peaks",
        "detect_r_peaks_two_stage",
        "apply_pipeline",
        "load_recording",
        "validate_record",
        "CandidateSuppressor",
        "delineate_qrs",
    ]
    assert all(hasattr(electrotrace, name) for name in expected)


def test_cli_list_is_runnable():
    completed = subprocess.run(
        [sys.executable, "-m", "electrotrace", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "pan-tompkins" in completed.stdout
    assert "hamilton" in completed.stdout
