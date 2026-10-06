from pathlib import Path

from electrotrace.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_readme_first_run_sample_executes(tmp_path):
    sample = ROOT / "sample_data" / "sample_ecg.csv"
    output = tmp_path / "peaks.csv"

    rc = main(
        [
            "detect",
            str(sample),
            "--detector",
            "pan-tompkins",
            "--channel",
            "0",
            "-o",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    assert output.with_name(output.name + ".manifest.json").exists()


def test_public_docs_identify_current_release():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
    assert "1.10.0 (release candidate)" not in readme
    assert "Current release: 1.10.0" in readme
    assert "ElectroTrace 1.10.0 is available from PyPI" in quickstart


def test_mkdocs_navigation_targets_exist():
    required = [
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "validation" / "README.md",
        ROOT / "docs" / "QTDB_VALIDATION.md",
        ROOT / "docs" / "QRS_DELINEATION.md",
        ROOT / "docs" / "CROSS_DATABASE_POLICY.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "LIMITATIONS.md",
        ROOT / "docs" / "DETECTOR_PLUGINS.md",
        ROOT / "docs" / "CONTRIBUTING.md",
    ]
    assert all(path.exists() for path in required)
