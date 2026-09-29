from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_certified_wfdb_mitdb_locked as bench


def _write_locked_inputs(root):
    for record in bench.LOCKED_TEST_RECORDS:
        for suffix in bench.REQUIRED_SUFFIXES:
            (root / f"{record}{suffix}").write_bytes(b"x")


def test_locked_split_matches_historical_12_record_partition():
    assert bench.LOCKED_TEST_RECORDS == [
        "105",
        "118",
        "122",
        "201",
        "207",
        "209",
        "214",
        "219",
        "230",
        "231",
        "232",
        "234",
    ]


def test_validate_locked_inputs_fails_closed(tmp_path):
    _write_locked_inputs(tmp_path)
    (tmp_path / "207.atr").unlink()

    with pytest.raises(SystemExit, match="input set is incomplete"):
        bench.validate_locked_inputs(tmp_path)


def test_evaluate_detector_scores_every_locked_record(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    work_dir = tmp_path / "work"
    data_dir.mkdir()
    _write_locked_inputs(data_dir)

    monkeypatch.setattr(bench.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(bench.wfdb, "rdheader", lambda *args, **kwargs: SimpleNamespace(fs=360.0))
    monkeypatch.setattr(
        bench,
        "reference_beats",
        lambda *args, **kwargs: np.asarray([100, 200, 300], dtype=np.int64),
    )
    monkeypatch.setattr(
        bench,
        "run_detector",
        lambda *args, **kwargs: np.asarray([100, 200, 300], dtype=np.int64),
    )

    results, executable = bench.evaluate_detector(
        "gqrs",
        data_dir=data_dir,
        work_dir=work_dir,
        tolerance_ms=75.0,
    )

    assert executable == "/usr/bin/gqrs"
    assert [result.record for result in results] == bench.LOCKED_TEST_RECORDS
    assert len(results) == 12
    assert all(result.metrics.f1 == pytest.approx(1.0) for result in results)


def test_run_detector_uses_record_directory_as_wfdb_search_root(monkeypatch, tmp_path):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    record_base = record_dir / "105"
    (record_base.with_suffix(".qrs")).write_bytes(b"x")
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bench.wfdb,
        "rdann",
        lambda *args, **kwargs: SimpleNamespace(
            sample=np.asarray([100, 200], dtype=int),
            symbol=["N", "N"],
        ),
    )

    detected = bench.run_detector("/usr/local/bin/gqrs", record_base)
    assert seen["args"] == ["/usr/local/bin/gqrs", "-r", "105", "-s", "0"]
    assert seen["cwd"] == record_dir
    assert detected.tolist() == [100, 200]
