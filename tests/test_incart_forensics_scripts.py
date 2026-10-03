import csv
import json
import sys
import types

import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from scripts import diagnose_wfdb_annotations as diag
from scripts import incart_fp_offsets as fp_script

FS = 360.0
R_PEAKS = np.arange(900, 6500, 900)


def _record(with_t_bumps=True, n=7200):
    t = np.arange(n) / FS
    x = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    for p in R_PEAKS:
        lo, hi = max(0, p - 4), min(n, p + 5)
        x[lo:hi] += np.hanning(hi - lo) * 2.0
        if with_t_bumps:
            q = p + 150  # ~417 ms after the R peak
            lo, hi = q - 6, q + 7
            x[lo:hi] += np.hanning(hi - lo) * 0.9
    return x


def _tiny_model(signal):
    candidates = np.asarray(R_PEAKS)
    X, names = _candidate_features(signal, FS, candidates, np.ones(len(candidates)))
    X_train = np.vstack([X, X + 0.01])
    y_train = np.array([1] * len(candidates) + [0] * len(candidates))
    model = CandidateSuppressor().fit(X_train, y_train, target_recall=0.9, n_estimators=20)
    model.feature_names = names
    return model


def test_analyze_signal_record_reports_false_positive_structure():
    x = _record()
    model = _tiny_model(x)
    result = fp_script.analyze_signal_record(model, x, FS, R_PEAKS, threshold=0.0)
    assert result["stage1_candidates"] >= len(R_PEAKS)
    assert result["metrics"]["reference_count"] == len(R_PEAKS)
    cats = result["false_positive_categories"]
    assert sum(cats.values()) == result["metrics"]["false_positive"]
    assert len(result["candidate_probability_true"]) + len(result["candidate_probability_false"]) == (
        result["stage1_candidates"]
    )
    row = fp_script.record_row("X01", FS, result)
    json.dumps(row, default=fp_script._json_default)


def _fake_wfdb(monkeypatch, signal, ann_sample):
    fake = types.ModuleType("wfdb")

    class Obj:
        pass

    def rdrecord(base, channels, physical):
        r = Obj()
        r.fs = FS
        r.p_signal = None
        r.d_signal = signal.reshape(-1, 1)
        return r

    def rdheader(base):
        h = Obj()
        h.fs = FS
        h.sig_len = len(signal)
        return h

    def rdann(base, extension):
        a = Obj()
        a.sample = np.asarray(ann_sample)
        a.symbol = ["N"] * len(ann_sample)
        return a

    fake.rdrecord, fake.rdheader, fake.rdann = rdrecord, rdheader, rdann
    monkeypatch.setitem(sys.modules, "wfdb", fake)


def test_main_writes_report_csv_and_audits_end_to_end(monkeypatch, tmp_path):
    x = _record()
    model = _tiny_model(x)
    data = tmp_path / "incart"
    data.mkdir()
    for suffix in (".hea", ".dat", ".atr"):
        (data / f"I01{suffix}").write_bytes(b"x")
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    out = tmp_path / "out"

    _fake_wfdb(monkeypatch, x, R_PEAKS)
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))
    argv = [
        "incart_fp_offsets.py",
        "--incart-dir",
        str(data),
        "--model",
        str(model_file),
        "--output-dir",
        str(out),
        "--threshold",
        "0.0",
        "--no-plot",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert fp_script.main() == 0

    report = json.loads((out / "incart_fp_forensics.json").read_text())
    assert report["evidence_status"] == "development_only"
    assert report["protocol"]["retraining"] is False
    assert report["annotation_audit"]["summary"]["kept_unchanged"] == 1
    assert report["groups"]["all"]["n_records"] == 1
    assert report["model"]["sha256"]
    assert "numpy" in report["package_versions"]

    with (out / "incart_fp_detail.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["records"][0]["false_positive"]
    if rows:
        assert rows[0]["category"] in fp_script.CATEGORIES


def test_main_reports_excluded_record_instead_of_silently_dropping(monkeypatch, tmp_path):
    x = _record()
    model = _tiny_model(x)
    data = tmp_path / "incart"
    data.mkdir()
    for name in ("I01", "I04"):
        for suffix in (".hea", ".dat", ".atr"):
            (data / f"{name}{suffix}").write_bytes(b"x")
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")

    calls = {"n": 0}

    def rdann(base, extension):
        calls["n"] += 1
        a = types.SimpleNamespace()
        bad = str(base).endswith("I04")
        a.sample = np.concatenate([[-20], R_PEAKS]) if bad else R_PEAKS
        a.symbol = ["N"] * len(a.sample)
        return a

    _fake_wfdb(monkeypatch, x, R_PEAKS)
    sys.modules["wfdb"].rdann = rdann
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "incart_fp_offsets.py",
            "--incart-dir",
            str(data),
            "--model",
            str(model_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--threshold",
            "0.0",
            "--no-plot",
        ],
    )
    assert fp_script.main() == 0
    report = json.loads((tmp_path / "out" / "incart_fp_forensics.json").read_text())
    assert [s["record"] for s in report["skipped_records"]] == ["I04"]
    assert report["annotation_audit"]["summary"]["excluded"] == 1
    assert report["annotation_audit"]["summary"]["excluded_records"][0]["reason"] == "negative annotation sample index"

    # The same file under drop_edges is repaired because Pan-Tompkins agrees with the rest.
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-3], "--annotation-policy", "drop_edges", "--threshold", "0.0", "--no-plot"])
    assert fp_script.main() == 0
    repaired = json.loads((tmp_path / "out" / "incart_fp_forensics.json").read_text())
    assert repaired["skipped_records"] == []
    assert repaired["annotation_audit"]["summary"]["kept_after_dropping_edges"] == 1


def test_diagnose_record_reports_policies_and_alignment(monkeypatch, tmp_path):
    x = _record(with_t_bumps=False)
    _fake_wfdb(monkeypatch, x, np.concatenate([[-25], R_PEAKS]))
    report = diag.diagnose_record(tmp_path / "I04", tolerance_ms=75.0)
    assert report["policies"]["error"]["action"] == "excluded"
    assert report["policies"]["drop_edges"]["action"] == "kept_after_dropping_edges"
    assert report["raw"]["negative_positions"] == [0]
    assert report["valid_remainder_alignment"]["coverage_vs_pan_tompkins"] > 0.8
