import json
import sys
import types

import numpy as np
import pytest

from electrotrace.candidate_suppressor import CandidateSuppressor, _candidate_features
from scripts import evaluate_frozen_model_mitdb as eval_script
from scripts import inspect_fp_traces as inspect_script
from scripts import recalibrate_threshold_grouped_cv as cv_script

FS = 360.0


def _peaks(n=20, step=300, start=200):
    return np.arange(start, start + n * step, step)


def _record(peaks, fs=FS, n=None, t_offset_samples=150, t_amp=0.9, noise=None):
    n = n or int(peaks[-1] + 2000)
    x = np.zeros(n)
    for p in peaks:
        lo, hi = max(0, p - 4), min(n, p + 5)
        x[lo:hi] += np.hanning(hi - lo) * 2.5
        q = p + t_offset_samples
        if 0 <= q < n:
            lo2, hi2 = max(0, q - 6), min(n, q + 7)
            x[lo2:hi2] += np.hanning(hi2 - lo2) * t_amp
    if noise is not None:
        x = x + noise
    return x


def _tiny_model(signal, peaks):
    X, names = _candidate_features(signal, FS, peaks, np.ones(len(peaks)))
    X_train = np.vstack([X, X + 0.01])
    y_train = np.array([1] * len(peaks) + [0] * len(peaks))
    model = CandidateSuppressor().fit(X_train, y_train, target_recall=0.9, n_estimators=20)
    model.feature_names = names
    return model


def _fake_wfdb(monkeypatch, records: dict):
    fake = types.ModuleType("wfdb")

    class Obj:
        pass

    def rdrecord(base, channels, physical):
        name = base.split("/")[-1]
        signal, fs, _ = records[name]
        r = Obj()
        r.fs = fs
        r.p_signal = None
        r.d_signal = signal.reshape(-1, 1)
        return r

    def rdheader(base):
        name = base.split("/")[-1]
        signal, fs, _ = records[name]
        h = Obj()
        h.fs = fs
        h.sig_len = len(signal)
        return h

    def rdann(base, extension):
        name = base.split("/")[-1]
        _, _, ann_sample = records[name]
        a = Obj()
        a.sample = np.asarray(ann_sample)
        a.symbol = ["N"] * len(ann_sample)
        return a

    fake.rdrecord, fake.rdheader, fake.rdann = rdrecord, rdheader, rdann
    monkeypatch.setitem(sys.modules, "wfdb", fake)


def _write_stub_files(tmp_path, names):
    for n in names:
        for suffix in (".hea", ".dat", ".atr"):
            (tmp_path / f"{n}{suffix}").write_bytes(b"x")


def test_evaluate_frozen_model_scores_locked_records_without_retraining(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["105"])
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")

    _fake_wfdb(monkeypatch, {"105": (x, FS, peaks)})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "105",
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert report["protocol"]["retraining"] is False
    assert report["protocol"]["polarity"] == "adaptive"
    assert report["summary"]["records"] == 1
    assert report["record_results"][0]["record"] == "105"
    assert "numpy" in report["package_versions"]
    assert report["model"]["sha256"]


def test_evaluate_frozen_model_reports_skipped_records_with_nonzero_exit(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["207"])
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")

    _fake_wfdb(monkeypatch, {"207": (x, FS, np.concatenate([[-5], peaks]))})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "207",
            "--output",
            str(out),
        ],
    )
    with pytest.raises(SystemExit, match="No usable records"):
        eval_script.main()


def test_evaluate_frozen_model_threshold_override_changes_score(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["105"])
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    _fake_wfdb(monkeypatch, {"105": (x, FS, peaks)})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))

    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frozen_model_mitdb.py",
            "--mitdb-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "105",
            "--threshold",
            "0.999",
            "--output",
            str(out),
        ],
    )
    assert eval_script.main() == 0
    report = json.loads(out.read_text())
    assert report["protocol"]["threshold_overridden"] is True
    assert report["summary"]["detected_count"] == 0


def test_collect_record_returns_sorted_candidates_and_labels(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["A01"])
    _fake_wfdb(monkeypatch, {"A01": (x, FS, peaks)})

    data, info = cv_script.collect_record(
        model, tmp_path / "A01", "incart", scale_method="windowed_std", policy="error", tolerance_ms=75.0
    )
    assert data is not None
    assert np.all(np.diff(data.candidates) >= 0)
    assert data.labels.sum() >= 1
    assert info["audit"].action == "kept_unchanged"


def test_collect_record_returns_none_and_reason_when_excluded(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["I04"])
    _fake_wfdb(monkeypatch, {"I04": (x, FS, np.concatenate([[-5], peaks]))})

    data, info = cv_script.collect_record(
        model, tmp_path / "I04", "incart", scale_method="windowed_std", policy="error", tolerance_ms=75.0
    )
    assert data is None
    assert info["reason"] == "negative annotation sample index"


def test_group_k_fold_never_splits_a_record_across_folds():
    groups = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 4 + ["d"] * 6)
    folds = cv_script.group_k_fold(groups, n_splits=3, seed=0)
    assert len(folds) == 3
    for _, test_idx in folds:
        test_groups = set(groups[test_idx].tolist())
        for g in test_groups:
            all_idx = np.flatnonzero(groups == g)
            assert set(all_idx).issubset(set(test_idx.tolist()))


def test_select_balanced_threshold_is_not_dominated_by_the_larger_database():
    """The larger database must not dominate a cross-database threshold fit."""
    big_true = np.full(1000, 0.9)
    big_false = np.full(1000, 0.55)
    small_true = np.full(100, 0.5)
    small_false = np.full(100, 0.1)

    labels = np.concatenate([np.ones(1000), np.zeros(1000), np.ones(100), np.zeros(100)]).astype(np.int8)
    probs = np.concatenate([big_true, big_false, small_true, small_false])
    databases = np.array(["big"] * 2000 + ["small"] * 200)

    balanced, per_db = cv_script.select_balanced_threshold(labels, probs, databases, min_recall=0.9)
    pooled = cv_script.select_threshold_for_f1(labels, probs, min_recall=0.9)

    assert per_db["big"] == pytest.approx(0.9)
    assert per_db["small"] == pytest.approx(0.5)
    assert pooled == pytest.approx(per_db["big"])
    assert balanced != pytest.approx(pooled)
    assert abs(balanced - per_db["small"]) < abs(pooled - per_db["small"])


def test_select_balanced_threshold_raises_when_no_database_has_positives():
    labels = np.zeros(10, dtype=np.int8)
    probs = np.linspace(0, 1, 10)
    databases = np.array(["a"] * 5 + ["b"] * 5)
    with pytest.raises(ValueError, match="no database"):
        cv_script.select_balanced_threshold(labels, probs, databases, min_recall=0.9)


def test_run_cv_pools_each_record_exactly_once():
    rng = np.random.default_rng(3)
    records = []
    for i in range(8):
        n_cand = 30
        cand = np.sort(rng.choice(np.arange(2000), size=n_cand, replace=False))
        labels = (rng.random(n_cand) < 0.5).astype(np.int8)
        probs = np.where(
            labels == 1, rng.uniform(0.6, 1.0, n_cand), rng.uniform(0.0, 0.4, n_cand)
        )
        ref = np.sort(cand[labels == 1])
        records.append(
            cv_script.RecordCandidates(
                record=f"R{i}",
                database="incart" if i % 2 else "mitdb_calibration",
                fs_hz=FS,
                candidates=cand,
                probabilities=probs,
                labels=labels,
                reference=ref,
            )
        )
    cv = cv_script.run_cv(records, n_splits=4, seed=1, tolerance_ms=75.0)
    assert cv["n_folds"] == 4
    tested_records = {r["record"] for r in cv["pooled_record_results"]}
    assert tested_records == {r.record for r in records}
    assert len(cv["pooled_record_results"]) == len(records)
    assert set(cv["pooled_summary_by_database"]) == {"incart", "mitdb_calibration"}
    for fold in cv["folds"]:
        assert set(fold["threshold_by_database"]) <= {"incart", "mitdb_calibration"}


def test_group_k_fold_falls_back_when_shuffle_kwarg_unsupported(monkeypatch):
    from sklearn.model_selection import GroupKFold as RealGroupKFold

    class OldGroupKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, X, groups):
            return RealGroupKFold(n_splits=self.n_splits).split(X, groups=groups)

    monkeypatch.setattr("sklearn.model_selection.GroupKFold", OldGroupKFold)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    folds = cv_script.group_k_fold(groups, n_splits=3, seed=0)
    assert len(folds) == 3


def test_main_writes_report_with_baseline_cv_and_candidate_threshold(monkeypatch, tmp_path):
    mit_dir = tmp_path / "mit"
    incart_dir = tmp_path / "incart"
    mit_dir.mkdir()
    incart_dir.mkdir()

    calib_records = {}
    for i, name in enumerate(cv_script.MITDB_CALIBRATION_RECORDS):
        peaks = _peaks(start=100 + i * 7)
        calib_records[name] = (_record(peaks), FS, peaks)
    incart_records = {}
    for i, name in enumerate(["I01", "I02", "I03", "I05"]):
        peaks = _peaks(start=150 + i * 11)
        incart_records[name] = (_record(peaks), FS, peaks)

    all_records = {**calib_records, **incart_records}
    _write_stub_files(mit_dir, calib_records)
    _write_stub_files(incart_dir, incart_records)
    _fake_wfdb(monkeypatch, all_records)

    x0, p0 = calib_records[cv_script.MITDB_CALIBRATION_RECORDS[0]][0], _peaks(start=100)
    model = _tiny_model(x0, p0)
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))

    out = tmp_path / "cv.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recalibrate_threshold_grouped_cv.py",
            "--mitdb-dir",
            str(mit_dir),
            "--incart-dir",
            str(incart_dir),
            "--model",
            str(model_file),
            "--folds",
            "3",
            "--output",
            str(out),
        ],
    )
    assert cv_script.main() == 0
    report = json.loads(out.read_text())
    assert report["evidence_status"] == "development_only"
    assert report["protocol"]["mitdb_held_out_records_excluded"] is True
    assert report["grouped_cv"]["n_folds"] <= 3
    assert "deployment_candidate_threshold" in report
    assert report["deployment_candidate_threshold"]["note"]
    n_records = len(calib_records) + len(incart_records)
    assert report["current_fixed_threshold"]["summary_all"]["n_records"] == n_records


def test_select_samples_spreads_across_records_and_respects_cap():
    pooled = (
        [{"record": "A", "id": i} for i in range(10)]
        + [{"record": "B", "id": i} for i in range(2)]
        + [{"record": "C", "id": i} for i in range(1)]
    )
    selected = inspect_script.select_samples(pooled, n_samples=6, seed=0, max_per_record=3)
    counts = {}
    for s in selected:
        counts[s["record"]] = counts.get(s["record"], 0) + 1
    assert len(selected) == 6
    assert all(c <= 3 for c in counts.values())
    assert counts.get("B", 0) == 2
    assert counts.get("C", 0) == 1


def test_select_samples_handles_empty_pool():
    assert inspect_script.select_samples([], n_samples=5, seed=0) == []


def test_build_pool_filters_by_amp_ratio_and_polarity(monkeypatch, tmp_path):
    peaks = _peaks(n=10, step=400, start=200)
    x = _record(peaks, t_offset_samples=180, t_amp=1.4)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["I01"])
    _fake_wfdb(monkeypatch, {"I01": (x, FS, peaks)})

    pooled, signals, skipped = inspect_script.build_pool(
        model,
        tmp_path,
        ["I01"],
        scale_method="windowed_std",
        threshold=0.0,
        tolerance_ms=75.0,
        policy="error",
        amp_ratio_min=0.0,
        polarity_filter="any",
    )
    assert skipped == []
    assert "I01" in signals
    assert all(row["amp_ratio"] >= 0.0 for row in pooled)

    strict = inspect_script.build_pool(
        model,
        tmp_path,
        ["I01"],
        scale_method="windowed_std",
        threshold=0.0,
        tolerance_ms=75.0,
        policy="error",
        amp_ratio_min=0.9,
        polarity_filter="same",
    )[0]
    assert all(row["amp_ratio"] >= 0.9 and row["same_polarity"] for row in strict)
    assert len(strict) <= len(pooled)


def test_build_pool_skips_excluded_records(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["I04"])
    _fake_wfdb(monkeypatch, {"I04": (x, FS, np.concatenate([[-3], peaks]))})
    pooled, signals, skipped = inspect_script.build_pool(
        model,
        tmp_path,
        ["I04"],
        scale_method="windowed_std",
        threshold=0.0,
        tolerance_ms=75.0,
        policy="error",
        amp_ratio_min=0.0,
        polarity_filter="any",
    )
    assert pooled == []
    assert signals == {}
    assert skipped == [{"record": "I04", "reason": "negative annotation sample index"}]


def test_plot_samples_writes_png(tmp_path):
    fs = FS
    signal = _record(_peaks(n=8, step=500, start=200))
    signals = {"I01": (signal, fs, _peaks(n=8, step=500, start=200))}
    samples = [
        {
            "record": "I01",
            "sample": 700,
            "category": "post_qrs_t_window",
            "amp_ratio": 0.7,
            "same_polarity": True,
            "probability": 0.4,
        }
    ]
    out = tmp_path / "fig.png"
    assert inspect_script.plot_samples(out, samples, signals, window_s=0.5) is True
    assert out.exists() and out.stat().st_size > 0


def test_plot_samples_returns_false_for_empty_samples(tmp_path):
    assert inspect_script.plot_samples(tmp_path / "x.png", [], {}, window_s=0.5) is False


def test_main_writes_manifest_and_figure(monkeypatch, tmp_path):
    peaks = _peaks(n=10, step=400, start=200)
    x = _record(peaks, t_offset_samples=180, t_amp=1.3)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["I03"])
    _fake_wfdb(monkeypatch, {"I03": (x, FS, peaks)})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))

    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_fp_traces.py",
            "--incart-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "I03",
            "--threshold",
            "0.0",
            "--amp-ratio-min",
            "0.0",
            "--n-samples",
            "4",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert inspect_script.main() == 0
    manifest = json.loads((out_dir / "incart_fp_trace_samples.json").read_text())
    assert manifest["protocol"]["amp_ratio_min"] == 0.0
    assert len(manifest["samples"]) <= 4
    assert manifest["figure"] == "incart_fp_trace_samples.png"
    assert (out_dir / "incart_fp_trace_samples.png").exists()


def test_main_raises_when_nothing_matches(monkeypatch, tmp_path):
    peaks = _peaks()
    x = _record(peaks)
    model = _tiny_model(x, peaks)
    _write_stub_files(tmp_path, ["I03"])
    _fake_wfdb(monkeypatch, {"I03": (x, FS, peaks)})
    monkeypatch.setattr(CandidateSuppressor, "load", staticmethod(lambda path, **kw: model))
    model_file = tmp_path / "model.skops"
    model_file.write_bytes(b"model")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_fp_traces.py",
            "--incart-dir",
            str(tmp_path),
            "--model",
            str(model_file),
            "--records",
            "I03",
            "--threshold",
            "0.999",
            "--amp-ratio-min",
            "0.0",
            "--output-dir",
            str(tmp_path / "out2"),
        ],
    )
    with pytest.raises(SystemExit, match="No matching false positives"):
        inspect_script.main()
