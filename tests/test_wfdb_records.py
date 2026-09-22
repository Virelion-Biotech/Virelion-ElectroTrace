import sys
import types

import numpy as np
import pytest

from electrotrace.wfdb_records import (
    RecordExcluded,
    clean_reference_annotations,
    estimate_alignment,
    load_annotated_record,
    summarize_audits,
)

FS = 250.0


def _beats(n=40, step=200, start=100):
    return np.arange(start, start + n * step, step)


def _peaks_like(ref, jitter=2):
    return ref + jitter


def _clean(samples, symbols=None, **kw):
    samples = np.asarray(samples)
    symbols = symbols if symbols is not None else ["N"] * len(samples)
    kw.setdefault("n_samples", int(samples.max()) + 500 if samples.size else 1000)
    kw.setdefault("fs_hz", FS)
    return clean_reference_annotations(samples, symbols, **kw)


def test_clean_record_is_kept_unchanged_under_both_policies():
    ref = _beats()
    for policy in ("error", "drop_edges"):
        out, audit = _clean(ref, policy=policy)
        assert np.array_equal(out, ref)
        assert audit.action == "kept_unchanged"
        assert audit.alignment_checked is False


def test_error_policy_reproduces_historical_negative_index_failure():
    ref = np.concatenate([[-40], _beats()])
    with pytest.raises(RecordExcluded, match="negative annotation sample index") as exc:
        _clean(ref, policy="error")
    assert isinstance(exc.value, ValueError)  # old callers catch ValueError
    assert exc.value.audit.n_negative_beat == 1
    assert exc.value.audit.first_invalid == [[-40, "N"]]


def test_error_policy_does_not_reject_annotations_past_signal_end():
    # Historical scripts never checked the upper bound; keep results reproducible.
    ref = _beats()
    out, audit = _clean(ref, n_samples=int(ref[-1]) - 10, policy="error")
    assert np.array_equal(out, ref)
    assert audit.n_beyond_end_beat == 1
    assert audit.action == "kept_unchanged"


def test_non_beat_symbols_are_ignored_even_when_negative():
    ref = _beats()
    samples = np.concatenate([[-5], ref])
    symbols = ["+"] + ["N"] * len(ref)  # rhythm annotation, not a beat
    out, audit = _clean(samples, symbols, policy="error")
    assert np.array_equal(out, ref)
    assert audit.n_negative_beat == 0


def test_drop_edges_repairs_leading_negatives_when_alignment_verified():
    ref = _beats()
    samples = np.concatenate([[-90, -30], ref])
    out, audit = _clean(
        samples, policy="drop_edges", alignment_peaks=_peaks_like(ref)
    )
    assert np.array_equal(out, ref)
    assert audit.action == "kept_after_dropping_edges"
    assert audit.n_leading_invalid == 2
    assert audit.alignment_checked is True
    assert audit.alignment_coverage == pytest.approx(1.0)


def test_drop_edges_repairs_trailing_beyond_end():
    ref = _beats()
    n = int(ref[-1]) - 10
    out, audit = _clean(ref, n_samples=n, policy="drop_edges", alignment_peaks=ref)
    assert np.array_equal(out, ref[:-1])
    assert audit.n_trailing_invalid == 1


def test_drop_edges_refuses_unverified_repair():
    ref = _beats()
    samples = np.concatenate([[-30], ref])
    with pytest.raises(RecordExcluded, match="not verified"):
        _clean(samples, policy="drop_edges", alignment_peaks=None)


def test_drop_edges_rejects_repair_when_reference_is_shifted():
    # A misparsed skip would shift every annotation; independent peaks expose it.
    ref = _beats()
    samples = np.concatenate([[-30], ref + 97])  # ~388 ms shift, half an RR
    with pytest.raises(RecordExcluded, match="repair rejected") as exc:
        _clean(samples, policy="drop_edges", alignment_peaks=ref)
    assert exc.value.audit.alignment_checked is True
    assert exc.value.audit.alignment_coverage < 0.5


def test_drop_edges_rejects_interior_invalid_annotations():
    ref = _beats()
    samples = np.concatenate([ref[:10], [-3], ref[10:]])
    with pytest.raises(RecordExcluded, match="interior"):
        _clean(samples, policy="drop_edges", alignment_peaks=ref)


def test_non_monotonic_reference_is_excluded_not_repaired():
    ref = _beats()
    samples = ref.copy()
    samples[5], samples[6] = samples[6], samples[5]
    for policy in ("error", "drop_edges"):
        with pytest.raises(RecordExcluded, match="strictly increasing"):
            _clean(samples, policy=policy, alignment_peaks=ref)


def test_alignment_peaks_callable_only_invoked_when_repair_needed():
    calls = []

    def factory():
        calls.append(1)
        return _beats()

    _clean(_beats(), policy="drop_edges", alignment_peaks=factory)
    assert calls == []
    _clean(np.concatenate([[-3], _beats()]), policy="drop_edges", alignment_peaks=factory)
    assert calls == [1]


def test_estimate_alignment_reports_coverage_and_signed_offset():
    ref = _beats()
    coverage, offset = estimate_alignment(ref + 5, ref, FS)
    assert coverage == pytest.approx(1.0)
    assert offset == pytest.approx(5 * 1000 / FS)
    assert estimate_alignment([], ref, FS) == (0.0, None)


def test_summarize_audits_counts_each_outcome():
    ref = _beats()
    _, a = _clean(ref)
    _, b = _clean(
        np.concatenate([[-1], ref]), policy="drop_edges", alignment_peaks=ref
    )
    with pytest.raises(RecordExcluded) as exc:
        _clean(np.concatenate([[-1], ref]), record="I04")
    summary = summarize_audits([a, b, exc.value.audit])
    assert summary["kept_unchanged"] == 1
    assert summary["kept_after_dropping_edges"] == 1
    assert summary["excluded"] == 1
    assert summary["excluded_records"][0]["record"] == "I04"


def test_invalid_policy_and_shape_mismatch():
    with pytest.raises(ValueError, match="policy"):
        _clean(_beats(), policy="drop_everything")
    with pytest.raises(ValueError, match="same length"):
        clean_reference_annotations([1, 2], ["N"], n_samples=10, fs_hz=FS)


def _install_fake_wfdb(monkeypatch, *, signal, fs, ann_sample, ann_symbol):
    fake = types.ModuleType("wfdb")

    class Rec:
        pass

    def rdrecord(base, channels, physical):
        r = Rec()
        r.fs = fs
        r.p_signal = None
        r.d_signal = np.asarray(signal, dtype=float).reshape(-1, 1)
        return r

    def rdheader(base):
        h = Rec()
        h.fs = fs
        h.sig_len = len(signal)
        return h

    def rdann(base, extension):
        a = Rec()
        a.sample = np.asarray(ann_sample)
        a.symbol = list(ann_symbol)
        return a

    fake.rdrecord, fake.rdheader, fake.rdann = rdrecord, rdheader, rdann
    monkeypatch.setitem(sys.modules, "wfdb", fake)


def _ecg(fs=FS, beats=None, n=12000):
    x = np.zeros(n)
    for b in beats:
        lo, hi = max(0, b - 3), min(n, b + 4)
        x[lo:hi] += np.hanning(hi - lo) * 3.0
    return x + 0.01 * np.sin(np.arange(n) / 50.0)


def test_load_annotated_record_uses_pan_tompkins_to_verify_repair(monkeypatch):
    ref = np.arange(300, 11000, 250)
    signal = _ecg(beats=ref)
    _install_fake_wfdb(
        monkeypatch,
        signal=signal,
        fs=FS,
        ann_sample=np.concatenate([[-50], ref]),
        ann_symbol=["N"] * (len(ref) + 1),
    )
    with pytest.raises(RecordExcluded):
        load_annotated_record("/tmp/I99", policy="error")
    rec = load_annotated_record("/tmp/I99", policy="drop_edges")
    assert np.array_equal(rec.reference, ref)
    assert rec.audit.action == "kept_after_dropping_edges"
    assert rec.audit.alignment_coverage > 0.9
    assert rec.signal is not None and rec.fs_hz == FS


def test_load_annotated_record_header_only_needs_external_peaks(monkeypatch):
    ref = np.arange(300, 11000, 250)
    _install_fake_wfdb(
        monkeypatch,
        signal=np.zeros(12000),
        fs=FS,
        ann_sample=np.concatenate([[-50], ref]),
        ann_symbol=["N"] * (len(ref) + 1),
    )
    with pytest.raises(RecordExcluded, match="not verified"):
        load_annotated_record("/tmp/I99", policy="drop_edges", load_signal=False)
    rec = load_annotated_record(
        "/tmp/I99", policy="drop_edges", load_signal=False, alignment_peaks=ref + 3
    )
    assert rec.signal is None
    assert np.array_equal(rec.reference, ref)
