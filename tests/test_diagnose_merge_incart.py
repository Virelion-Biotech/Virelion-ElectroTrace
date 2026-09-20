import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")

from scripts.diagnose_merge_incart import aggregate, evaluate_record, variant_specs
from test_dual_polarity_merge import FS, WidthStub, _t_discordant_record


def test_evaluate_record_end_to_end_and_pooled_is_distinguishable(tmp_path):
    x, refs, _ = _t_discordant_record()
    wfdb.wrsamp("I62", fs=int(FS), units=["mV"], sig_name=["i"], p_signal=x[:, None], fmt=["16"], write_dir=str(tmp_path))
    wfdb.wrann("I62", "atr", sample=refs, symbol=["N"] * len(refs), write_dir=str(tmp_path))
    specs = variant_specs()
    rec = evaluate_record(tmp_path / "I62", WidthStub(), "windowed_std", specs, want_detail=True)
    assert set(rec["variants"]) == {"adaptive", *specs}

    gaps_keys = [k for k in rec["variants"] if "gaps" in k and k.startswith("per_stream_")]
    assert gaps_keys, sorted(rec["variants"])
    # At least one gaps variant should not be worse on sensitivity than adaptive
    # on this synthetic T-discordant record (majority-only misses the notches).
    assert any(
        rec["variants"][k]["sens"] >= rec["variants"]["adaptive"]["sens"] - 1e-9
        for k in gaps_keys
    )

    d = rec["detail"]
    assert d["adaptive_missed_refs"]["of_which_no_candidate_in_majority_stream"] > 0
    assert d["reference_coverage"]["either_has_candidate"] >= d["reference_coverage"]["majority_has_candidate"]

    key = gaps_keys[0]
    agg = aggregate([rec], ["adaptive", key], "I62")
    assert key in agg
    assert "I62_delta_f1" in agg[key]
