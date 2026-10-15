from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_ludb as ludb


def ann(samples, symbols):
    return SimpleNamespace(sample=np.array(samples, dtype=int), symbol=list(symbols))


def test_incomplete_qrs_does_not_shift_subsequent_triplets():
    # Verified LUDB 1.0.1 data/7.v1 starts N, ) and contains N, (, ).
    a = ann(
        [968, 996, 1095, 1133, 1157, 1471, 1472, 1502, 1765, 1766, 1796],
        ["N", ")", "(", "t", ")", "(", "N", ")", "N", "(", ")"],
    )
    assert len(a.symbol) % 3  # The v1 worker rejects this real layout.
    on, off, audit = ludb.qrs_reference(a, 5000)
    assert on.tolist() == [1471]
    assert off.tolist() == [1502]
    assert audit["qrs_markers"] == 3
    assert audit["eligible_paired_qrs"] == 1
    assert audit["excluded_qrs"] == 2
    assert [x["peak_sample"] for x in audit["excluded"]] == [968, 1765]


def test_missing_offset_and_overlapping_markers_are_explicit_exclusions():
    on, off, audit = ludb.qrs_reference(ann([10, 15, 17, 20, 25], ["(", "N", "(", "t", ")"]), 30)
    assert not len(on) and not len(off)
    assert audit["excluded"][0]["reasons"] == ["missing_or_nonadjacent_offset"]


def test_ordinary_v1_triplets_are_unchanged():
    on, off, audit = ludb.qrs_reference(ann([1, 2, 3, 4, 5, 6], ["(", "p", ")", "(", "N", ")"]), 10)
    assert (on.tolist(), off.tolist()) == ([4], [6])
    assert audit["excluded_qrs"] == 0


@pytest.mark.parametrize(
    "a",
    [
        ann([-1, 2, 3], ["(", "N", ")"]),
        ann([3, 2, 4], ["(", "N", ")"]),
        ann([1, 2, 10], ["(", "N", ")"]),
        ann([1, 2, 3], ["(", "V", ")"]),
    ],
)
def test_bad_input_fails_closed(a):
    with pytest.raises(ValueError):
        ludb.qrs_reference(a, 10)


def test_summary_requires_all_patients():
    with pytest.raises(ValueError, match="all 200 distinct"):
        ludb.summarize([])
