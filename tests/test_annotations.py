import json

import pytest

from electrotrace.annotations import Annotation, AnnotationStore, interval_iou, point_agreement


def test_interval_validation_and_roundtrip():
    store = AnnotationStore(duration_s=2.0)
    ann = store.add(Annotation(label="QRS", type="interval", channel="Lead_II", start=0.5, end=0.6))
    payload = json.loads(store.to_json("x.csv"))
    restored = AnnotationStore.from_dict(payload, duration_s=2.0)
    assert restored.items[0].id == ann.id
    assert restored.items[0].start == 0.5


def test_invalid_interval_is_rejected():
    store = AnnotationStore(duration_s=2.0)
    with pytest.raises(ValueError):
        store.add(Annotation(label="QRS", type="interval", channel="Lead_II", start=1.0, end=1.0))


def test_update_duplicate_delete():
    store = AnnotationStore(duration_s=2.0)
    ann = store.add(Annotation(label="R Peak", type="point", channel="Lead_II", time=1.0))
    updated = store.update(ann.id, confidence=0.75)
    assert updated.confidence == 0.75
    copied = store.duplicate(ann.id)
    assert copied.id != ann.id
    assert store.delete(ann.id)
    assert len(store.items) == 1


def test_interval_iou():
    a = Annotation(label="QRS", type="interval", channel="II", start=1.0, end=2.0)
    b = Annotation(label="QRS", type="interval", channel="II", start=1.5, end=2.5)
    assert interval_iou(a, b) == pytest.approx(1 / 3)


def test_point_agreement():
    a = [Annotation(label="R Peak", type="point", channel="II", time=1.000)]
    b = [Annotation(label="R Peak", type="point", channel="II", time=1.020)]
    result = point_agreement(a, b, tolerance_s=0.04)
    assert result["matches"] == 1
    assert result["agreement_rate"] == 1
    assert result["mean_absolute_error_s"] == pytest.approx(0.02)


def test_point_agreement_does_not_cross_channels():
    a = [Annotation(label="R Peak", type="point", channel="I", time=1.000)]
    b = [Annotation(label="R Peak", type="point", channel="II", time=1.010)]
    result = point_agreement(a, b, tolerance_s=0.04)
    assert result["matches"] == 0
    assert result["agreement_rate"] == 0
