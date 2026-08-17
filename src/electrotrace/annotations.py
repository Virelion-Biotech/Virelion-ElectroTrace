"""Annotation model, validation, serialization, review state, and agreement metrics."""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

AnnotationType = Literal["interval", "point"]
ReviewStatus = Literal["unreviewed", "accepted", "flagged"]
DEFAULT_LABELS = ["P Wave", "QRS", "T Wave", "Pacemaker Spike", "R Peak", "Artifact", "Abnormal Beat", "Graft Activation", "Arrhythmia", "Custom"]


@dataclass
class Annotation:
    label: str
    type: AnnotationType
    channel: str
    start: Optional[float] = None
    end: Optional[float] = None
    time: Optional[float] = None
    confidence: float = 1.0
    notes: str = ""
    annotator: str = ""
    status: ReviewStatus = "unreviewed"
    reviewer: str = ""
    review_notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def validate(self, duration_s: float | None = None, start_time_s: float = 0.0, end_time_s: float | None = None) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not self.channel.strip():
            raise ValueError("channel must not be empty")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.status not in {"unreviewed", "accepted", "flagged"}:
            raise ValueError("invalid review status")
        start_bound = float(start_time_s)
        end_bound = float(end_time_s) if end_time_s is not None else (start_bound + float(duration_s) if duration_s is not None else None)
        if not math.isfinite(start_bound) or (end_bound is not None and not math.isfinite(end_bound)) or (end_bound is not None and end_bound < start_bound):
            raise ValueError("invalid recording time bounds")
        if self.type == "interval":
            if self.start is None or self.end is None:
                raise ValueError("interval annotations require start and end")
            if not math.isfinite(self.start) or not math.isfinite(self.end):
                raise ValueError("interval coordinates must be finite")
            if self.end <= self.start:
                raise ValueError("end must be greater than start")
            if self.time is not None:
                raise ValueError("interval annotations must not define time")
            if end_bound is not None and (self.start < start_bound or self.end > end_bound):
                raise ValueError("interval is outside the recording bounds")
        elif self.type == "point":
            if self.time is None:
                raise ValueError("point annotations require time")
            if not math.isfinite(self.time):
                raise ValueError("point time must be finite")
            if self.start is not None or self.end is not None:
                raise ValueError("point annotations must not define start/end")
            if end_bound is not None and not start_bound <= self.time <= end_bound:
                raise ValueError("point is outside the recording bounds")
        else:
            raise ValueError(f"unknown annotation type: {self.type}")

    @property
    def position(self) -> float:
        return self.start if self.type == "interval" else self.time  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return asdict(self)


class AnnotationStore:
    def __init__(self, duration_s: float | None = None, start_time_s: float = 0.0, end_time_s: float | None = None):
        self.duration_s = duration_s
        self.start_time_s = float(start_time_s)
        self.end_time_s = float(end_time_s) if end_time_s is not None else (self.start_time_s + float(duration_s) if duration_s is not None else None)
        self._items: list[Annotation] = []

    @property
    def items(self) -> list[Annotation]:
        return list(self._items)

    def add(self, ann: Annotation) -> Annotation:
        ann.validate(self.duration_s, self.start_time_s, self.end_time_s)
        if any(a.id == ann.id for a in self._items):
            raise ValueError(f"duplicate annotation id: {ann.id}")
        self._items.append(ann)
        self._sort()
        return ann

    def update(self, ann_id: str, **changes) -> Annotation:
        for idx, existing in enumerate(self._items):
            if existing.id == ann_id:
                data = {**asdict(existing), **changes}
                updated = Annotation(**data)
                updated.validate(self.duration_s, self.start_time_s, self.end_time_s)
                self._items[idx] = updated
                self._sort()
                return updated
        raise KeyError(ann_id)

    def delete(self, ann_id: str) -> bool:
        before = len(self._items)
        self._items = [a for a in self._items if a.id != ann_id]
        return len(self._items) != before

    def duplicate(self, ann_id: str) -> Annotation:
        for existing in self._items:
            if existing.id == ann_id:
                data = asdict(existing)
                data["id"] = uuid.uuid4().hex[:10]
                data["status"] = "unreviewed"
                data["reviewer"] = ""
                data["review_notes"] = ""
                return self.add(Annotation(**data))
        raise KeyError(ann_id)

    def clear(self) -> None:
        self._items.clear()

    def _sort(self) -> None:
        self._items.sort(key=lambda a: a.position)

    def to_dict(self, source_file: str = "", metadata: dict | None = None) -> dict:
        meta = dict(metadata or {})
        meta.setdefault("time_start_s", self.start_time_s)
        if self.end_time_s is not None:
            meta.setdefault("time_end_s", self.end_time_s)
        return {"schema": "electrotrace.annotation/v2", "file": source_file, "metadata": meta, "annotations": [a.to_dict() for a in self._items]}

    def to_json(self, source_file: str = "", metadata: dict | None = None) -> str:
        return json.dumps(self.to_dict(source_file, metadata), indent=2)

    @classmethod
    def from_dict(cls, data: dict, duration_s: float | None = None, start_time_s: float | None = None, end_time_s: float | None = None) -> "AnnotationStore":
        metadata = data.get("metadata") or {}
        start = float(metadata.get("time_start_s", 0.0) if start_time_s is None else start_time_s)
        end = metadata.get("time_end_s") if end_time_s is None else end_time_s
        store = cls(duration_s=duration_s, start_time_s=start, end_time_s=float(end) if end is not None else None)
        schema = data.get("schema", data.get("annotation_schema", ""))
        if schema and schema not in {"electrotrace.annotation/v2", "v1"}:
            raise ValueError(f"unsupported annotation schema: {schema}")
        for raw in data.get("annotations", []):
            store.add(Annotation(**raw))
        return store

    @classmethod
    def from_json(cls, text: str, duration_s: float | None = None, start_time_s: float | None = None, end_time_s: float | None = None) -> "AnnotationStore":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(data, duration_s=duration_s, start_time_s=start_time_s, end_time_s=end_time_s)

    def to_csv_rows(self, source_file: str = "") -> list[dict]:
        return [{"file": source_file, "id": a.id, "type": a.type, "label": a.label, "channel": a.channel, "start": a.start, "end": a.end, "time": a.time, "confidence": a.confidence, "notes": a.notes, "annotator": a.annotator, "status": a.status, "reviewer": a.reviewer, "review_notes": a.review_notes} for a in self._items]


def point_agreement(a: list[Annotation], b: list[Annotation], tolerance_s: float = 0.04) -> dict:
    if tolerance_s <= 0 or not math.isfinite(tolerance_s):
        raise ValueError("tolerance_s must be positive and finite")
    left = [x for x in a if x.type == "point"]
    right = [x for x in b if x.type == "point"]
    used: set[str] = set()
    errors = []
    matches = 0
    for x in left:
        candidates = [
            y for y in right
            if y.id not in used
            and y.label == x.label
            and y.channel == x.channel
            and y.time is not None
            and x.time is not None
            and abs(y.time - x.time) <= tolerance_s
        ]
        if candidates:
            y = min(candidates, key=lambda z: abs(z.time - x.time))
            used.add(y.id)
            matches += 1
            errors.append(abs(y.time - x.time))
    total = max(len(left), len(right), 1)
    return {"matches": matches, "agreement_rate": matches / total, "mean_absolute_error_s": sum(errors) / len(errors) if errors else None}


def interval_iou(a: Annotation, b: Annotation) -> float:
    if a.type != "interval" or b.type != "interval":
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))  # type: ignore[arg-type]
    union = max(a.end, b.end) - min(a.start, b.start)  # type: ignore[arg-type]
    return inter / union if union else 0.0
