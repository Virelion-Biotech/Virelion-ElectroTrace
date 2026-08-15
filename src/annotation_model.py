"""
Annotation data model for the ECG Trace Annotator.

Supports two annotation kinds:
  - interval: has start/end (QRS, P wave, T wave, artifact, arrhythmia, etc.)
  - point:    has a single time (R peak, pacing spike, activation time, etc.)

The schema is intentionally extensible -- labels are free-form strings so the
same tool works for clinical ECG, pacemaker experiments, graft
electrophysiology, animal studies, etc.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal

AnnotationType = Literal["interval", "point"]

DEFAULT_LABELS = [
    "P Wave",
    "QRS",
    "T Wave",
    "Pacemaker Spike",
    "R Peak",
    "Artifact",
    "Abnormal Beat",
    "Graft Activation",
    "Arrhythmia",
    "Custom",
]


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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self) -> None:
        if self.type == "interval":
            if self.start is None or self.end is None:
                raise ValueError("Interval annotations require start and end")
            if self.end <= self.start:
                raise ValueError("end must be greater than start")
        elif self.type == "point":
            if self.time is None:
                raise ValueError("Point annotations require time")
        else:
            raise ValueError(f"Unknown annotation type: {self.type}")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop irrelevant fields depending on type, keeps exports tidy
        if self.type == "interval":
            d.pop("time", None)
        else:
            d.pop("start", None)
            d.pop("end", None)
        return d


class AnnotationStore:
    """In-memory collection of annotations for the currently loaded recording."""

    def __init__(self) -> None:
        self._items: List[Annotation] = []

    def add(self, ann: Annotation) -> Annotation:
        ann.validate()
        self._items.append(ann)
        self._sort()
        return ann

    def update(self, ann_id: str, **changes) -> Optional[Annotation]:
        for i, a in enumerate(self._items):
            if a.id == ann_id:
                updated = Annotation(**{**asdict(a), **changes})
                updated.validate()
                self._items[i] = updated
                self._sort()
                return updated
        return None

    def delete(self, ann_id: str) -> bool:
        before = len(self._items)
        self._items = [a for a in self._items if a.id != ann_id]
        return len(self._items) != before

    def duplicate(self, ann_id: str) -> Optional[Annotation]:
        for a in self._items:
            if a.id == ann_id:
                new_ann = Annotation(**{**asdict(a), "id": uuid.uuid4().hex[:8]})
                self._items.append(new_ann)
                self._sort()
                return new_ann
        return None

    def clear(self) -> None:
        self._items = []

    def _sort_key(self, a: Annotation):
        return a.start if a.type == "interval" else a.time

    def _sort(self) -> None:
        self._items.sort(key=self._sort_key)

    @property
    def items(self) -> List[Annotation]:
        return list(self._items)

    def to_json(self, source_file: str = "") -> str:
        payload = {
            "file": source_file,
            "annotation_schema": "v1",
            "annotations": [a.to_dict() for a in self._items],
        }
        return json.dumps(payload, indent=2)

    def to_csv_rows(self, source_file: str = "") -> List[dict]:
        rows = []
        for a in self._items:
            rows.append(
                {
                    "file": source_file,
                    "id": a.id,
                    "type": a.type,
                    "label": a.label,
                    "channel": a.channel,
                    "start": a.start if a.type == "interval" else "",
                    "end": a.end if a.type == "interval" else "",
                    "time": a.time if a.type == "point" else "",
                    "confidence": a.confidence,
                    "notes": a.notes,
                }
            )
        return rows

    @classmethod
    def from_json(cls, text: str) -> "AnnotationStore":
        store = cls()
        data = json.loads(text)
        for item in data.get("annotations", []):
            store._items.append(Annotation(**item))
        store._sort()
        return store
