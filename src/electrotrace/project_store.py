"""Persistent project metadata and chunked signal access."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RecordingRef:
    recording_id: str
    subject_id: str
    group: str = ""
    visit: str = ""
    source_path: str = ""
    format: str = ""
    sampling_rate_hz: float | None = None
    duration_s: float | None = None
    channels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    project_id: str
    name: str
    created_at: str = ""
    updated_at: str = ""
    recordings: list[RecordingRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ProjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recordings_dir = self.root / "recordings"
        self.recordings_dir.mkdir(exist_ok=True)
        self.annotations_dir = self.root / "annotations"
        self.annotations_dir.mkdir(exist_ok=True)
        self.project_path = self.root / "project.json"

    def load(self) -> Project:
        if not self.project_path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return Project(project_id=self.root.name, name=self.root.name, created_at=now, updated_at=now)
        try:
            data = json.loads(self.project_path.read_text(encoding="utf-8"))
            refs = [RecordingRef(**r) for r in data.get("recordings", [])]
            return Project(project_id=data["project_id"], name=data["name"], created_at=data.get("created_at", ""), updated_at=data.get("updated_at", ""), recordings=refs)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"project metadata is invalid or unreadable: {exc}") from exc

    def save(self, project: Project) -> None:
        project.updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(project.to_dict(), indent=2)
        temp_path = self.project_path.with_suffix(".json.tmp")
        try:
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(self.project_path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"could not save project metadata: {exc}") from exc

    def add_recording(self, ref: RecordingRef) -> Project:
        project = self.load()
        project.recordings = [r for r in project.recordings if r.recording_id != ref.recording_id]
        project.recordings.append(ref)
        self.save(project)
        return project

    def window(self, recording_path: str | Path, start: int, stop: int) -> dict[str, np.ndarray]:
        """Read a bounded sample window without materializing a full recording."""
        path = Path(recording_path)
        if start < 0 or stop <= start:
            raise ValueError("window must satisfy 0 <= start < stop")
        if path.suffix.lower() == ".npy":
            arr = np.load(path, mmap_mode="r")
            end = min(stop, arr.shape[0])
            return {"signal": np.asarray(arr[start:end])}
        if path.suffix.lower() == ".npz":
            # NPZ archives do not provide true mmap semantics; close the archive after reading.
            with np.load(path) as data:
                return {k: np.asarray(data[k][start:min(stop, data[k].shape[0])]) for k in data.files}
        if path.suffix.lower() == ".csv":
            import pandas as pd
            header = pd.read_csv(path, nrows=0)
            if len(header.columns) == 0:
                raise ValueError("CSV has no columns")
            frame = pd.read_csv(path, skiprows=range(1, start + 1), nrows=stop - start)
            return {str(c): frame[c].to_numpy() for c in frame.columns}
        raise ValueError(f"Unsupported chunked format: {path.suffix}")
