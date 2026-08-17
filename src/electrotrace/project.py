"""Recording metadata and project provenance helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib

@dataclass
class RecordingMetadata:
    recording_id: str
    filename: str
    sampling_rate_hz: float | None
    duration_s: float | None
    channels: list[str]
    annotator: str = ""
    source: str = ""
    created_at: str = ""
    preprocessing: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

def file_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]

def make_metadata(filename: str, sampling_rate_hz: float | None, duration_s: float | None, channels: list[str], annotator: str = "", source: str = "") -> RecordingMetadata:
    return RecordingMetadata(
        recording_id=file_id(filename.encode("utf-8")),
        filename=filename,
        sampling_rate_hz=sampling_rate_hz,
        duration_s=duration_s,
        channels=channels,
        annotator=annotator,
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(),
        preprocessing={},
    )
