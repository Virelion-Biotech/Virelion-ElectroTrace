"""Recording metadata and project provenance helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError("file_id expects the file contents as bytes")
    return hashlib.sha256(bytes(file_bytes)).hexdigest()[:16]


def make_metadata(filename: str, sampling_rate_hz: float | None, duration_s: float | None, channels: list[str], annotator: str = "", source: str = "", file_bytes: bytes | None = None) -> RecordingMetadata:
    identifier_source = file_bytes if file_bytes is not None else filename.encode("utf-8")
    return RecordingMetadata(
        recording_id=file_id(identifier_source),
        filename=filename,
        sampling_rate_hz=sampling_rate_hz,
        duration_s=duration_s,
        channels=channels,
        annotator=annotator,
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(),
        preprocessing={},
    )
