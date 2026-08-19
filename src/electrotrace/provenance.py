"""Deterministic study and dataset provenance manifests.

The manifest model is intentionally dependency-light so it can be used by
validation, phenotype, and machine-learning workflows without coupling those
workflows to the web application.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = "electrotrace-study-manifest-v1"


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, set):
        return sorted(_clean(v) for v in value)
    return value


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable description of a research dataset and its processing state."""

    dataset_id: str
    dataset_version: str
    source: str
    records: tuple[str, ...]
    subject_ids: tuple[str, ...] = ()
    annotation_policy: str = ""
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    detector_config: Mapping[str, Any] = field(default_factory=dict)
    split_manifest: Mapping[str, Sequence[str]] = field(default_factory=dict)
    calibration_records: tuple[str, ...] = ()
    software_version: str = ""
    software_commit: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        required = {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source": self.source,
            "software_version": self.software_version,
            "software_commit": self.software_commit,
        }
        missing = [key for key, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Manifest fields must be non-empty: {', '.join(missing)}")
        records = tuple(str(x) for x in self.records)
        if not records:
            raise ValueError("Manifest must contain at least one record")
        if len(set(records)) != len(records):
            raise ValueError("Manifest records must be unique")
        calibration = set(self.calibration_records)
        if not calibration.issubset(set(records)):
            raise ValueError("Calibration records must be a subset of records")
        split_records: list[str] = []
        for split_name, split in self.split_manifest.items():
            if not str(split_name).strip():
                raise ValueError("Split names must be non-empty")
            values = [str(x) for x in split]
            if len(values) != len(set(values)):
                raise ValueError(f"Split '{split_name}' contains duplicate records")
            split_records.extend(values)
        if split_records and set(split_records) != set(records):
            raise ValueError("Split manifest must partition the complete record list")
        if len(split_records) != len(set(split_records)):
            raise ValueError("A record may occur in only one split")
        try:
            datetime.fromisoformat(self.generated_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("generated_at_utc must be ISO-8601") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["records"] = list(self.records)
        data["subject_ids"] = list(self.subject_ids)
        data["calibration_records"] = list(self.calibration_records)
        data["preprocessing"] = _clean(self.preprocessing)
        data["detector_config"] = _clean(self.detector_config)
        data["split_manifest"] = _clean(self.split_manifest)
        return _clean(data)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_generated_timestamp(self, timestamp: str) -> "DatasetManifest":
        return DatasetManifest(**{**self.to_dict(), "generated_at_utc": timestamp, "records": tuple(self.records), "subject_ids": tuple(self.subject_ids), "calibration_records": tuple(self.calibration_records)})


def manifest_from_dict(data: Mapping[str, Any]) -> DatasetManifest:
    """Construct and validate a manifest from serialized JSON-like data."""
    manifest = DatasetManifest(
        dataset_id=str(data["dataset_id"]),
        dataset_version=str(data["dataset_version"]),
        source=str(data["source"]),
        records=tuple(str(x) for x in data["records"]),
        subject_ids=tuple(str(x) for x in data.get("subject_ids", ())),
        annotation_policy=str(data.get("annotation_policy", "")),
        preprocessing=dict(data.get("preprocessing", {})),
        detector_config=dict(data.get("detector_config", {})),
        split_manifest={str(k): tuple(str(x) for x in v) for k, v in data.get("split_manifest", {}).items()},
        calibration_records=tuple(str(x) for x in data.get("calibration_records", ())),
        software_version=str(data.get("software_version", "")),
        software_commit=str(data.get("software_commit", "")),
        generated_at_utc=str(data.get("generated_at_utc", "")),
        schema_version=str(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
    )
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {manifest.schema_version}")
    manifest.validate()
    return manifest
