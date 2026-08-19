"""Deterministic study and dataset provenance manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = "electrotrace-study-manifest-v2"


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
    record_subject_map: Mapping[str, str] = field(default_factory=dict)
    input_files: Mapping[str, str] = field(default_factory=dict)
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
        records = tuple(str(x).strip() for x in self.records)
        if not records or any(not x for x in records):
            raise ValueError("Manifest records must be non-empty")
        if len(set(records)) != len(records):
            raise ValueError("Manifest records must be unique")

        mapping = {str(k).strip(): str(v).strip() for k, v in self.record_subject_map.items()}
        if mapping:
            if set(mapping) != set(records):
                raise ValueError("record_subject_map must contain exactly one entry for every record")
            if any(not record or not subject for record, subject in mapping.items()):
                raise ValueError("record_subject_map identifiers must be non-empty")
        elif self.subject_ids:
            if len(self.subject_ids) != len(records):
                raise ValueError("subject_ids must align one-to-one with records")
            if any(not str(x).strip() for x in self.subject_ids):
                raise ValueError("subject_ids must be non-empty")
            mapping = dict(zip(records, (str(x).strip() for x in self.subject_ids)))
        elif self.subject_ids == ():
            mapping = {}

        declared_subjects = {str(x).strip() for x in self.subject_ids if str(x).strip()}
        if declared_subjects and declared_subjects != set(mapping.values()):
            raise ValueError("subject_ids must match record_subject_map values")

        for relative_path, sha256 in self.input_files.items():
            if not str(relative_path).strip():
                raise ValueError("input_files paths must be non-empty")
            digest = str(sha256).strip().lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid SHA-256 for input file '{relative_path}'")

        calibration = {str(x) for x in self.calibration_records}
        if not calibration.issubset(set(records)):
            raise ValueError("Calibration records must be a subset of records")
        split_records: list[str] = []
        for split_name, split in self.split_manifest.items():
            if not str(split_name).strip():
                raise ValueError("Split names must be non-empty")
            values = [str(x).strip() for x in split]
            if any(not x for x in values):
                raise ValueError(f"Split '{split_name}' contains an empty record identifier")
            if len(values) != len(set(values)):
                raise ValueError(f"Split '{split_name}' contains duplicate records")
            split_records.extend(values)
        if split_records and set(split_records) != set(records):
            raise ValueError("Split manifest must partition the complete record list")
        if len(split_records) != len(set(split_records)):
            raise ValueError("A record may occur in only one split")
        try:
            datetime.fromisoformat(self.generated_at_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("generated_at_utc must be ISO-8601") from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["records"] = [str(x).strip() for x in self.records]
        data["subject_ids"] = [str(x).strip() for x in self.subject_ids]
        data["record_subject_map"] = _clean(self.record_subject_map)
        data["input_files"] = _clean(self.input_files)
        data["calibration_records"] = [str(x).strip() for x in self.calibration_records]
        data["preprocessing"] = _clean(self.preprocessing)
        data["detector_config"] = _clean(self.detector_config)
        data["split_manifest"] = _clean(self.split_manifest)
        return _clean(data)

    def canonical_json(self) -> str:
        """Canonical identity/configuration payload used for provenance hashing."""
        data = self.to_dict()
        data.pop("generated_at_utc", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def manifest_from_dict(data: Mapping[str, Any]) -> DatasetManifest:
    """Construct and validate a manifest from serialized JSON-like data."""
    manifest = DatasetManifest(
        dataset_id=str(data["dataset_id"]),
        dataset_version=str(data["dataset_version"]),
        source=str(data["source"]),
        records=tuple(str(x) for x in data["records"]),
        subject_ids=tuple(str(x) for x in data.get("subject_ids", ())),
        record_subject_map={str(k): str(v) for k, v in data.get("record_subject_map", {}).items()},
        input_files={str(k): str(v) for k, v in data.get("input_files", {}).items()},
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
