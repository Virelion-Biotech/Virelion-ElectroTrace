"""HeartTwin local-command adapter for ElectroTrace electrical.analyze."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .formats import load_electrophysiology
from .io import load_csv, validate_dataframe


def _find_input(payload: dict) -> tuple[str, dict]:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "electrical" and "input_path" in obs.get("values", {}):
            return str(obs["values"]["input_path"]), obs["values"]
    raise ValueError("No 'electrical' observation with values.input_path was provided")


def _analyze_file(file_path: Path, params: dict) -> dict:
    raw = file_path.read_bytes()
    filename = file_path.name
    if file_path.suffix.lower() == ".csv":
        dataframe = load_csv(raw)
        result = validate_dataframe(dataframe, str(params.get("time_col") or "") or None)
        output = {
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "time_col": result.time_col,
            "signal_cols": result.signal_cols,
            "sampling_rate_hz": result.sampling_rate_hz,
            "duration_s": result.duration_s,
            "n_samples": result.n_samples,
            "time_start_s": float(dataframe[result.time_col].iloc[0]) if result.valid else None,
            "time_end_s": float(dataframe[result.time_col].iloc[-1]) if result.valid else None,
            "format": "csv",
            "source_format": "CSV",
            "filename": filename,
        }
        if result.valid:
            output["time"] = dataframe[result.time_col].astype(float).tolist()
            output["signals"] = {
                channel: dataframe[channel].astype(float).to_numpy().tolist()
                for channel in result.signal_cols
            }
        return output

    output = load_electrophysiology(raw, filename)
    output.update(
        valid=True,
        errors=[],
        warnings=[],
        format=output["source_format"].lower(),
        filename=filename,
        time_start_s=output["time"][0],
        time_end_s=output["time"][-1],
    )
    return output


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        path, params = _find_input(payload)
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        body = _analyze_file(file_path, params)
        if not body.get("valid", False):
            raise RuntimeError("; ".join(body.get("errors") or ["ElectroTrace analysis failed"]))
        print(
            json.dumps(
                {"entity_id": payload.get("entity_id"), "electrical_analysis": body},
                allow_nan=False,
                default=str,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
