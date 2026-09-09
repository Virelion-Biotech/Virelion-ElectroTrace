"""HeartTwin local-command adapter for ElectroTrace electrical.analyze."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .server_app import app


def _find_input(payload: dict) -> tuple[str, dict]:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "electrical" and "input_path" in obs.get("values", {}):
            return str(obs["values"]["input_path"]), obs["values"]
    raise ValueError("No 'electrical' observation with values.input_path was provided")


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
        with app.test_client() as client, file_path.open("rb") as handle:
            response = client.post(
                "/api/analyze",
                data={"file": (handle, file_path.name), "time_col": str(params.get("time_col", ""))},
                content_type="multipart/form-data",
            )
        body = response.get_json(silent=True) or {"error": response.get_data(as_text=True)}
        if response.status_code >= 400:
            raise RuntimeError(body.get("error", "ElectroTrace analysis failed"))
        print(json.dumps({"entity_id": payload.get("entity_id"), "electrical_analysis": body}, allow_nan=False, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
