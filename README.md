# ElectroTrace

ElectroTrace is a research-oriented ECG/electrophysiology annotation tool for turning waveform recordings into structured, reviewable datasets.

The original MVP used Streamlit. **Streamlit has been removed.** The current application is a lightweight Flask API plus a browser-based JavaScript/Plotly interface.

## What it does

- Upload and validate ECG CSV recordings.
- Infer sampling rate from the time axis and warn about irregular sampling.
- Plot multiple channels with zoom, pan, and range selection.
- Create point annotations such as R peaks and interval annotations such as QRS/P/T/artifact regions.
- Edit, duplicate, delete, accept, and flag annotations.
- Record annotator and study/source metadata.
- Apply non-destructive display filtering: baseline removal, high-pass, low-pass, and notch.
- Preserve raw signals unchanged and export the active preprocessing configuration.
- Run an automatic R-peak candidate detector; candidates are explicitly marked for human review.
- Import another annotator's JSON export and calculate basic point-level agreement.
- Export JSON and CSV annotation datasets.
- Use the Python package independently of the web UI for IO, validation, signal processing, and annotation/QC logic.

## Quickstart

Python 3.10+ is recommended.

```bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -U pip
pip install -e .
python server.py
```

Open `http://127.0.0.1:5000` in a browser.

The **Load sample** button uses the repository's synthetic ECG so the interface can be tested without preparing a recording first.

## CSV format

The minimum expected structure is a monotonic time column plus one or more numeric signal columns.

```csv
time,Lead_I,Lead_II
0.000,0.012,0.031
0.002,0.015,0.034
0.004,0.020,0.038
```

Recognized time-column names include `time`, `t`, `timestamp`, `time_s`, and `seconds`. Other column names can still be loaded through the Python IO layer by explicitly specifying the time column.

Validation checks include missing time axes, non-monotonic timestamps, NaN/infinite values, empty signal channels, positive duration, and irregular sampling intervals.

## Annotation schema

Exports use `electrotrace.annotation/v2` and include provenance fields alongside annotations.

```json
{
  "schema": "electrotrace.annotation/v2",
  "file": "recording.csv",
  "metadata": {
    "sampling_rate_hz": 500,
    "duration_s": 12.4,
    "channels": ["Lead_II"],
    "annotator": "annotator_1",
    "source": "study_A",
    "preprocessing": {
      "baseline": false,
      "highpass_hz": 0.5,
      "lowpass_hz": 40,
      "notch_hz": null
    }
  },
  "annotations": [
    {
      "id": "example-id",
      "label": "QRS",
      "type": "interval",
      "channel": "Lead_II",
      "start": 1.42,
      "end": 1.51,
      "time": null,
      "confidence": 0.98,
      "notes": "",
      "annotator": "annotator_1",
      "status": "accepted",
      "reviewer": "reviewer_1",
      "review_notes": ""
    }
  ]
}
```

Statuses are `unreviewed`, `accepted`, and `flagged`. Automated R-peak candidates are created as `unreviewed` with a note that they require human verification.

## Architecture

```text
CSV upload
   |
   v
Flask API ---- validation / IO ----> normalized recording
   |                                      |
   |                                      v
   |                                browser Plotly view
   |                                      |
   +---- signal processing <-------------+
   |
   +---- R-peak candidate detection

Browser state
   |
   +---- annotation editor
   +---- review / QC
   +---- JSON / CSV export
```

### Repository layout

```text
.
├── server.py                  # Flask application/API
├── web/
│   ├── index.html             # browser UI
│   ├── app.js                 # annotation/QC workflow
│   └── styles.css
├── src/electrotrace/
│   ├── __init__.py
│   ├── annotations.py         # annotation model, review, agreement
│   ├── io.py                  # CSV loading and validation
│   ├── project.py             # recording provenance helpers
│   └── signal.py              # validated non-destructive filters
├── tests/
│   ├── test_annotations.py
│   ├── test_io.py
│   ├── test_signal.py
│   └── test_api.py
├── sample_data/sample_ecg.csv
├── pyproject.toml
└── requirements.txt
```

## Testing

Run:

```bash
pytest
```

The tests cover annotation validation/round-tripping, annotation CRUD, point agreement, interval IoU, CSV validation, sampling-rate inference, filtering immutability, filter parameter validation, and basic HTTP API behavior.

## Scientific-use notes

ElectroTrace is an annotation and dataset-preparation tool, not a clinical diagnostic device. Automatic peak detection is deliberately conservative in the workflow: generated candidates remain unreviewed until a human accepts them.

Filtering is display-only. Raw signal arrays are copied before processing and are never overwritten. Filter settings are exported so annotation provenance can be reconstructed.

For multi-annotator projects, use separate JSON exports for each annotator and compare them through the Review/QC panel. The current agreement implementation provides point-event timing agreement within a configurable tolerance in the Python library; interval IoU is also available programmatically. A future production workflow can add formal consensus rules and richer categorical agreement statistics.

## Roadmap

The foundation for a more serious dataset-generation platform is now in place. Logical next additions are:

1. Multi-file project management with persistent `project.json` metadata.
2. Full interval-level agreement reports and reviewer consensus tools.
3. Keyboard-first annotation controls for high-throughput labeling.
4. WFDB/EDF input and standardized export.
5. Model-assisted QRS/beat classification with confidence-aware human review.
6. Optional image ECG annotation with calibrated time/amplitude coordinates.

## License

MIT
