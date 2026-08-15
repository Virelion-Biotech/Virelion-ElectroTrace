# ECG Trace Annotator

A lightweight Streamlit app for building labeled ECG datasets: **Upload → Visualize → Annotate → Review → Export**.

## Features (MVP)

- Upload a CSV with a time column + one or more signal channels (auto-detects the time column, or pick manually)
- Interactive Plotly waveform: zoom, pan, multi-channel
- **Box-select on the chart** to pre-fill a new interval annotation (or enter times manually)
- Point annotations (R peak, pacing spike, etc.) and interval annotations (QRS, P/T wave, artifact, etc.)
- Non-destructive optional filtering for display only (baseline wander removal, low-pass, high-pass, notch) — raw data is never modified
- Annotation table: view, duplicate, delete
- Export to JSON and CSV; re-import a previously saved `annotations.json`
- Built-in synthetic sample ECG so you can try it with no file of your own

## Quickstart

```bash
git clone <your-repo-url>
cd ecg-trace-annotator
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), and click **Load sample data** in the sidebar to try it immediately.

## CSV format

```csv
time,Lead_I,Lead_II
0.000,0.012,0.031
0.004,0.015,0.034
...
```

- `time` should be in seconds.
- Any number of additional signal columns is supported; pick which ones to display/annotate in the sidebar.
- Sampling rate is inferred automatically from the time column.

## Annotation schema

Annotations are stored as:

```json
{
  "file": "sample_ecg.csv",
  "annotation_schema": "v1",
  "annotations": [
    {
      "id": "a1b2c3d4",
      "type": "interval",
      "label": "QRS",
      "channel": "Lead_II",
      "start": 1.42,
      "end": 1.51,
      "confidence": 0.98,
      "notes": ""
    },
    {
      "id": "e5f6a7b8",
      "type": "point",
      "label": "R_peak",
      "channel": "Lead_II",
      "time": 1.465,
      "confidence": 0.99,
      "notes": ""
    }
  ]
}
```

CSV export flattens this into one row per annotation (`file,id,type,label,channel,start,end,time,confidence,notes`) for use in ML pipelines.

## Project layout

```
ecg-trace-annotator/
├── app.py                     # Streamlit application (main entry point)
├── src/
│   ├── annotation_model.py    # Annotation + AnnotationStore data model
│   ├── io_utils.py            # CSV loading, column/sampling-rate inference
│   └── signal_processing.py   # Optional non-destructive filters
├── sample_data/
│   └── sample_ecg.csv         # Synthetic demo recording
├── requirements.txt
└── README.md
```

## Roadmap (not yet implemented)

These are natural next steps if you want to extend the tool, per the original design doc:

- PDF/image ECG upload with bounding-box annotation (pixel coords, optional grid calibration to time)
- Keyboard shortcuts for fast labeling (Q/P/T/A, arrows to move cursor, etc.)
- Undo/redo
- Multi-file "project" mode (`project.json` + one folder per recording)
- Multi-annotator agreement metrics (IoU, Cohen's kappa)
- ML-assisted pre-labeling (automatic R-peak/QRS detector + human review queue)
- YOLO/COCO/segmentation export for image-based annotation

## Deploying

Works as-is on [Streamlit Community Cloud](https://streamlit.io/cloud): push this repo to GitHub, point Streamlit Cloud at `app.py`, done.

## License

MIT — adapt freely for your research.
