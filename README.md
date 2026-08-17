# ElectroTrace

ElectroTrace is a research-oriented ECG/electrophysiology annotation and dataset-generation platform. The original Streamlit MVP has been removed; the current application is a lightweight Flask API with a browser-based Plotly interface.

## Core capabilities

- **Native import:** CSV, EDF/EDF+, and zipped WFDB records (`.hea` + signal files in a ZIP).
- **Signal validation:** monotonic timestamps, rejection of missing/infinite samples, sampling-rate inference, and irregular-sampling warnings.
- **Interactive waveform review:** multi-channel Plotly visualization, zoom, range selection, point capture, and non-destructive display filters.
- **Beat segmentation:** R-peak detection followed by configurable beat windows with RR interval and heart-rate metadata.
- **Structured annotation:** point and interval labels, confidence, notes, annotator identity, review status, and provenance.
- **QC/review:** accepted / unreviewed / flagged workflow plus basic multi-annotator point agreement.
- **Model-assisted labeling:** train a Random Forest classifier from accepted point annotations and rank unlabeled beats by predictive uncertainty.
- **Active learning:** surface the most uncertain beats first; model suggestions remain explicitly unreviewed until human confirmation.
- **Research exports:** versioned JSON and CSV annotations including source format, absolute recording time bounds, preprocessing, and beat-count metadata.
- **Research analysis:** beat-level phenotype summaries, group statistics, FDR adjustment, and leakage-safe subject-level ML benchmarking.

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

Open `http://127.0.0.1:5000`.

## Input formats

### CSV

```csv
time,Lead_I,Lead_II
0.000,0.012,0.031
0.002,0.015,0.034
0.004,0.020,0.038
```

The loader recognizes `time`, `t`, `timestamp`, `time_s`, and `seconds` as time columns. Missing/non-numeric signal samples are rejected rather than silently converted to zero.

### EDF

Upload an `.edf`/EDF+ recording directly. Channels must use a common sampling rate in the current implementation.

### WFDB

Upload a `.zip` archive containing a WFDB header (`.hea`) and its matching signal file(s). ElectroTrace reads the record into a common representation for annotation and analysis. Uploads are bounded and ZIP paths are validated before extraction.

## Beat workflow

```text
Recording
   ↓
Validation / import
   ↓
Display preprocessing (raw signal preserved)
   ↓
R-peak detection
   ↓
Beat windows + RR / HR
   ↓
Human annotations
   ↓
Accepted labels
   ↓
Model training
   ↓
Uncertainty ranking
   ↓
Human review of highest-value candidates
   ↓
Exported research dataset
```

The model-assisted stage is intentionally conservative. Training requires accepted point annotations from at least two classes. Suggestions are ranked by predictive entropy and are exported as `unreviewed` candidates with provenance notes.

## Leakage-safe benchmarking

Benchmark inputs contain beat-level features plus a subject/group identifier. ElectroTrace uses **StratifiedGroupKFold** so observations from the same subject are kept out of the test fold when they are used for model benchmarking. The tool also checks that each outcome class has enough distinct subjects for the requested folds.

## Absolute recording time

Recordings are not assumed to start at `t = 0`. Annotation exports preserve `time_start_s` and `time_end_s`, and the browser keeps annotations in the recording's absolute time coordinate system when switching signal windows.

## Annotation schema

Exports use `electrotrace.annotation/v2`:

```json
{
  "schema": "electrotrace.annotation/v2",
  "file": "recording.edf",
  "metadata": {
    "sampling_rate_hz": 500,
    "duration_s": 120.0,
    "time_start_s": 0.0,
    "time_end_s": 119.998,
    "channels": ["II"],
    "source_format": "EDF",
    "annotator": "annotator_1",
    "preprocessing": {},
    "beat_count": 119
  },
  "annotations": []
}
```

## Architecture

```text
                 ┌───────────────┐
CSV / EDF / WFDB │ import + QC   │
───────────────► └───────┬───────┘
                         │
                  normalized signal
                         │
           ┌─────────────┴─────────────┐
           │                           │
      Plotly browser             Python analysis
           │                           │
     annotation/QC             filters + beats
           │                           │
           └─────────────┬─────────────┘
                         │
                 accepted labels
                         │
                  ML / active learning
                         │
                uncertainty-ranked beats
                         │
                  reviewed dataset
```

## Repository layout

```text
.
├── server.py
├── web/
│   ├── index.html
│   ├── app.js
│   ├── research.js
│   ├── shortcuts.js
│   └── styles.css
├── src/electrotrace/
│   ├── annotations.py
│   ├── beats.py
│   ├── benchmark.py
│   ├── formats.py
│   ├── io.py
│   ├── ml.py
│   ├── phenotype.py
│   ├── project.py
│   ├── project_store.py
│   ├── signal.py
│   └── statistics.py
├── tests/
└── sample_data/
```

## Testing

```bash
pytest -q
```

CI runs the test suite on Python 3.10–3.12.

## Scientific-use notes

ElectroTrace is research software, not a clinical diagnostic device. Automatic R-peak detection is a candidate generator, not a validated clinical algorithm. Model-assisted suggestions are only as good as the accepted annotations used for training and must remain human-reviewed.

Raw signal arrays are never overwritten by display preprocessing. Filter configuration, source format, absolute time bounds, and annotation/review provenance are included in exports so downstream analyses remain auditable.

## License

MIT
