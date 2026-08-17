# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, and leakage-safe machine learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated, publication-ready datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.2.0  
**License:** MIT  
**Status:** Active research software

## Quick Start

### Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e .
```

### Running the Server

```bash
python server.py
```

Open `http://127.0.0.1:5000` in your browser.

**Configuration via environment variables:**
- `ELECTROTRACE_HOST` – server listen address (default: `127.0.0.1`)
- `ELECTROTRACE_PORT` – server port (default: `5000`)
- `ELECTROTRACE_UPLOAD_ROOT` – temporary directory for uploaded files (default: system temp)
- `ELECTROTRACE_PROJECT_ROOT` – persistent project storage (default: `./projects`)

## Core Capabilities

### Recording Import & Validation

- **CSV import:** time column + signal columns. Recognizes `time`, `t`, `timestamp`, `time_s`, `seconds` as time columns.
- **EDF/EDF+ support:** multi-channel recordings with automatic sampling rate detection.
- **WFDB ZIP archives:** compressed records with `.hea` header + signal files.
- **Validation:** monotonic timestamps, rejection of NaN/infinite samples, automatic sampling rate inference, irregular-sampling warnings.
- **Non-destructive preprocessing:** raw signal arrays preserved; filters applied only for display.

### Interactive Annotation

- **Multi-channel Plotly visualization:** zoom, pan, drag-to-select intervals, click-to-capture points.
- **Point and interval annotations:** flexible labeling with confidence scores and notes.
- **Annotator identity & review states:** unreviewed, accepted, flagged workflow.
- **Real-time QC dashboard:** annotation count, review status summary, multi-annotator agreement metrics.

### Beat Segmentation & Phenotyping

- **R-peak detection:** automatic candidate generation (not validated clinical-grade) using peak prominence and distance thresholding.
- **Configurable beat windows:** default 0.35 s pre-R, 0.55 s post-R; adjustable per session.
- **Beat-level features:** RR intervals (previous and next), heart rate, R-wave amplitude, QRS width proxy, baseline estimation.
- **Phenotype summaries:** beat counts, heart rate statistics, and group-level comparisons.

### Machine-Assisted Labeling

- **Random Forest model training:** learns from human-accepted point annotations across labeled beats.
- **Predictive uncertainty ranking:** surfaces uncertain beats for human review in active-learning fashion.
- **Explicit review workflow:** model suggestions remain unreviewed until human confirmation; prevents silent errors.
- **Threshold controls:** minimum accepted annotations (≥4 across ≥2 classes) to prevent low-signal model training.

### Research-Grade Analysis & Exports

- **Leakage-safe benchmarking:** StratifiedGroupKFold ensures beats from the same subject never leak into validation folds.
- **Multi-model comparison:** Logistic Regression and Random Forest with metrics (accuracy, balanced accuracy, macro/weighted F1, ROC-AUC).
- **Phenotype statistics:** Welch t-test, Mann-Whitney U, Cohen's d for group comparisons.
- **FDR adjustment:** Benjamini-Hochberg correction for multiple hypothesis testing.
- **Versioned JSON export:** `electrotrace.annotation/v2` schema with full provenance (source format, absolute time bounds, preprocessing, annotator, review status).
- **CSV export:** flat table of annotations with all metadata for downstream analysis.

### Project & Recording Management

- **Project storage:** persistent metadata in JSON (project name, created/updated timestamps, recording inventory).
- **Subject/group/visit tracking:** organize recordings by study structure; enables stratified ML workflows.
- **Windowed access:** load large recordings in sample-based chunks for memory-efficient batch processing.
- **Safe archive extraction:** validates ZIP structure; prevents path traversal and symbolic link attacks.

## How It's Organized

```
.
├── server.py                 Flask REST API server + static file serving
├── web/                      Browser frontend (HTML, CSS, JavaScript)
│   ├── index.html            Main UI shell
│   ├── app.js                Core application state, file I/O, plotting
│   ├── research.js           Project storage, phenotyping, benchmarking
│   ├── shortcuts.js          Keyboard bindings
│   └── styles.css            Layout and theming
├── src/electrotrace/         Python analysis backend (89% of repo)
│   ├── __init__.py           Package metadata (v1.2.0)
│   ├── io.py                 CSV loading, validation, channel labeling
│   ├── formats.py            EDF and WFDB ZIP import + safe extraction
│   ├── signal.py             Butterworth filtering (high-pass, low-pass, notch)
│   ├── beats.py              Beat segmentation from R-peak indices
│   ├── annotations.py        Annotation model, validation, review states
│   ├── ml.py                 Random Forest training, feature extraction, uncertainty ranking
│   ├── phenotype.py          Beat-level cardiac phenotypes (amplitude, rate, intervals)
│   ├── statistics.py         Group comparisons, FDR adjustment
│   ├── benchmark.py          Leakage-safe cross-validation (StratifiedGroupKFold)
│   ├── project.py            Recording metadata helpers (file hashing, timestamps)
│   └── project_store.py      Persistent project JSON + chunked recording access
├── tests/                    pytest suite (Python 3.10–3.12 in CI)
├── sample_data/              Example CSV/EDF recordings for demo
├── pyproject.toml            Build config, dependencies, test settings
├── requirements.txt          Pinned dependency list
└── LICENSE                   MIT
```

## Data Flow

1. **Upload & parse:** User selects CSV/EDF/WFDB ZIP → `server.py` validates & returns normalized signal + metadata
2. **Display & interact:** Browser plots multi-channel signal via Plotly; user selects annotation regions or points
3. **Segment & feature:** User triggers R-peak detection → automatic beat window segmentation + RR/HR metrics
4. **Label & review:** User creates annotations (point/interval); marks as accepted/flagged; system computes QC stats
5. **Train & suggest:** User trains Random Forest on accepted annotations; model ranks uncertain beats
6. **Export & analyze:** User exports JSON (full provenance) or CSV (flat table); performs group statistics + leakage-safe ML benchmarking
7. **Store for reuse:** User saves recording + metadata to project for windowed batch access

## Dependencies

| Dependency | Purpose |
|---|---|
| **Flask ≥ 3.0** | REST API framework |
| **NumPy ≥ 1.24** | Numerical arrays, signal processing |
| **SciPy ≥ 1.10** | Peak detection, filtering, statistics |
| **Pandas ≥ 2.0** | CSV parsing, tabular data |
| **scikit-learn ≥ 1.4** | Random Forest, logistic regression, cross-validation |
| **pyedflib ≥ 0.1.38** | EDF file reading |
| **wfdb ≥ 4.1** | WFDB record import |

**Front-end:**
- Plotly 2.35+ (CDN) for interactive waveform visualization
- Vanilla JavaScript (no framework)

## API Endpoints

### Recording Upload & Analysis
- `POST /api/analyze` – validate and inspect CSV file
- `POST /api/recording` – upload and persist CSV/EDF/WFDB file
- `GET /api/recording/<id>/window` – fetch sample window from stored recording

### Signal Processing
- `POST /api/filter` – apply baseline removal, high-pass, low-pass, notch filters
- `POST /api/detect/r-peaks` – detect R-peak candidates via peak prominence

### Beat Operations
- `POST /api/beats` or `POST /api/segment` – segment beats from R-peaks with configurable windows

### ML & Analysis
- `POST /api/ml/train` – train Random Forest from accepted point annotations
- `POST /api/ml/suggest` – train + return uncertainty-ranked beat suggestions
- `POST /api/phenotype` – extract cardiac phenotypes from beat signal windows
- `POST /api/statistics/compare` – Welch t-test, Mann-Whitney U, Cohen's d
- `POST /api/statistics/fdr` – Benjamini-Hochberg FDR adjustment
- `POST /api/benchmark` – StratifiedGroupKFold cross-validation (Logistic Regression + Random Forest)

### Project Management
- `GET /api/project?name=<project>` – load project metadata
- `POST /api/project/recording` – add recording to project with subject/group/visit info

## Export Schema

**JSON (`electrotrace.annotation/v2`):**
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
  "annotations": [
    {
      "id": "...",
      "label": "QRS",
      "type": "interval",
      "channel": "II",
      "start": 1.5,
      "end": 1.65,
      "confidence": 1.0,
      "status": "accepted",
      "annotator": "annotator_1",
      "notes": ""
    }
  ]
}
```

**CSV export:** flat table with `file`, `id`, `type`, `label`, `channel`, `start`, `end`, `time`, `confidence`, `status`, `annotator`, `notes`.

## Safety & Validation

- **File upload limits:** 512 MB per file; ZIP archives limited to 256 members, 512 MB uncompressed.
- **Path traversal prevention:** ZIP extraction validated; symlinks rejected.
- **Signal validation:** monotonic time, finite values, no NaN/infinite samples.
- **Filter validation:** Nyquist frequency checks, signal length validation for stable zero-phase filtering.
- **Project name validation:** alphanumeric + `.`, `_`, `-`; max 64 characters; path traversal checks.

## Testing

```bash
# Run test suite
pytest -q

# Run with coverage
pytest --cov=src/electrotrace tests/
```

CI runs on Python 3.10, 3.11, 3.12.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection is a candidate generator, not a validated clinical algorithm. Use only for research; clinical applications require regulatory validation.

**Reproducibility & transparency:** Raw signal arrays are never overwritten by display preprocessing. All filter configurations, source formats, absolute time bounds, and annotation/review provenance are preserved in exports. This allows downstream investigators to audit and reproduce analyses.

**Leakage prevention:** The benchmark module enforces subject-level stratification (StratifiedGroupKFold), preventing beats from the same subject from appearing in both training and validation folds. This is essential for realistic generalization estimates.

**Model uncertainty is exploratory:** Trained Random Forest models serve active-learning (uncertainty ranking) and exploratory purposes only. Scores and feature importances should not be over-interpreted; always validate with domain expertise and independent cohorts.

## Development

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[test]"
pytest -q
```

### Adding a New Analysis Module
1. Create `src/electrotrace/new_module.py`
2. Add endpoint to `server.py` (e.g., `@app.post("/api/new_module")`)
3. Add corresponding UI handler in `web/app.js` or `web/research.js`
4. Add unit tests to `tests/`

## Citation

If you use ElectroTrace in research, please cite:

```
ElectroTrace: Research-grade ECG annotation and leakage-safe ML toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

## Support

For issues, feature requests, or questions:
- Open a GitHub Issue
- Check existing documentation and test cases for examples
- Review API response payloads (detailed error messages provided)

---

**ElectroTrace v1.2** · research annotation, electrophysiology phenotyping, and leakage-safe ML dataset platform.
