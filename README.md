# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, external validation, and leakage-safe machine-learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated research datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.4.0  
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
pip install -e ".[test]"
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
- **EDF/EDF+ support:** multi-channel recordings with automatic sampling-rate detection.
- **WFDB ZIP archives:** compressed records with `.hea` header + signal files.
- **Validation:** monotonic timestamps, strict rejection of NaN/infinite samples, sampling-rate inference, and irregular-sampling warnings.
- **Non-destructive preprocessing:** raw signal arrays preserved; filters applied only for display/analysis copies.

### Large-recording architecture

- **Metadata-only registration:** upload/registration uses EDF/WFDB headers and bounded-memory CSV scanning instead of materializing the complete recording.
- **Native lazy windows:** EDF uses direct sample-range reads; WFDB uses `sampfrom`/`sampto`; CSV uses bounded row reads.
- **Bounded browser state:** the browser loads only the requested signal window for large recordings.

### Interactive Annotation

- **Multi-channel Plotly visualization:** zoom, pan, drag-to-select intervals, click-to-capture points.
- **Point and interval annotations:** flexible labeling with confidence scores and notes.
- **Annotator identity & review states:** unreviewed, accepted, flagged workflow.
- **Real-time QC dashboard:** annotation counts, review status, and multi-annotator agreement metrics.

### Beat Segmentation & Phenotyping

- **R-peak detection:** automatic candidate generation (not validated clinical-grade) using prominence and distance thresholding.
- **Configurable beat windows:** default 0.35 s pre-R, 0.55 s post-R; adjustable per session.
- **Beat-level features:** RR intervals, heart rate, R-wave amplitude, QRS-width proxy, baseline estimation.
- **Phenotype summaries:** beat counts, heart-rate statistics, and group-level comparisons.

### Machine-Assisted Labeling

- **Random Forest model training:** learns only from human-accepted point annotations.
- **Training safeguards:** requires at least 4 matched accepted examples from at least 2 labels.
- **Reproducibility metadata:** model version, feature-schema version, seed, estimator count, class weighting, and training-example count are returned with training metrics.
- **Predictive uncertainty ranking:** surfaces uncertain beats for human review in active-learning fashion.
- **Active-learning hygiene:** already annotated/training beats are excluded and temporally redundant adjacent candidates can be suppressed.
- **Explicit review workflow:** model suggestions remain unreviewed until human confirmation.

### Research-Grade Analysis & Exports

- **Leakage-safe benchmarking:** `StratifiedGroupKFold` keeps subjects out of both training and validation folds simultaneously.
- **Multi-model comparison:** Logistic Regression and Random Forest with accuracy, balanced accuracy, macro/weighted F1, and ROC-AUC where defined.
- **Uncertainty reporting:** benchmark output includes fold-level metrics plus mean, SD, and empirical 95% CI.
- **Experimental-unit-aware statistics:** repeated beat observations can be aggregated by subject/animal before inference; observation-level analyses are explicitly flagged for pseudoreplication risk.
- **Phenotype statistics:** Welch t-test, Mann-Whitney U, Cohen's d for group comparisons.
- **FDR adjustment:** Benjamini-Hochberg correction for multiple hypothesis testing.
- **Versioned JSON export:** `electrotrace.annotation/v2` schema with provenance, source format, absolute time bounds, preprocessing, annotator, and review status.
- **CSV export:** flat table of annotations with metadata for downstream analysis.

### External Scientific Validation

ElectroTrace includes a reproducible PhysioNet/WFDB validation harness. It compares detector sample indices against reference beat annotations and reports:

- sensitivity/recall
- positive predictive value
- F1 score
- true/false positives and negatives
- median absolute timing error
- 95th-percentile timing error
- per-record and aggregate results

No external dataset is bundled with the repository.

Example against locally downloaded WFDB records:

```bash
python -m electrotrace.validation record_100 record_101 \
  --detector my_detector_module:detect \
  --annotation-extension atr \
  --tolerance-ms 75
```

The detector callable must accept `(signal, fs_hz)` and return sample indices. For MIT-BIH-style beat annotations, the harness defaults to standard beat symbols and can be overridden with `--symbols N,V,...`.

**Recommended validation sequence:**
1. Validate on a development subset.
2. Freeze detector parameters.
3. Run the complete held-out record set.
4. Report per-record and aggregate performance with the tolerance explicitly stated.
5. Report external validation separately from any model-training benchmark.

MIT-BIH Arrhythmia is a strong first external benchmark because its complete 48-record database includes computer-readable beat reference annotations generated from independent cardiologist annotation with adjudication. citeturn345972search0turn345972search1 For long-recording stress tests, PhysioNet also provides long-term ECG resources; AFDB contains 10-hour two-channel recordings with rhythm annotations, although its `.qrs` annotations are automated and manually corrected `.qrsc` files should be preferred when available. citeturn345972search6

For waveform-boundary validation, the QT Database provides manual waveform onset/offset annotations for P, QRS, and T waves in addition to source beat annotations. citeturn345972search5

### Project & Recording Management

- **Project storage:** persistent metadata in JSON (project name, timestamps, recording inventory).
- **Subject/group/visit tracking:** organize recordings by study structure and ML grouping.
- **Windowed access:** large recordings are accessed in sample-based chunks.
- **Safe archive extraction:** validates ZIP structure; prevents path traversal and symlink attacks.

## How It's Organized

```text
.
├── server.py                 Flask REST API server + static file serving
├── web/                      Browser frontend (HTML, CSS, JavaScript)
│   ├── index.html            Main UI shell
│   ├── app.js                Core application state, file I/O, plotting
│   ├── research.js           Project storage, phenotyping, benchmarking
│   ├── shortcuts.js          Keyboard bindings
│   └── styles.css            Layout and theming
├── src/electrotrace/         Python analysis backend
│   ├── __init__.py           Package metadata
│   ├── io.py                 CSV loading, validation, channel labeling
│   ├── formats.py            EDF and WFDB ZIP import + safe extraction
│   ├── metadata.py           Metadata-only recording discovery
│   ├── window.py             Native lazy sample-window access
│   ├── signal.py             Butterworth filtering
│   ├── beats.py              Beat segmentation from R-peak indices
│   ├── annotations.py        Annotation model, validation, review states
│   ├── ml.py                 Random Forest training, features, active learning
│   ├── phenotype.py          Beat-level cardiac phenotypes
│   ├── statistics.py         Group comparisons, FDR adjustment
│   ├── benchmark.py          Leakage-safe cross-validation
│   ├── validation.py         External PhysioNet/WFDB validation harness
│   ├── project.py            Recording metadata helpers
│   └── project_store.py      Persistent project JSON + chunked recording access
├── tests/                    pytest suite
├── sample_data/              Example CSV recordings for demo
├── pyproject.toml            Build config, dependencies, test settings
├── requirements.txt          Dependency list
└── LICENSE                   MIT
```

## Data Flow

1. **Upload & register:** CSV/EDF/WFDB → metadata validation without full signal materialization.
2. **Window & display:** browser requests only the needed sample range.
3. **Segment & feature:** R-peak candidate generation → beat windows + RR/HR metrics.
4. **Label & review:** human point/interval annotation → acceptance/QC.
5. **Train & suggest:** accepted labels → model training → uncertainty/diversity-based suggestions.
6. **Export & analyze:** provenance-preserving export → phenotype/statistics/ML benchmark.
7. **Externally validate:** run the detector/analysis against independent annotated datasets.

## API Endpoints

### Recording Upload & Analysis
- `POST /api/analyze` – validate and inspect uploaded recordings
- `POST /api/recording` – upload/register CSV/EDF/WFDB metadata without eager signal materialization
- `GET /api/recording/<id>/window` – fetch only the requested sample window

### Signal Processing
- `POST /api/filter` – baseline/high-pass/low-pass/notch filtering
- `POST /api/detect/r-peaks` – detect R-peak candidates

### Beat Operations
- `POST /api/beats` or `POST /api/segment` – segment beats from peaks

### ML & Analysis
- `POST /api/ml/train` – train Random Forest from accepted point annotations
- `POST /api/ml/suggest` – train + return uncertainty/diversity-ranked suggestions
- `POST /api/phenotype` – extract beat phenotypes
- `POST /api/statistics/compare` – Welch t-test, Mann-Whitney U, Cohen's d with unit-aware aggregation support
- `POST /api/statistics/fdr` – Benjamini-Hochberg FDR adjustment
- `POST /api/benchmark` – subject-level `StratifiedGroupKFold` benchmark

### Project Management
- `GET /api/project?name=<project>` – load project metadata
- `POST /api/project/recording` – add recording with subject/group/visit metadata

## Testing

```bash
pytest -q
pytest --cov=src/electrotrace tests/
```

CI runs on Python 3.10, 3.11, and 3.12 with test dependencies installed from `.[test]`.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection is a candidate generator, not a validated clinical algorithm. External benchmark performance does not establish clinical safety or clinical utility.

**Reproducibility & transparency:** Raw signals are not overwritten by display preprocessing. Source formats, absolute time bounds, filter configuration, annotation/review provenance, and model metadata are preserved where applicable.

**Leakage prevention:** The benchmark module enforces subject-level stratification, and the statistics layer can aggregate repeated observations to the experimental unit before inference.

**External validation:** Validation datasets are kept outside the repository. Report the exact dataset version, record list, annotation source, detector configuration, and matching tolerance for reproducibility.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.4.0** · annotation, electrophysiology phenotyping, external validation, and leakage-safe ML.
