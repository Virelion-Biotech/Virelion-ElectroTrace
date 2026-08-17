# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, external validation, and leakage-safe machine-learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated research datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.5.0  
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

- **R-peak detection:** high-recall candidate generation using prominence and distance thresholding; not a validated clinical algorithm.
- **Two-stage R-peak pipeline:** Stage 1 generates high-recall candidates; a trained Random Forest verifier suppresses likely false positives using morphology, slope, energy, and RR-context features.
- **Configurable beat windows:** default 0.35 s pre-R, 0.55 s post-R; adjustable per session.
- **Beat-level features:** RR intervals, heart rate, R-wave amplitude, QRS-width proxy, baseline estimation.
- **Phenotype summaries:** beat counts, heart-rate statistics, and group-level comparisons.

### Machine-Assisted Labeling

- **Random Forest model training:** learns only from human-accepted point annotations.
- **Training safeguards:** requires at least 4 matched accepted examples from at least 2 labels.
- **Reproducibility metadata:** model version, feature-schema version, seed, estimator count, class weighting, and training-example count are returned with training metrics.
- **Predictive uncertainty ranking:** surfaces uncertain beats for human review in active-learning fashion.
- **Active-learning hygiene:** already annotated/training beats are excluded and temporally redundant adjacent candidates can be suppressed.
- **Two-stage suppressor safeguards:** threshold selection is performed only on training records; held-out records are used only for final evaluation.
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

ElectroTrace includes a reproducible PhysioNet/WFDB validation harness. It compares detector sample indices against reference beat annotations and reports sensitivity, positive predictive value, F1, false positives/negatives, timing error, and per-record results.

No external dataset is bundled with the repository.

For MIT-BIH, use the included validation scripts and keep the dataset outside Git:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75
```

For the two-stage verifier:

```bash
python scripts/benchmark_two_stage_mitdb.py \
  --data-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_two_stage_validation.json
```

The two-stage benchmark performs record-level train/test splitting and chooses its operating threshold from training records only. See `docs/VALIDATION.md` and `docs/TWO_STAGE_RPEAK.md` for the reproducible protocol.

### Project & Recording Management

- **Project storage:** persistent metadata in JSON (project name, timestamps, recording inventory).
- **Subject/group/visit tracking:** organize recordings by study structure and ML grouping.
- **Windowed access:** large recordings are accessed in sample-based chunks.
- **Safe archive extraction:** validates ZIP structure; prevents path traversal and symlink attacks.

## How It's Organized

```text
.
├── server.py                 Flask REST API server + static file serving
├── web/                      Browser frontend
├── src/electrotrace/
│   ├── io.py                 CSV loading and validation
│   ├── formats.py            EDF/WFDB import
│   ├── metadata.py           Metadata-only recording discovery
│   ├── window.py             Native lazy sample-window access
│   ├── signal.py             Signal filtering
│   ├── beats.py              Beat segmentation
│   ├── annotations.py        Annotation model and review states
│   ├── ml.py                 Random Forest active learning
│   ├── candidate_suppressor.py Two-stage false-positive suppression
│   ├── phenotype.py          Beat-level phenotypes
│   ├── statistics.py         Group statistics and FDR
│   ├── benchmark.py          Leakage-safe ML benchmarking
│   ├── validation.py         External PhysioNet validation
│   └── project_store.py      Persistent project storage
├── scripts/                  Reproducible external-validation runners
├── docs/                     Validation and two-stage methodology
├── tests/                    pytest suite
├── sample_data/              Example CSV recordings
├── pyproject.toml            Build configuration
├── requirements.txt          Dependency list
└── LICENSE                   MIT
```

## API Endpoints

### Recording
- `POST /api/analyze`
- `POST /api/recording`
- `GET /api/recording/<id>/window`

### Signal / Beats
- `POST /api/filter`
- `POST /api/detect/r-peaks`
- `POST /api/beats`
- `POST /api/segment`

### ML / Analysis
- `POST /api/ml/train`
- `POST /api/ml/suggest`
- `POST /api/phenotype`
- `POST /api/statistics/compare`
- `POST /api/statistics/fdr`
- `POST /api/benchmark`

### Project Management
- `GET /api/project?name=<project>`
- `POST /api/project/recording`

## Testing

```bash
pytest -q
```

CI runs on Python 3.10, 3.11, and 3.12 with test dependencies installed from `.[test]`.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection and the second-stage suppressor are candidate-generation/research algorithms, not validated clinical algorithms. External benchmark performance does not establish clinical safety or utility.

**Reproducibility:** Raw signals are not overwritten by display preprocessing. Source format, absolute time bounds, preprocessing, annotation/review provenance, model metadata, train/test record lists, and threshold settings are preserved where applicable.

**Leakage prevention:** ML benchmarking is subject/record aware, and the two-stage MIT-BIH pipeline chooses its threshold on training records only before evaluating held-out records.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.5.0** · annotation, two-stage R-peak verification, external validation, and leakage-safe ML.
