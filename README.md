# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, external validation, and leakage-safe machine-learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated research datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.5.3  
**License:** MIT  
**Status:** Active research software

## Quick Start

### Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[test]"
```

### Running the Server

```bash
python server.py
```

Open `http://127.0.0.1:5000` in your browser.

## Core Capabilities

### Recording Import & Validation
- CSV, EDF/EDF+, and WFDB ZIP import.
- Strict finite-sample validation and sampling-rate checks.
- Non-destructive preprocessing.

### Large-recording architecture
- Metadata-only registration.
- Native lazy EDF/WFDB windows and bounded CSV windows.
- CSV browser loading now uses the same persisted/windowed path as native recordings.
- Bounded browser state for long recordings.

### Interactive Annotation
- Multi-channel Plotly visualization.
- Point/interval annotations with confidence and notes.
- Annotator/reviewer identity and QC states.
- Multi-annotator agreement support.

### Beat Segmentation & R-peak Detection
- High-recall Stage 1 candidate generation using prominence and minimum distance.
- **Signal-level adaptive polarity selector:** chooses positive versus negative polarity for a recording rather than merging both, using a conservative candidate-count rule validated on the complete MIT-BIH database.
- **Two-stage R-peak pipeline:** Stage 1 candidates followed by a Random Forest false-positive suppressor using morphology, slope, energy, and RR-context features.
- **Optional long-gap recovery:** adds at most one relaxed candidate inside unusually long RR gaps and passes it through the same Stage-2 verifier; it remains off by default pending external validation.
- Configurable beat windows and beat-level phenotype extraction.

### Machine-Assisted Labeling
- Random Forest learning from human-accepted labels.
- Training safeguards and reproducibility metadata.
- Uncertainty/diversity-based active learning.
- Already annotated/training beats excluded from suggestions.
- Pipeline class handling is explicit for prediction/reporting.

### Research Analysis
- Subject/record-level leakage-safe ML benchmarking.
- Fold-level metrics plus mean, SD, and empirical 95% CI.
- Experimental-unit-aware statistics and pseudoreplication warnings.
- Welch t-test, Mann-Whitney U, Cohen's d, and Benjamini-Hochberg FDR.

### External Scientific Validation
ElectroTrace includes a reproducible PhysioNet/WFDB validation harness reporting sensitivity, PPV, F1, false positives/negatives, timing error, and per-record performance.

No external dataset is bundled with the repository.

For MIT-BIH:

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

The benchmark runner supports explicit polarity and optional recovery settings so experiments use the same candidate-generation protocol during training and held-out testing.

See `docs/VALIDATION.md` and `docs/TWO_STAGE_RPEAK.md` for the reproducible protocol.

## Project & Recording Management

- Persistent project metadata and recording inventory.
- Subject/group/visit tracking.
- Chunked access to large recordings.
- Safe archive extraction.

## How It's Organized

```text
.
├── server.py
├── web/
├── src/electrotrace/
│   ├── io.py
│   ├── formats.py
│   ├── metadata.py
│   ├── window.py
│   ├── signal.py
│   ├── beats.py
│   ├── annotations.py
│   ├── ml.py
│   ├── candidate_suppressor.py
│   ├── phenotype.py
│   ├── statistics.py
│   ├── benchmark.py
│   ├── validation.py
│   ├── validation_detectors.py
│   └── project_store.py
├── scripts/
├── docs/
├── tests/
├── sample_data/
├── pyproject.toml
└── LICENSE
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

## Testing

```bash
pytest -q
```

CI runs on Python 3.10, 3.11, and 3.12.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection and the false-positive suppressor are research algorithms, not validated clinical algorithms.

**Reproducibility:** Raw signals are not overwritten by display preprocessing. Source format, absolute time bounds, provenance, model metadata, train/test record lists, and threshold settings are preserved where applicable. Stage-2 suppressor thresholds are calibrated on held-out training candidates by default rather than the same examples used to fit the final model.

**External validation:** Report the exact dataset version, record list, detector configuration, polarity mode, matching tolerance, recovery setting, and model threshold.

**Large-data API:** JSON signal requests are capped at 64 MB; use persistent recording registration and window access for long recordings.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.5.3** · annotation, adaptive-polarity R-peak detection, two-stage verification, external validation, and leakage-safe ML.
