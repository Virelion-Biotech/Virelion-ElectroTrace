# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, external validation, and leakage-safe machine-learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated research datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.6.0  
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

For shared/non-local deployment, set a strong `ELECTROTRACE_API_KEY` before binding beyond localhost. See `SECURITY.md`.

## Core Capabilities

### Recording Import & Validation
- CSV, EDF/EDF+, and WFDB ZIP import.
- Strict finite-sample validation and sampling-rate checks.
- Non-destructive preprocessing.

### Large-recording architecture
- Metadata-only registration.
- Native lazy EDF/WFDB windows and bounded CSV windows.
- CSV browser loading uses the persisted/windowed path.
- Bounded browser state for long recordings.
- Automatic upload retention cleanup via `ELECTROTRACE_UPLOAD_TTL_S`.

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

The two-stage benchmark supports explicit polarity and optional recovery settings, and its current protocol supports record-level calibration records separate from model-fitting records and held-out test records.

See `docs/VALIDATION.md`, `docs/TWO_STAGE_RPEAK.md`, and `docs/benchmarks/MITBIH_TWO_STAGE_2026-08-17.md`.

## Project & Recording Management

- Persistent project metadata and recording inventory.
- Subject/group/visit tracking.
- Chunked access to large recordings.
- Safe archive extraction.
- Cross-process metadata locking.

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

## Testing & Security

```bash
pytest -q
```

CI runs on Python 3.10, 3.11, and 3.12. A separate security workflow runs `pip-audit` and Bandit.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection and the false-positive suppressor are research algorithms, not validated clinical algorithms.

**Reproducibility:** Raw signals are not overwritten by display preprocessing. Source format, absolute time bounds, provenance, model metadata, train/test record lists, calibration records, and threshold settings are preserved where applicable.

**External validation:** Report the exact dataset version, record list, detector configuration, polarity mode, matching tolerance, recovery setting, calibration records, and model threshold.

**Large-data API:** JSON signal requests are capped at 64 MB; use persistent recording registration and window access for long recordings.

**Deployment:** The development server defaults to localhost. Non-local binds require `ELECTROTRACE_API_KEY`; use a TLS-capable reverse proxy for shared deployments. Trusted-model loading uses Python pickle and must never consume untrusted model files.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.6.0** · annotation, adaptive-polarity R-peak detection, two-stage verification, external validation, and leakage-safe ML.
