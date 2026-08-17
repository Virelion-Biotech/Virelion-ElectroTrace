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

The default local server is `http://127.0.0.1:5000`.

For any non-loopback bind, set a strong API key first:

```bash
export ELECTROTRACE_API_KEY='replace-with-a-strong-secret'
export ELECTROTRACE_HOST='0.0.0.0'
python server.py
```

API clients then send `Authorization: Bearer <key>` on `/api/*` requests. The default uploaded-recording retention is 24 hours and can be changed with `ELECTROTRACE_UPLOAD_TTL_S`.

## Core Capabilities

### Recording Import & Validation
- CSV, EDF/EDF+, and WFDB ZIP import.
- Strict finite-sample validation and sampling-rate checks.
- Non-destructive preprocessing.

### Large-recording architecture
- Metadata-only registration.
- Native lazy EDF/WFDB windows and bounded CSV windows.
- CSV browser loading uses the same persisted/windowed path as native recordings.
- Bounded browser state for long recordings.
- Automatic cleanup of stale uploaded recordings.

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
- Accepted annotations are matched to detected beats one-to-one.

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

The benchmark runner supports explicit polarity and optional recovery settings. Stage-2 threshold calibration is performed on whole training records kept separate from the records used to fit the final verifier and from the held-out test records.

See `docs/VALIDATION.md` and `docs/TWO_STAGE_RPEAK.md` for the reproducible protocol.

## Project & Recording Management

- Persistent project metadata and recording inventory.
- Subject/group/visit tracking.
- Cross-process locking for project metadata updates.
- Chunked access to large recordings.
- Safe archive extraction.

## Security

- Non-loopback deployment fails closed unless `ELECTROTRACE_API_KEY` is configured.
- JSON signal requests are capped at 64 MB; long recordings should use persistent windowed access.
- Uploaded recordings are automatically removed after the configured TTL.
- Pickle-based model loading is supported only for trusted local artifacts; do not load untrusted model files.
- CI runs `pip-audit` and Bandit in addition to the test matrix.

## Testing

```bash
pytest -q
```

CI runs on Python 3.10, 3.11, and 3.12, plus dependency/security auditing on Python 3.12.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection and the false-positive suppressor are research algorithms, not validated clinical algorithms.

**Reproducibility:** Raw signals are not overwritten by display preprocessing. Source format, absolute time bounds, provenance, model metadata, train/test record lists, calibration records, and threshold settings are preserved where applicable.

**External validation:** Report the exact dataset version, record list, detector configuration, polarity mode, matching tolerance, recovery setting, calibration records, and model threshold.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.6.0** · annotation, adaptive-polarity R-peak detection, two-stage verification, external validation, secure deployment, and leakage-safe ML.
