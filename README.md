# ElectroTrace

Research-grade ECG and electrophysiology annotation, segmentation, phenotyping, external validation, and leakage-safe machine-learning toolkit. ElectroTrace provides an interactive web interface backed by a Python REST API for building curated research datasets with machine-assisted labeling and subject-stratified validation.

**Version:** 1.8.1  
**License:** AGPL-3.0  
**Status:** Active research software · validation-focused release

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
- **Signal-level adaptive polarity selector:** chooses positive versus negative polarity for a recording rather than merging both, with a conservative candidate-count rule and a guarded QRS-band fallback.
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
- Deterministic dataset/study manifests with SHA-256 provenance hashes.
- Rigorous validation reports with pooled metrics, macro record-level metrics, and record-bootstrap uncertainty intervals.
- Phenotype integrity checks and experimental-unit aggregation helpers.

## External Scientific Validation

Live scoreboard: [`validation_reports/VALIDATION_STATUS.md`](validation_reports/VALIDATION_STATUS.md).

ElectroTrace 1.8.1 has a **locked, held-out MIT-BIH primary validation protocol** with record-level splitting, separate calibration records, explicit provenance, and reproducible reports. The current primary protocol uses seed=42, 12 held-out records, a 75 ms matching tolerance, adaptive polarity with the guarded v2 fallback, a 200-tree Stage-2 Random Forest, and an F1-max threshold subject to a minimum calibration recall of 0.97.

### Locked MIT-BIH primary result

| Detector | Sensitivity | PPV | F1 |
|----------|-------------|-----|-----|
| **ElectroTrace two-stage** | **0.9924** | **0.9879** | **0.9902** |
| Pan-Tompkins (research reimplementation) | 0.9908 | 0.9954 | **0.9931** |
| Hamilton (research reimplementation) | 0.9990 | 0.9303 | 0.9634 |
| ElectroTrace Stage-1 adaptive | 0.9931 | 0.7553 | 0.8580 |

The two-stage detector is therefore **competitive with the research Pan-Tompkins reimplementation on this held-out split**, but does not outperform it on F1. Classical detectors in this table are research reimplementations, not certified reference binaries.

### Independent INCART external pilot

A 7-record independent INCART pilot was added on 2026-08-29. It is **not a completed 75-record external study** and should not be presented as population-level validation. The MIT-BIH-trained Stage-2 suppressor showed substantial cross-database domain shift:

| Detector | Sensitivity | PPV | F1 |
|----------|-------------|-----|-----|
| Hamilton | 0.975 | 0.912 | **0.943** |
| Pan-Tompkins | 0.817 | 0.983 | 0.892 |
| ElectroTrace Stage-1 | 0.638 | 0.883 | 0.741 |
| ElectroTrace two-stage (MIT-BIH model) | 0.333 | **0.990** | 0.499 |

This result is a **failure-mode/domain-shift finding**, not evidence of successful external generalization. Full INCART validation remains pending.

### Validation harness

ElectroTrace includes a reproducible PhysioNet/WFDB validation harness reporting sensitivity, PPV, F1, false positives/negatives, timing error, per-record performance, pooled estimates, macro record-level estimates, and record-level bootstrap confidence intervals.

No external dataset is bundled with the repository.

For long QTDB runs that may be interrupted, prefer the checkpointed harness:

```bash
python scripts/validate_qtdb_resumable.py .cache/physionet/qtdb \
  --output validation_reports/qtdb_validation.json \
  --checkpoint validation_reports/qtdb_checkpoint.json \
  --tolerance-ms 75
```

Stage-1 default polarity is `adaptive` (`detect_r_peaks(..., polarity="adaptive")`). The 1.8.1 selector includes a guarded QRS-band fallback when polarity confidence is below 0.15. MIT-BIH **207** remains an explicit documented edge case and is covered by the locked validation artifacts.

For the rigorous MIT-BIH protocol:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75 \
  --polarity adaptive \
  --bootstrap 2000 \
  --seed 42
```

The report preserves the complete record list, dataset/version metadata, detector configuration, software version/commit when supplied by the runtime, exact matching tolerance, per-record results, pooled metrics, macro record-level statistics, record-bootstrap uncertainty, and failures. The associated manifest is canonicalized and SHA-256 hashed.

**Deployment recommendation:** use the two-stage pipeline (`detect_r_peaks_two_stage` with a trained `CandidateSuppressor`) when false positives matter and the deployment domain has been appropriately validated. Single-stage adaptive detection remains the candidate generator.

For the locked two-stage verifier benchmark:

```bash
python scripts/benchmark_two_stage_mitdb.py \
  --data-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_two_stage_validation.json
```

The two-stage benchmark supports explicit polarity and optional recovery settings, and its current protocol keeps calibration records separate from model-fitting and held-out test records.

See `docs/VALIDATION.md`, `docs/TWO_STAGE_RPEAK.md`, `docs/RESEARCH_VALIDATION.md`, `docs/PEER_REVIEW_RESPONSE.md`, `docs/REMAINING_VALIDATION_STEPS.md`, and `docs/benchmarks/MITBIH_TWO_STAGE_2026-08-17.md`.

## Validation Status & Remaining Work

**Locked:**
- MIT-BIH held-out primary endpoint.
- Provenance schema v7 and SHA-256 dataset/software provenance.
- Protocol-matched Pan-Tompkins and Hamilton comparison.
- 1.8.1 adaptive/hybrid polarity behavior and regression coverage.
- INCART 7-record external pilot and documented domain-shift result.
- 103-test local suite, including baseline detector tests.

**Still required for stronger algorithm-paper claims:**
1. Full 75-record INCART external evaluation with per-record and macro-record analysis.
2. Certified WFDB `gqrs` / `sqrs` baseline comparison.
3. Full prespecified QTDB QRS delineation tolerance-curve analysis with uncertainty intervals.
4. Optional second external database (e.g., SVDB) using the same locked scoring framework.
5. Frozen tagged release with regenerated validation artifacts.

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

The current repository validation status records **103 passing tests**, including baseline detector coverage. CI runs on Python 3.10, 3.11, and 3.12. A separate security workflow runs `pip-audit` and Bandit.

## Scientific Use Notes

**Not a clinical device:** ElectroTrace is research software. Automatic R-peak detection, the false-positive suppressor, and phenotype extraction are research algorithms, not validated clinical algorithms.

**Reproducibility:** Raw signals are not overwritten by display preprocessing. Source format, absolute time bounds, provenance, model metadata, train/test record lists, calibration records, and threshold settings are preserved where applicable.

**External validation:** Report the exact dataset version, record list, detector configuration, polarity mode, matching tolerance, recovery setting, calibration records, and model threshold. Do not treat MIT-BIH/QTDB results as proof of population generalization, and do not treat the INCART pilot as a completed external validation study.

**Evaluation mode:** The locked two-stage benchmark is retrospective full-record evaluation, not streaming. Whole-record signal statistics are used by the current research protocol; no online/real-time performance claim is made.

**Large-data API:** JSON signal requests are capped at 64 MB; use persistent recording registration and window access for long recordings.

**Deployment:** The development server defaults to localhost. Non-local binds require `ELECTROTRACE_API_KEY`; use a TLS-capable reverse proxy for shared deployments. Trusted-model loading uses Python pickle and must never consume untrusted model files.

**Research statistics:** Prefer subject/record-level aggregation before inferential statistics. Do not treat individual ECG beats as independent experimental units unless the study design explicitly justifies that assumption.

**Phenotype QC:** `electrotrace.phenotype_validation.quality_report` checks structural and mathematical consistency without imposing clinical cutoffs; use study-specific clinical thresholds separately.

## Citation

```text
ElectroTrace: ECG annotation, electrophysiology phenotyping, external validation,
and leakage-safe machine-learning toolkit.
Virelion-Biotech, 2024–2026.
https://github.com/Virelion-Biotech/Virelion-ElectroTrace
```

---

**ElectroTrace v1.8.1** · annotation, adaptive/hybrid-polarity R-peak detection, two-stage verification, external validation, rigorous provenance, and leakage-safe ML.
