# External Validation Status

**Generated:** 2026-08-29 15:57 UTC  
**Software:** electrotrace 1.8.0  
**Protocol date:** 2026-08-29  

This document records live PhysioNet validation runs executed against the public MIT-BIH Arrhythmia and QT Databases. No PhysioNet signal files are redistributed.

## Unit tests

| Suite | Result |
|-------|--------|
| `pytest` | **101 passed** |

## 1. Official MIT-BIH R-peak protocol

Command (README):

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation_official_2026-08-29.json \
  --tolerance-ms 75 --polarity adaptive --bootstrap 2000 --seed 42
```

| Field | Value |
|-------|-------|
| Status | `complete` |
| Records | 48/48 |
| Tolerance | 75.0 ms |
| Polarity | adaptive |
| Bootstrap | 2000 |
| Manifest SHA-256 | `234da9f6384775e6319bdcd00fe06df8750db5f7472a0a69d7cff59827f9d798` |
| Git HEAD (at run) | `9578678ecea3fe8bbb12ea4296ace3d75f746354` |

### Pooled metrics

| Metric | Value |
|--------|-------|
| Sensitivity | 0.9656 |
| PPV | 0.6719 |
| F1 | 0.7924 |
| TP / FP / FN | 105727 / 51621 / 3767 |

### Macro record-level (bootstrap 95% CI)

| Metric | Mean | 95% CI |
|--------|------|--------|
| Sensitivity | 0.9640 | [0.9239, 0.9901] |
| PPV | 0.6895 | [0.6358, 0.7431] |
| F1 | 0.7914 | [0.7428, 0.8339] |

**Artifact:** `validation_reports/mitdb_rpeak_validation_official_2026-08-29.json`

## 2. Two-stage MIT-BIH (Stage-2 FP suppressor)

Held-out test split (seed=42, polarity=adaptive, recovery=off):

| Metric | Value |
|--------|-------|
| Test records | 12 |
| Sensitivity | 0.9377 |
| PPV | 0.9932 |
| F1 | 0.9646 |
| TP / FP / FN | 25252 / 173 / 1678 |

Stage-2 recovers PPV (0.67 → 0.99) relative to single-stage pooled MIT-BIH.

**Artifact:** `validation_reports/mitdb_two_stage_validation_2026-08-29.json`

## 3. QT Database QRS-boundary confirmation

| Field | Value |
|-------|-------|
| Status | `complete` |
| Records | 105/105 |
| Detector | adaptive R-peak + delineator |
| Event tolerance | 75.0 ms |
| Failures | 0 |

| Aggregate | Value |
|-----------|-------|
| Reference QRS | 3623 |
| Matched QRS | 3351 (92.5% of reference) |
| Detected centers | 167264 |

| Boundary | Median abs (ms) | Mean abs (ms) | p95 abs (ms) |
|----------|-----------------|---------------|--------------|
| Onset | 8.0 | 14.8 | 46.0 |
| Offset | 12.0 | 19.0 | 64.0 |

Single-stage over-detection is expected; prefer the two-stage pipeline for deployment.

**Artifact:** `validation_reports/qtdb_validation_2026-08-29.json`

## Notes

- PhysioNet data were downloaded locally via `wfdb` and are **not** committed.
- The two-stage `.pkl` model artifact is **not** committed (regenerate with `scripts/benchmark_two_stage_mitdb.py`).
- Intermediate QTDB checkpoint files are not committed.

## Hard case: MIT-BIH 207

Record 207 is an inverted-lead edge case under the validated adaptive count rule:

| Setting | Sensitivity | PPV |
|---------|-------------|-----|
| adaptive / positive (default rule) | ~0.15 | ~0.11 |
| **negative (explicit)** | **~0.92** | **~0.72** |
| two-stage + negative | ~0.91 | ~0.94 |

**Guidance:** pass `polarity="negative"` for this recording. Global morphology-score polarity overrides were evaluated on all 48 MIT-BIH records and rejected due to pooled regression.
