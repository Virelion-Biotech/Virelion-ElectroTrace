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

| Field | Value |
|-------|-------|
| Status | `complete` |
| Records | 48/48 |
| Tolerance | 75.0 ms |
| Polarity | adaptive |
| Bootstrap | 2000 |
| Manifest SHA-256 | `234da9f6384775e6319bdcd00fe06df8750db5f7472a0a69d7cff59827f9d798` |

### Pooled metrics

| Metric | Value |
|--------|-------|
| Sensitivity | 0.9656 |
| PPV | 0.6719 |
| F1 | 0.7924 |

### Macro record-level (bootstrap 95% CI)

| Metric | Mean | 95% CI |
|--------|------|--------|
| Sensitivity | 0.9640 | [0.9239, 0.9901] |
| PPV | 0.6895 | [0.6358, 0.7431] |
| F1 | 0.7914 | [0.7428, 0.8339] |

## 2. Two-stage MIT-BIH (Stage-2 FP suppressor)

Held-out test split (seed=42, polarity=adaptive, recovery=off):

| Metric | Value |
|--------|-------|
| Test records | 12 |
| Sensitivity | 0.9377 |
| PPV | 0.9932 |
| F1 | 0.9646 |

## 3. QT Database QRS-boundary confirmation (single-stage)

| Field | Value |
|-------|-------|
| Status | `complete` |
| Records | 105/105 |
| Matched QRS | 3351 / 3623 (92.5%) |
| Onset median abs | 8.0 ms |
| Offset median abs | 12.0 ms |

## 4. Remaining gates (2026-08-29 follow-up)

### MIT-BIH record 207 polarity lock

| Mode | Sensitivity | PPV | F1 |
|------|-------------|-----|-----|
| single adaptive / positive | 0.147 | 0.111 | 0.127 |
| **single negative** | **0.922** | **0.718** | **0.807** |
| two-stage adaptive | 0.133 | 0.751 | 0.226 |
| **two-stage negative** | **0.907** | **0.951** | **0.929** |

**Policy:** for record 207 (and similar inverted leads), set `polarity="negative"` explicitly. Do not change the global adaptive rule.

### Two-stage full pool (adaptive, seed=42, train 36 / held-out 12)

| Scope | Sens | PPV | F1 | Notes |
|-------|------|-----|-----|-------|
| Held-out 12 | 0.938 | 0.993 | 0.965 | Honest test estimate |
| All 48 (incl. train) | 0.964 | 0.997 | 0.981 | Optimistic on train records |

**Artifact:** `validation_reports/mitdb_remaining_validation_2026-08-29.json`

### Recovery A/B (held-out, adaptive)

| Setting | Sens | PPV | F1 |
|---------|------|-----|-----|
| recovery off | 0.9377 | 0.9932 | 0.9646 |
| recovery on | 0.9368 | 0.9953 | 0.9651 |

ΔF1 ≈ +0.0005. **Keep recovery off by default**; treat as exploratory.

### QTDB two-stage

See `validation_reports/qtdb_two_stage_validation_2026-08-29.json` when complete (MIT-BIH-trained Stage-2 model applied to QTDB centers before delineation).

## Notes

- PhysioNet data were downloaded locally via `wfdb` and are **not** committed.
- Stage-2 `.pkl` models are **not** committed (regenerate with training scripts).
- Intermediate checkpoint files are not committed.
