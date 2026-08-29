# External Validation Status

**Generated:** 2026-08-29  
**Software:** electrotrace 1.8.0  
**Protocol date:** 2026-08-29  

This document records live PhysioNet validation runs executed against the public MIT-BIH Arrhythmia and QT Databases. No PhysioNet signal files are redistributed.

## Unit tests

| Suite | Result |
|-------|--------|
| `pytest` | **101 passed** |

## 1. Official MIT-BIH R-peak protocol (single-stage adaptive)

| Metric | Value |
|--------|-------|
| Records | 48/48 |
| Sensitivity | 0.9656 |
| PPV | 0.6719 |
| F1 | 0.7924 |
| Bootstrap | 2000 |

## 2. Two-stage MIT-BIH held-out (seed=42, adaptive, recovery off)

| Metric | Value |
|--------|-------|
| Test records | 12 |
| Sensitivity | 0.9377 |
| PPV | **0.9932** |
| F1 | 0.9646 |

## 3. QTDB single-stage boundary

| Field | Value |
|-------|-------|
| Records | 105/105 |
| Matched | 3351 / 3623 (92.5%) |
| Onset / offset median abs | 8.0 / 12.0 ms |

## 4. Remaining gates (completed)

### MIT-BIH record 207 polarity lock

| Mode | Sens | PPV | F1 |
|------|------|-----|-----|
| single adaptive/positive | 0.147 | 0.111 | 0.127 |
| **single negative** | **0.922** | **0.718** | **0.807** |
| two-stage adaptive | 0.133 | 0.751 | 0.226 |
| **two-stage negative** | **0.907** | **0.951** | **0.929** |

**Policy:** set `polarity="negative"` for record 207. Do not change global adaptive rule.

### Two-stage full pool (adaptive, train 36 / held-out 12)

| Scope | Sens | PPV | F1 |
|-------|------|-----|-----|
| Held-out 12 | 0.938 | 0.993 | 0.965 |
| All 48 (incl. train) | 0.964 | 0.997 | 0.981 |

### Recovery A/B (held-out)

| Setting | Sens | PPV | F1 |
|---------|------|-----|-----|
| off | 0.9377 | 0.9932 | 0.9646 |
| on | 0.9368 | 0.9953 | 0.9651 |

**Keep recovery off by default** (ΔF1 ≈ +0.0005).

### QTDB two-stage (MIT-BIH-trained Stage-2 model)

| Aggregate | Value |
|-----------|-------|
| Records | 105/105 |
| Matched | 3319 / 3623 (91.6%) |
| Detected centers | 101710 |
| Onset / offset median abs | 8.0 / 12.0 ms |

Domain shift remains; study-specific calibration recommended before production claims on QTDB.

**Artifacts:** `mitdb_remaining_validation_2026-08-29.json`, `qtdb_two_stage_validation_2026-08-29.json`

## Notes

- PhysioNet signals and Stage-2 `.pkl` models are **not** redistributed.
