# External Validation Status

**Software:** electrotrace 1.8.1  
**Protocol date:** 2026-08-29

## Unit tests

| Suite | Result |
|-------|--------|
| `pytest` | **101 passed** |

## Detector improvements (1.8.1)

1. **Hybrid polarity:** count-rule confidence < 0.15 → QRS-band polarity v2 (fixes 207).
2. **Stage-2 threshold:** F1-optimal on calibration with min_recall=0.97.
3. **RF:** n_estimators 200.

### Single-stage adaptive (48 records)

| Metric | 1.8.0 | **1.8.1** |
|--------|-------|-----------|
| Sensitivity | 0.9656 | **0.9788** |
| PPV | 0.6719 | **0.6814** |
| F1 | 0.7924 | **0.8034** |
| 207 sens | 0.15 | **0.922** |

### Two-stage held-out (seed=42)

| Metric | 1.8.0 | **1.8.1** |
|--------|-------|-----------|
| Sensitivity | 0.9377 | **0.9924** |
| PPV | 0.9932 | **0.9879** |
| F1 | 0.9646 | **0.9902** |
| FN | 1678 | **204** |

### Two-stage full pool 48

| Metric | Value |
|--------|-------|
| Sensitivity | **0.9784** |
| PPV | **0.9951** |
| F1 | **0.9867** |
| 207 | sens 0.917 / PPV 0.907 |

**Artifact:** `validation_reports/mitdb_two_stage_improved_2026-08-29.json`

## Notes

- PhysioNet signals and `.pkl` models are not redistributed.
- Recovery remains **off by default**.
