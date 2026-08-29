# External Validation Status

**Software:** electrotrace 1.8.1  
**Primary scientific endpoint:** held-out MIT-BIH test records only (not full-pool)

## Unit tests

| Suite | Result |
|-------|--------|
| `pytest` | **103 passed** (incl. baseline detectors) |

## Protocol freeze (two-stage)

| Field | Value |
|-------|--------|
| Split | seed=42, test_fraction=0.25 → 12 held-out / 36 train |
| Tolerance | 75 ms |
| Polarity | adaptive count; v2 if confidence < 0.15 |
| Stage-2 RF | n_estimators=200 |
| Threshold | F1-max on calibration, min_recall=0.97 |
| Evaluation mode | retrospective full-record (not streaming) |

## 1. Primary: two-stage held-out

| Metric | Value |
|--------|-------|
| Sensitivity | **0.9924** |
| PPV | **0.9879** |
| F1 | **0.9902** |

Artifact: `mitdb_two_stage_locked_1.8.1.json`

## 7. Locked baseline comparison (held-out 12, seed=42)

| Detector | Sens | PPV | F1 |
|----------|------|-----|-----|
| **Pan–Tompkins** (research reimpl.) | 0.9908 | **0.9954** | **0.9931** |
| **ElectroTrace two-stage** | 0.9924 | 0.9879 | 0.9902 |
| Hamilton (research reimpl.) | 0.9990 | 0.9303 | 0.9634 |
| ElectroTrace Stage-1 adaptive | 0.9931 | 0.7553 | 0.8580 |

Record 207: Pan–Tompkins F1≈0.96; two-stage≈0.91; Stage-1≈0.81.

Classical detectors are research reimplementations, not certified reference binaries. Two-stage is competitive with Pan–Tompkins on this split (ΔF1 ≈ 0.003).

**Artifact:** `validation_reports/mitdb_baseline_comparison_locked.json`

## Explicit non-claims

- Not clinical-grade; not streaming.
- Full-pool is optimistic.
- MIT-BIH + QTDB do not prove population generalization.

See `docs/PEER_REVIEW_RESPONSE.md`.
