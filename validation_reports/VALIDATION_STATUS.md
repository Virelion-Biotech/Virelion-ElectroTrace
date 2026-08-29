# External Validation Status

**Software:** electrotrace 1.8.1  
**Primary scientific endpoint:** held-out MIT-BIH test records only (not full-pool)

PhysioNet signals are **not** redistributed. Full-pool metrics include training
records and are **optimistic**—do not use them as abstract headlines.

## Unit tests

| Suite | Result |
|-------|--------|
| `pytest` | **101 passed** |

## Protocol freeze (two-stage)

| Field | Value |
|-------|--------|
| Split | seed=42, test_fraction=0.25 → 12 held-out / 36 train |
| Calibration | record-level groups disjoint from model-fit |
| Tolerance | 75 ms |
| Polarity | adaptive count; v2 if confidence < 0.15 |
| Stage-2 RF | n_estimators=200 |
| Threshold | F1-max on calibration, min_recall=0.97 |
| Recovery | **off** by default |
| Evaluation mode | retrospective full-record (not streaming) |

## 1. Primary: two-stage held-out (seed=42, adaptive)

| Metric | Value (locked 1.8.1) |
|--------|----------------------|
| Test records | 12 |
| Sensitivity | **0.9924** |
| PPV | **0.9879** |
| F1 | **0.9902** |
| TP / FP / FN | 26726 / 326 / 204 |
| Git HEAD | `c50b37a` |
| Schema | v7 (provenance embedded) |

Record **207** (inverted lead): two-stage adaptive 1.8.1 → sens ≈ 0.92 / PPV ≈ 0.91.
Always report per-record tables in manuscripts.

**Artifact:** `validation_reports/mitdb_two_stage_locked_1.8.1.json`

## 2. Single-stage adaptive (baseline only)

| Metric | 1.8.1 |
|--------|-------|
| Sensitivity | 0.9788 |
| PPV | 0.6814 |
| F1 | 0.8034 |

Deploy **two-stage** when PPV matters.

## 3. Secondary / optimistic: full-pool 48

Sens 0.978 / PPV 0.995 / F1 0.987 — **not a primary endpoint.**

## 4. QTDB (confirmatory only)

105/105 records; matched ~92%; onset/offset median abs ~8/12 ms. Not a full
delineation claim.

## Explicit non-claims

- Not a clinical device.
- Not a streaming detector.
- MIT-BIH + QTDB do not prove population generalization.
- Full-pool F1 is not the headline result.

See `docs/PEER_REVIEW_RESPONSE.md`.
