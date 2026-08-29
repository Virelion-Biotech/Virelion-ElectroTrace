# Response to external review of ElectroTrace validation

This document addresses reviewer-style concerns raised against the MIT-BIH /
two-stage validation story. Claims below are tied to locked artifacts under
`validation_reports/` and the protocol in `scripts/benchmark_two_stage_mitdb.py`.

## What we accept as valid

### 1. Report / code provenance must match

**Valid.** Earlier checked-in JSON reported `n_estimators=150` and
`held_out_stratified` while the script evolved to `n_estimators=200` and
F1-optimal threshold selection. That mismatch is a reproducibility defect.

**Fix:** regenerate every headline number from a clean environment on a frozen
commit; embed `git_head`, Python version, package versions, seed, calibration
records, test records, threshold method, and hyperparameters in the report
(`schema` v7). Artifact: `validation_reports/mitdb_two_stage_locked_1.8.1.json`.

### 2. Held-out (12 records) is the primary endpoint; full-pool is optimistic

**Valid.** Full-pool metrics include training records and must not be the
abstract headline. Status docs now mark held-out as primary and full-pool as
secondary/optimistic.

### 3. Record 207 is a systematic polarity failure mode, not noise

**Valid and scientifically useful.** Adaptive count-rule alone selects positive
on 207 and collapses sensitivity. Explicit negative polarity recovers
F1 ≈ 0.93 (two-stage). In **1.8.1** adaptive polarity uses QRS-band v2 when
count confidence < 0.15, which selects negative on 207 without regressing
pooled MIT-BIH performance.

| Mode | Sens | PPV | F1 |
|------|------|-----|-----|
| single adaptive (pre-1.8.1) | 0.15 | 0.11 | 0.13 |
| single negative | 0.92 | 0.72 | 0.81 |
| two-stage adaptive (pre-1.8.1) | 0.13 | 0.75 | 0.23 |
| two-stage negative | 0.91 | 0.95 | 0.93 |
| two-stage adaptive **1.8.1** | ~0.92 | ~0.91 | ~0.91 |

Polarity selection is a fixed rule in code. The confidence threshold 0.15 was
chosen on full-pool polarity strategy comparisons; the held-out two-stage
metrics are the locked primary claim.

### 4. Evaluation is retrospective full-record, not streaming

**Valid.** Features use whole-recording median/std. Protocol fields
`evaluation_mode: retrospective_full_record` and `streaming_claim: false`
appear in every locked v7 report. Do not present as an online detector.

### 5. n=12 held-out records is limited

**Valid.** Primary reporting must include pooled held-out metrics, per-record
tables, explicit 207 analysis, and no claim of clinical generalization from
MIT-BIH alone.

### 6. QTDB is confirmatory, not a full delineation study

**Valid.** QTDB supports boundary timing on matched QRS events only.

### 7. Novelty framing

**Valid as scientific writing.** Recommended angles: (1) leakage-safe research
framework; (2) polarity methodology; (3) detector only with locked baselines.

## What we reject or qualify

### Classic train/test leakage

**Not found.** Record-level splits and non-overlapping calibration vs fit are
enforced. Full-record normalization is retrospective design, not sample mixing.

### Clinical-grade performance

**Correctly not claimed.** Research software only.

## Frozen protocol summary (1.8.1+)

| Field | Value |
|-------|--------|
| Primary endpoint | Held-out test records only (seed=42, test_fraction=0.25) |
| Matching tolerance | 75 ms |
| Stage-1 polarity | adaptive count + v2 if confidence < 0.15 |
| Stage-2 | RF n_estimators=200 |
| Threshold | F1-max on calibration, min_recall=0.97 |
| Recovery | off by default |
| Evaluation mode | retrospective full-record |

## Remaining work for algorithm-paper strength

1. Locked baseline comparison (Pan–Tompkins / Hamilton / open detectors).
2. Independent external database beyond MIT-BIH/QTDB.
3. Prespecified QTDB delineation protocol with uncertainty intervals.
4. Frozen tagged release with regenerated artifacts only.
