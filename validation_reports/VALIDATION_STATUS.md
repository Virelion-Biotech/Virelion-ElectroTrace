# Validation status

**Software:** electrotrace 1.8.1  
**Primary scientific endpoint:** held-out MIT-BIH test records only (not full-pool)  
**Current hardening branch:** release-hardening-v1.9

## Unit tests

The locked historical validation artifact reports 103 passing tests at the time of the 1.8.1 scientific freeze. The hardening branch adds public-API, detector-registry, and model-persistence regression coverage; current CI is the authoritative result for the new branch.

## Locked two-stage protocol

| Field | Value |
|---|---|
| Split | seed=42, test_fraction=0.25 -> 12 held-out / 36 train |
| Tolerance | 75 ms |
| Polarity | adaptive count; v2 if confidence < 0.15 |
| Stage-2 RF | n_estimators=200 |
| Threshold | F1-max on calibration, min_recall=0.97 |
| Evaluation mode | retrospective full-record (not streaming) |

## MIT-BIH held-out

| Detector | Sensitivity | PPV | F1 |
|---|---:|---:|---:|
| Pan-Tompkins research reimplementation | 0.9908 | 0.9954 | 0.9931 |
| ElectroTrace two-stage | 0.9924 | 0.9879 | 0.9902 |
| Hamilton research reimplementation | 0.9990 | 0.9303 | 0.9634 |
| ElectroTrace Stage-1 adaptive | 0.9931 | 0.7553 | 0.8580 |

Record 207 remains a known difficult record for the ElectroTrace two-stage path.

Artifacts:
- mitdb_two_stage_locked_1.8.1.json
- mitdb_baseline_comparison_locked.json

## INCART external comparison

The repository's certified comparison uses 68 records and the same 75 ms matching rule:

| Detector | Sensitivity | PPV | F1 |
|---|---:|---:|---:|
| ElectroTrace two-stage | 0.3239 | 0.9679 | 0.4854 |
| WFDB gqrs | 0.9324 | 0.9265 | 0.9294 |
| WFDB sqrs | 0.7488 | 0.9513 | 0.8380 |

This is evidence of a large cross-database generalization gap for the current two-stage model. It is not a clinical performance claim.

Artifact: incart_wfdb_vs_electrotrace_comparison_2026-09-11.json

## Model artifact policy

The two legacy pickle model files are removed from the current source tree. Historical JSON artifacts may still mention their original paths because those reports are immutable provenance records.

New models should be stored outside git in .skops format with a JSON metadata sidecar. The migration script exists under scripts/convert_model_pickle_to_skops.py and requires an explicit trusted-local pickle opt-in.

## Explicit non-claims

- No clinical validation claim.
- No real-time/streaming claim.
- No population generalization claim.
- No detector-superiority claim.
- Historical validation artifacts are not silently rewritten during release hardening.
