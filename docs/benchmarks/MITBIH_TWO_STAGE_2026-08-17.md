# MIT-BIH Two-Stage R-Peak Benchmark

**Date:** 2026-08-17  
**Software:** ElectroTrace 1.5.x  
**Dataset:** MIT-BIH Arrhythmia Database, v1.0.0  
**Records:** 48 total; 12 held-out test records in the fixed local development split

## Purpose

This benchmark evaluates the Stage-2 false-positive suppressor on top of ElectroTrace's existing Stage-1 candidate detector. The test records are kept separate from model fitting and threshold calibration.

## Stage 1 baseline

The current single-polarity Stage-1 detector uses 250 ms minimum distance and prominence of 0.5 signal SD.

On the 12 held-out records used for the development benchmark:

| Metric | Stage 1 |
|---|---:|
| Reference beats | 26,923 |
| Candidate detections | 35,473 |
| TP | 22,423 |
| FP | 13,050 |
| FN | 4,500 |
| Sensitivity | 83.29% |
| PPV | 63.21% |
| F1 | 71.87% |

## Stage 2 result

A Random Forest verifier was trained on the Stage-1 candidate features from the training records. A separate 20% candidate-level calibration subset from the training data was used to select the probability threshold targeting 99.5% training recall. The held-out records were not used for fitting or threshold selection.

| Metric | Stage 1 | Stage 2 |
|---|---:|---:|
| Reference beats | 26,923 | 26,923 |
| Detections | 35,473 | 24,416 |
| TP | 22,423 | 21,405 |
| FP | 13,050 | 3,011 |
| FN | 4,500 | 5,518 |
| Sensitivity | 83.29% | 79.50% |
| PPV | 63.21% | **87.67%** |
| F1 | 71.87% | **83.39%** |

The suppressor therefore reduced the number of candidate detections by approximately 31.2% and false positives by approximately 76.9%, while sensitivity decreased by approximately 3.8 percentage points.

## Important limitation

This result does **not** establish a clinically validated R-peak detector. It is a development benchmark showing that a second-stage candidate verifier can substantially improve precision. Difficult records remain challenging, and a verifier cannot recover beats that Stage 1 never proposes.

## Polarity experiment

A full 48-record exploratory comparison of detecting both positive and negative peaks showed:

| Stage-1 strategy | Sensitivity | PPV | F1 |
|---|---:|---:|---:|
| Existing single-polarity | 94.05% | 57.33% | **71.23%** |
| Both polarities merged | 90.91% | 49.84% | **64.38%** |

The both-polarities approach was therefore **not promoted to the default**. It did recover severe polarity-dependent failures (for example, record 207 sensitivity increased from 14.68% to 93.06%), but the additional candidates created too many false positives across the full database.

The polarity-adaptive option remains available for controlled experiments. It must be benchmarked together with the Stage-2 verifier before it is considered for production use.

## Reproducibility requirements

Future detector comparisons should retain:

1. the exact MIT-BIH database version;
2. the record-level train/test split;
3. Stage-1 parameters;
4. feature-schema/model version;
5. threshold-calibration records or calibration partition;
6. matching tolerance;
7. per-record TP/FP/FN and timing error;
8. pooled and macro-averaged metrics.

No MIT-BIH source files are stored in the repository.