# Validation status

**Software:** electrotrace 1.8.1  
**Primary scientific endpoint:** held-out MIT-BIH test records only (not full-pool)  
**Current hardening branch:** merged into `main`; there is no separate hardening branch to check out

## Unit tests

The locked historical validation artifact reports 103 passing tests at the time of the 1.8.1 scientific freeze. The release-hardening work added public-API, detector-registry, model-persistence, annotation-audit, forensics, and Step-5 regression coverage; current CI is the authoritative result for `main`.

## Locked two-stage protocol

| Field | Value |
|---|---|
| Split | seed=42, test_fraction=0.25 -> 12 held-out / 36 train |
| Tolerance | 75 ms |
| Polarity | adaptive count; historical v2 fallback default 0.15 (mechanism/gate historically informed by pooled MIT-BIH; adaptive locked-split results are legacy non-regression) |
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

The table above preserves historical locked-split results for reproducibility.
Because adaptive-polarity mechanism selection used information from record 207 /
pooled MIT-BIH, those adaptive rows must not be described as prospective
held-out validation of that mechanism. Issue #14's clean derivation prevents
additional tuning but cannot undo that historical exposure.

Artifacts:
- mitdb_two_stage_locked_1.8.1.json
- mitdb_baseline_comparison_locked.json

## INCART external comparison

Two model generations are present in the immutable validation history and must be kept distinct:

| Model | Feature schema | Records | Sensitivity | PPV | F1 |
|---|---|---:|---:|---:|---:|
| Earlier two-stage model | candidate-features-v3 | 68 | 0.3239 | 0.9679 | 0.4854 |
| Windowed-std two-stage model | candidate-features-v4 | 68 | 0.9011 | 0.7594 | 0.8242 |

The second row is the current windowed-std model and is the relevant post-fix external result. WFDB gqrs on the same 68-record protocol has sensitivity 0.9324, PPV 0.9265, F1 0.9294; sqrs has sensitivity 0.7488, PPV 0.9513, F1 0.8380. These are cross-database detector results under the stated matching protocol, not clinical performance claims.

Artifacts:
- incart_two_stage_external_full_windowed_std_2026-09-09.json
- incart_wfdb_vs_electrotrace_comparison_2026-09-11.json

### Phase-3 preprocessing/threshold ablation

A 2026-09-21 Phase-3 run evaluated 66 usable local records; the two missing records were then rerun separately and both completed with no skipped records. The combined I09/I61 artifact is `incart_phase3_missing_I09_I61.json`. Across the original 66-record run, mean record-level F1 was 0.8236 on raw signals, 0.8235 after 257->360 Hz resampling, and unchanged by robust scaling. The threshold sweep is exploratory only; its highest mean record-level F1 on that run was 0.8729 at threshold 0.50, but this threshold was not selected for deployment.

Interpretation: the Phase-3 ablation does not support resampling or robust scaling as the main explanation for the remaining INCART gap. Further work should focus on record-level polarity/candidate errors and score/threshold domain shift rather than immediately changing the model architecture.
### False-positive forensics (2026-09-22)

`scripts/incart_fp_offsets.py` classifies every false positive on the 68-record windowed-std run by its position relative to neighbouring reference beats. 74% sit 150-450 ms after the preceding true beat (the post-QRS T-wave window); approximately 0% are duplicate detections. Stage-2 ranking AUC is 0.998 even on the 13 worst over-detecting records, so the candidate ranking is strong and the remaining error is concentrated in operating-threshold calibration rather than simple ranking failure. These are development-data diagnostics, not held-out test results.

Seven INCART records were previously excluded for a leading negative annotation index. Six repair cleanly under `--annotation-policy drop_edges`; I57 remains excluded pending a second reference-detector check.

`scripts/recalibrate_threshold_grouped_cv.py` is the direct follow-up. It performs record-grouped cross-validation across the MIT-BIH calibration records plus INCART, never reads the locked 12-record MIT-BIH held-out split, and never retrains the RF. It reports both an out-of-fold threshold estimate and a full-pool deployment-candidate threshold; neither is validated until tested on a genuinely fresh database.


### Phase-5 consolidated validation (2026-09-27)

The Phase-5 frozen v4 artifact scores the locked 12-record MIT-BIH partition using windowed-standard-deviation scaling, adaptive polarity, the existing 75 ms tolerance, and the existing operating threshold (0.2752). The 12 records had no skips; aggregate sensitivity was 0.9912, PPV 0.9891, and F1 0.9901. **Interpret this as legacy non-regression, not prospective held-out validation of adaptive polarity**: the v2 fallback/gate was historically informed by pooled MIT-BIH behavior including record 207.

The original seven-record polarity-width sweep is **non-identifying** for the width gate: its Stage-1 objective is flat over the tested values, so the reported 0.05 is a deterministic tie-break artifact rather than a derived biological/algorithmic cutoff and must not be promoted. `scripts/derive_polarity_thresholds_mitdb_extended.py` is the replacement development-only procedure; it requires the complete 36-record non-held-out MIT-BIH pool in normal mode and reports explicit non-identifying dimensions. Grouped threshold recalibration remains development-only. None of these results establishes deployment validity; prospective adaptive-polarity validation still requires a genuinely independent database that did not inform the mechanism.

The consolidated INCART domain-shift analysis classifies the 68-record comparison as 22 candidate-generation failures, 10 over-suppression cases, 17 acceptable cases, and 19 mixed/other cases. The poor-performing cohort is dominated by low Stage-1 candidate coverage; the report therefore directs the next work toward front-end candidate generation and only then threshold-domain-shift analysis. Polarity is predominantly positive and does not explain the split by itself. The accompanying false-positive figure is a manual-inspection aid and is not a metric.

Artifacts:
- validation_reports/experiments/2026-09-polarity-width/mitdb_frozen_v4_evaluation.json
- validation_reports/experiments/2026-09-polarity-width/mitdb_polarity_width_threshold.json
- validation_reports/experiments/2026-09-recalibration/threshold_grouped_cv.json
- validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.json
- validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.png
- validation_reports/incart_domain_shift_analysis_2026-09-12.json
- validation_reports/incart_domain_shift_record_table_2026-09-12.csv
- validation_reports/experiments/phase5_reproducibility_manifest.json

## Model artifact policy

The two legacy pickle model files are removed from the current source tree. Historical JSON artifacts may still mention their original paths because those reports are immutable provenance records.

New models should be stored outside git in .skops format with a JSON metadata sidecar. The migration script exists under scripts/convert_model_pickle_to_skops.py and requires an explicit trusted-local pickle opt-in.

## Explicit non-claims

- No clinical validation claim.
- No real-time/streaming claim.
- No population generalization claim.
- No detector-superiority claim.
- Historical validation artifacts are not silently rewritten during release hardening.
