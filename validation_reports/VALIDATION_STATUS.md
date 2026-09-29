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

`scripts/recalibrate_threshold_grouped_cv.py` is the direct follow-up. The corrected workflow fits candidate thresholds **only on the seven MIT-BIH calibration records**, never reads the locked 12-record MIT-BIH held-out split, never uses INCART labels for threshold fitting/model selection, and never retrains the RF. INCART is evaluated only after the candidate threshold is frozen and remains exposed development data. Any candidate threshold still requires a genuinely fresh database before it can be described as externally validated.


### Cross-database research-use policy

The RF suppressor must not be assumed to transfer to an unseen database merely
because the input is ECG. ElectroTrace currently has no validated automatic
out-of-domain classifier, so it does not silently infer that a recording is
in-domain. On an unvalidated domain, the RF path remains experimental and
should be run alongside an established/reference baseline when comparative
reliability matters. Database-specific calibration or adaptation must use a
declared development partition and be frozen before separate held-out or
untouched external evaluation. Development-data gains on INCART remain
development evidence and cannot be relabeled as external validation.

The full policy and supporting artifacts are in
`docs/CROSS_DATABASE_POLICY.md`.

### Phase-5 consolidated validation (2026-09-27)

The Phase-5 frozen v4 artifact scores the locked 12-record MIT-BIH partition using windowed-standard-deviation scaling, adaptive polarity, the existing 75 ms tolerance, and the existing operating threshold (0.2752). The 12 records had no skips; aggregate sensitivity was 0.9912, PPV 0.9891, and F1 0.9901. **Interpret this as legacy non-regression, not prospective held-out validation of adaptive polarity**: the v2 fallback/gate was historically informed by pooled MIT-BIH behavior including record 207.

The original seven-record polarity-width sweep is **non-identifying** for the width gate: its Stage-1 objective is flat over the tested values, so the reported 0.05 is a deterministic tie-break artifact rather than a derived biological/algorithmic cutoff and must not be promoted. `scripts/derive_polarity_thresholds_mitdb_extended.py` is the replacement development-only procedure; it requires the complete 36-record non-held-out MIT-BIH pool in normal mode and reports explicit non-identifying dimensions. Grouped threshold recalibration remains development-only. None of these results establishes deployment validity; prospective adaptive-polarity validation still requires a genuinely independent database that did not inform the mechanism.

The consolidated INCART domain-shift analysis classifies the 68-record comparison as 22 candidate-generation failures, 10 over-suppression cases, 17 acceptable cases, and 19 mixed/other cases. The poor-performing cohort is dominated by low Stage-1 candidate coverage; the report therefore directs the next work toward front-end candidate generation and only then threshold-domain-shift analysis. Polarity is predominantly positive and does not explain the split by itself. The accompanying false-positive figure is a manual-inspection aid and is not a metric.

### Issue #14 leakage-safe polarity freeze (v3, 2026-09-28)

The corrected v3 derivation used all 36 canonical MIT-BIH development records and no
locked held-out or INCART labels. Its prespecified selection rule froze
`v2_gate_confidence=0.0` and `width_override_confidence=0.0`. The exact zero
values are deterministic lower-gate tie-break choices within a broad best-F1
plateau, not uniquely identified biological/algorithmic thresholds: global-best
development points include v2 gates 0.00-0.21 and width gates 0.00-0.20.

The subsequent full 12-record locked audit used those frozen values without
retraining or protocol overrides. Aggregate sensitivity was **0.9390**, PPV
**0.9913**, and F1 **0.9644** (25,286 TP, 221 FP, 1,644 FN). Record 207 fell to
F1 **0.2512** (sensitivity 0.1511, PPV 0.7454). This degradation is retained
rather than tuned away: it is the leakage-safe outcome of the development-only
selection procedure.

For comparison, the historical-default 0.15/0.38 audit had F1 0.9901 and record
207 F1 0.9080. That earlier result remains useful as legacy non-regression
history but cannot be used to select gates after seeing the locked split.

The v3 locked report is explicitly labeled
`legacy_validation_non_regression`, not prospective validation. Historical
mechanism-selection exposure remains: MIT-BIH informed the adaptive polarity
mechanism and INCART informed the width override. Prospective validation still
requires a database that informed neither mechanism.

Provenance hashes from the completed run:
- v3 derivation SHA-256: `7c69699c6dcc0cce5745d90c77806a8377ba98e28cc5a49dec415ec64eaf7ca7`
- v3 locked audit SHA-256: `23803a8d9e0ee891fac4ba91d678873a48a993608437ff0440a0d5c2ca863c9e`
- code commit used by both artifacts: `3dde141153c4f559b054aa57020a3c7da38b63a6`

Issue #14 is considered complete once these results are recorded because the
threshold-selection procedure was fixed before v3 scoring, the locked labels
were excluded from threshold selection, frozen aggregate/per-record results
were reproduced, and the unavoidable historical exposure remains explicitly
labeled as legacy validation.

Artifacts:
- validation_reports/experiments/2026-09-polarity-width/mitdb_frozen_v4_evaluation.json
- validation_reports/experiments/2026-09-polarity-width/mitdb_polarity_width_threshold.json
- validation_reports/experiments/2026-09-recalibration/threshold_grouped_cv.json
- validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.json
- validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.png
- validation_reports/incart_domain_shift_analysis_2026-09-12.json
- validation_reports/incart_domain_shift_record_table_2026-09-12.csv
- validation_reports/experiments/phase5_reproducibility_manifest.json

## Prospective European ST-T external validation (2026-09-28)

Issue #26 preregistered a one-shot external evaluation on all 90 canonical
European ST-T Database (EDB) records before any EDB detector outcome was
inspected. The successful first scored run was GitHub Actions run
`36442894752` at code commit
`a610ec3511b5937da683be43dc72711637354e89`.

The frozen protocol used channel 0, `atr` reference annotations, the existing
75 ms ElectroTrace matcher, `windowed_std` Stage-1 scaling, adaptive polarity,
the corrected v3 frozen polarity gates (v2=0.0, width=0.0), recovery disabled,
the stored v4 model threshold, and no retraining. All **90/90** preregistered
records scored; none were skipped or repaired.

Primary prospective result:

| Records | Sensitivity | PPV | F1 | Macro mean record F1 | Macro median record F1 |
|---:|---:|---:|---:|---:|---:|
| 90 | **0.9056** | **0.9644** | **0.9341** | **0.9094** | **0.9927** |

Counts were 790,560 reference beats, 715,963 true positives, 74,597 false
negatives, and 26,411 false positives.

Performance was heterogeneous rather than uniformly mediocre: 79/90 records
had F1 >= 0.90 and 48/90 had F1 >= 0.99, but 11/90 fell below 0.80 and 8/90
fell below 0.50. The worst records were `e0106` (F1 0.0269), `e0133`
(0.0707), `e0501` (0.2594), `e0817` (0.2768), `e1302` (0.2982),
`e0205` (0.3656), `e0403` (0.3936), and `e0415` (0.4663). These
outcomes are now exposed development information and must not be used to alter
the detector and then re-label EDB as prospective validation.

All 90 polarity selections were positive. That observation is descriptive of
the frozen first run only; it must not be used to perform post-hoc lead or
polarity selection while preserving a prospective EDB claim.

Important scope: this was a prospective external ElectroTrace evaluation under
the preregistered 75 ms full-record matching protocol. It is **not** an
ANSI/AAMI `bxb` compliance analysis and supports no clinical-validation,
population-generalization, or regulatory-performance claim.

Permanent first-run provenance:
- protocol SHA-256: `05a1aa75069f21fb80ac649da9a800aa60c32d80a12e01ea17fce22c63cd3e8f`
- regenerated v3 derivation SHA-256: `1d9d3966e7c3ab157d5ccf69c6770ec7db7ef29ebd6a59e623c2e45fc19d3369`
- first-run result SHA-256: `fc9a3cb0833bce7ad15ba32098ebd328835eb72f133cb73550e6f6a1057aa491`
- GitHub Actions artifact ID: `10978828919` (`electrotrace-edb-prospective-first-run`)
- model SHA-256 recorded by the report: `2ba0c6945352522c52a80ec3371ea3d60ce576c4c711100e391090270b2b5fe4`
- model sidecar SHA-256: `71c6cf759551d51cb112a033cb904c14c4f82f5f32314937b0e62debdd92c589`

EDB is now exposed for this project. Any EDB-motivated model/protocol change
must use a different untouched database for the next prospective evaluation.

## Multi-lead selector development and prospective transfer (2026-09-28 to 2026-09-29)

The first prospective EDB run exposed a small but severe fixed-channel failure
tail. Post-hoc EDB diagnostics showed that the catastrophic records generally
had abundant Stage-1 candidates on channel 0 but candidates aligned to the
wrong deflections; channel 1 post-hoc rescued the 11 records below F1 0.80.
EDB became exposed development data at that point.

### Selector v1: EDB-informed retained-probability fallback

Selector v1 (`edb-informed-retained-probability-v1`) preserved channel 0
unless channel-0 median retained Stage-2 probability was below 0.995 and
channel 1 had a higher median. It was prospectively evaluated on the first
30 minutes of all 84 Long-Term AF Database records before LTAFDB was exposed.

Prospective LTAFDB result:

| Records | Sensitivity | PPV | F1 | Macro mean F1 | Macro median F1 |
|---:|---:|---:|---:|---:|---:|
| 84 | 0.9551 | 0.9021 | **0.9279** | 0.9180 | 0.9755 |

Channel 1 was selected on 4/84 records. The immutable first-run result SHA-256
is `72ecf37dba63c4fd0c5771710f08961c5283ea11ef9461d7ed72dcc52b3fe1fd`
(Actions run `36463413708`).

A post-hoc comparator showed that selector v1 only slightly improved aggregate
F1 over fixed channel 0 (0.9279 vs 0.9266). Two of four switches helped
(records 105 and 203), while two hurt (45 and 53); record 53 dropped from
F1 0.5257 on channel 0 to 0.1695 on the selected channel. LTAFDB is therefore
exposed development data, and selector v1 should not be described as a robustly
validated general lead selector.

### Selector v2: EDB+LTAFDB-informed quality consensus

Selector v2 (`edb-ltafdb-informed-quality-consensus-v2`) was frozen after
EDB and LTAFDB were exposed. It retained the v1 p50 gate and required channel 1
also to have higher retained QRS-band fraction and higher Stage-2 retention
fraction. The exact rule was preregistered before any SVDB outcome was
inspected.

The first two SVDB executions aborted before completion on annotation-window
boundary handling. Neither produced a result artifact or exposed
sensitivity/PPV/F1. After correcting WFDB's inclusive annotation-`sampto`
plumbing without changing the scientific protocol, the first completed run
scored all 78 canonical SVDB records.

Prospective SVDB selector-v2 result:

| Records | Sensitivity | PPV | F1 | Macro mean F1 | Macro median F1 |
|---:|---:|---:|---:|---:|---:|
| 78 | **0.9680** | **0.9878** | **0.9778** | **0.9752** | **0.9962** |

Counts were 184,581 reference beats, 178,678 TP, 2,206 FP and 5,903 FN.
Channel 0 was selected on 66/78 records and channel 1 on 12/78. Minimum
record F1 was 0.6124. The immutable first-run JSON SHA-256 is
`6e02fe1c2591fb2789e82eaac5bd5087e52a7f477bba20d5ad94fc63b6dd916f`
(Actions run `36501118072`, commit
`f1784519f720ddcb9c877db490e80655752f8068`, artifact ID
`11005980780`).

Post-hoc fixed-lead comparison reproduced all 78 archived selected channels,
quality inputs and selected-lead metrics before comparison:

| Strategy | Sensitivity | PPV | F1 | Macro mean F1 | Min record F1 |
|---|---:|---:|---:|---:|---:|
| Fixed channel 0 | 0.9050 | 0.9892 | 0.9452 | 0.9278 | 0.0489 |
| Fixed channel 1 | 0.6945 | 0.9280 | 0.7945 | 0.7315 | 0.0022 |
| Prospective selector v2 | **0.9680** | 0.9878 | **0.9778** | **0.9752** | 0.6124 |
| Post-hoc oracle best lead | 0.9802 | 0.9934 | 0.9868 | 0.9849 | 0.6480 |

Ten of the 12 prospective channel-1 switches improved record F1. Two
regressed: record 862 by 0.0023 F1 and record 801 substantially, from
channel-0 F1 0.9845 to selected-channel F1 0.6124. Thus selector v2 provides
strong prospective aggregate transfer but is not a perfect record-level
lead-quality oracle and should not be silently retuned on SVDB.

The post-hoc comparator JSON SHA-256 is
`94aa1cf8ac3b749133743b5977e099347933f016cf5696828c8e18f3c1a64cad`
(Actions run `36501839556`, artifact ID `11005747805`).

SVDB is now exposed. It is an untouched record set for the v2 prospective
evaluation, but belongs to the same broader MIT-BIH/Beth Israel ecosystem; this
is not a fully institutionally independent clinical-validation cohort. Any
selector v3 informed by SVDB requires another untouched record set for
prospective evaluation.


### Selector v3: conservative starvation-rescue rule

Selector v3 (`edb-ltafdb-svdb-informed-starvation-rescue-v3`) was frozen only
after EDB, LTAFDB and SVDB were exposed. It switches from channel 0 to channel 1
only when the primary retained rate is below 30 bpm, the alternate is above
30 bpm, and the alternate is more than 2x the primary. No annotation, diagnosis,
rhythm, reference-count, probability, QRS-band or third-channel information is
used for that decision.

Its first prospective external evaluation used the preregistered 18-record,
three-channel Zymed subset of the Long-Term ST Database, with only channels 0
and 1 available to the selector and a fixed first-30-minute window. The run
completed all 18 records with no skips, loaded reference annotations only after
lead selection, never loaded channel 2, and did not retrain or override the
frozen detector.

Prospective Zymed LTSTDB selector-v3 result:

| Records | Sensitivity | PPV | F1 | Macro mean F1 | Macro median F1 | Min F1 | Max F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 0.8718 | 0.9650 | **0.9160** | 0.9031 | 0.9807 | 0.3344 | 1.0000 |

Counts were 46,777 reference beats, 40,779 TP, 1,480 FP and 5,998 FN.
Channel 0 was selected on **18/18** records; channel 1 was selected on **0/18**.
Therefore selector v3 made **no prospective switches** in this cohort.

That null-switch outcome is scientifically informative but narrow. It shows that
the conservative rule did not create a false switch in these 18 untouched
records, but it does not prospectively demonstrate the benefit of a starvation
rescue because the rescue condition never fired. In particular, two low-F1
records (`s30761` and `s30771`) still had primary retained rates above the
frozen 30-bpm ceiling, so the preregistered rule correctly remained inactive.
Those exposed outcomes must not be used to retune v3 while preserving a
prospective claim.

Permanent first-run provenance:
- Actions run: `36532383204`
- head commit: `0028d4773deb523b04b8cdea99fa9a97488543fb`
- artifact ID: `11025074694` (`electrotrace-ltstdb-zymed-selector-v3-first-run`)
- artifact ZIP digest: `sha256:946e79c7fa366e590ddb1ef480820ac96f3aedc9e80c4012cf1f376169cb8e37`
- first-run JSON SHA-256: `853051f9c8921f053b72b52b418c79d33a01f642f254cea6b22ea6c694c46dd0`
- protocol Git blob SHA: `c0caa0a83da12ba6176cc19f46a9fe3616d1397f`
- protocol file SHA-256: `543f95763d7a3b1003e8e434ba0cc7a8e0f926fa7f076454fd37b8f5660623d1`
- frozen model SHA-256: `5fd8675103137b6adbd2188c11c002d022d80399797f628c22e82118c2fa8e94`
- model sidecar SHA-256: `963ad2213561aa422a14d7d7b22f603b7f4c6675433c4439e29829436a07150f`
- polarity derivation SHA-256: `1d9d3966e7c3ab157d5ccf69c6770ec7db7ef29ebd6a59e623c2e45fc19d3369`

The Zymed LTSTDB subset is now exposed development data. It cannot be reused as
prospective evidence for any selector changed after this run.


## QT Database QRS-boundary characterization (2026-09-29)

The frozen `qrs-edge-energy-v1` signal-only boundary locator completed the
prespecified QT Database v1.0.0 tolerance-curve study on **105/105 records**
with no primary exclusions. The primary endpoint intentionally supplies the
midpoint of each manual `.q1c` QRS onset/offset pair as the center so that
boundary localization is measured separately from R-peak event detection.
There were **3,623** manual QRS pairs; onset and offset searches reported a
boundary on all 3,623.

Primary reference-centered success fractions:

| Tolerance | Onset | Offset | Both boundaries | 95% record-bootstrap interval for both |
|---:|---:|---:|---:|---:|
| 20 ms | 0.8029 | 0.6969 | **0.5893** | 0.5169–0.6660 |
| 40 ms | 0.9246 | 0.8802 | **0.8454** | 0.7852–0.9009 |
| 60 ms | 0.9520 | 0.9431 | **0.9249** | 0.8762–0.9683 |
| 80 ms | 0.9661 | 0.9685 | **0.9542** | 0.9177–0.9879 |
| 100 ms | 0.9967 | 0.9970 | **0.9945** | 0.9866–1.0000 |

The 11 records with independent `.q2c` annotations provided a confirmatory
inter-observer reference. Among 404 matched observer-2 QRS events, joint
onset+offset agreement was 0.7698 at 20 ms, 0.9629 at 40 ms, 0.9950 at 60 ms,
and 0.9975 at both 80 and 100 ms. Observer-event coverage was 404/487 relative
to q1c and 404/404 relative to q2c under the frozen 75 ms center-matching rule.

The secondary end-to-end path deliberately used the frozen raw Stage-1 detector,
not the two-stage suppressor. It matched 3,591/3,623 manual QRS events
(sensitivity **0.9912**) but emitted 172,505 detections, for PPV **0.0208**.
QTDB marks selected beats for waveform delineation rather than every ECG beat,
so this low PPV must not be interpreted as a conventional full-beat detector
benchmark. Conditional joint-boundary success among the 3,591 matched events
rose from 0.5937 at 20 ms to 0.9669 at 100 ms.

Permanent provenance:
- workflow run: `36592730416`
- head commit: `e8b8331e5f0e604f7f4869c4d2d6dbf23e312f14`
- artifact ID: `11044358209` (`electrotrace-qtdb-qrs-delineation-tolerance-curve`)
- artifact ZIP digest: `sha256:ecdebd6acfffc06396246b5144d4ebaaa4dc0483e433e9c1f3f8c2068297ae55`
- result JSON SHA-256: `e7c827b2df1cf808c9ccf717dcc54153d1bd425898e4c75576f4e38326cadb94`
- frozen protocol SHA-256: `745ef90cfe20400ff4355d3546acf0980d45b7a1c5279ac4e65b03755fae3dd1`

This is confirmatory algorithmic characterization of QRS boundary localization,
not clinical validation, and QTDB's mixed source provenance does not make it a
fresh institutionally independent R-peak generalization cohort.

## Model artifact policy

The two legacy pickle model files are removed from the current source tree. Historical JSON artifacts may still mention their original paths because those reports are immutable provenance records.

New models should be stored outside git in .skops format with a JSON metadata sidecar. The migration script exists under scripts/convert_model_pickle_to_skops.py and requires an explicit trusted-local pickle opt-in.

## Explicit non-claims

- No clinical validation claim.
- No real-time/streaming claim.
- No population generalization claim.
- No detector-superiority claim.
- Historical validation artifacts are not silently rewritten during release hardening.
