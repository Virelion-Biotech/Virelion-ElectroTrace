# Cross-database detector policy

ElectroTrace is research software. This document defines the evidence boundary for
the two-stage Random Forest (RF) suppressor when moving between ECG databases.

## Current evidence

The repository preserves several scientifically different results that must not
be collapsed into one "generalization" number.

- The historical 1.8.1 two-stage model scored F1 0.9902 on its locked 12-record
  MIT-BIH split. Adaptive-polarity mechanism selection had historical MIT-BIH
  exposure, so this remains legacy non-regression evidence rather than clean
  prospective validation of that mechanism.
- The earlier cross-database INCART evaluation exposed a severe transfer failure:
  F1 0.4854 for the earlier feature generation, while certified WFDB `gqrs`
  scored F1 0.9294 under the same historical 68-record matching protocol.
- INCART then became development data. The complete frozen-v4/windowed-std
  source-cohort characterization now scores **75/75 official INCART records**
  at sensitivity 0.8977, PPV 0.7633, F1 0.8251. On the same 75 records,
  certified WFDB `gqrs` scores F1 0.9341 and `sqrs` F1 0.8469. This remains
  development evidence because INCART informed v4 feature-scaling changes.
  Seven formerly malformed edge annotations, including I57, are usable only
  under a documented rule requiring structural edge-only validity plus
  independent alignment authorization from both pinned certified WFDB
  detectors; detector outputs validate but never replace source labels.
- The frozen v4 detector subsequently completed a prospective external European
  ST-T Database run without retraining or threshold recalibration: 90/90 records,
  sensitivity 0.9056, PPV 0.9644, F1 0.9341. Record-level performance remained
  heterogeneous, so this is not a universal-transfer claim.
- The leakage-safe MIT-BIH polarity-gate audit froze the development-derived
  gates at 0.0/0.0 and retained the resulting degradation rather than tuning it
  away. That audit is explicitly labeled legacy non-regression.

## What the domain-shift investigations established

The INCART work separated multiple failure mechanisms rather than treating the
external gap as one RF problem.

1. The original catastrophic low-recall cases were dominated by Stage-1
   candidate starvation caused by a fragile global signal-scale estimate.
   Windowed standard-deviation scaling removed that failure mode on the
   prespecified focus records without meaningful MIT-BIH non-regression cost.
2. Stage-2 feature normalization had the same global-scale vulnerability and was
   moved to the same windowed-std scale family, requiring a clean retrain in the
   changed feature space.
3. In the later v4 INCART analysis, false positives concentrate in the post-QRS
   T-wave window. Stage-2 ranking AUC remains near ceiling even on the worst
   over-detecting records, which points to operating-point/calibration mismatch
   rather than a simple inability of the RF to rank candidates.
4. Resampling 257 to 360 Hz and robust signal scaling did not materially improve
   the Phase-3 result, so those are not supported as primary explanations.
5. The current grouped threshold-recalibration experiment fits candidate
   thresholds only on the seven MIT-BIH calibration records. INCART labels are
   excluded from threshold fitting/model selection and are used only for
   development-data evaluation after the candidate threshold is frozen.

## Research-use deployment policy

There is **no automatic out-of-domain detector** in ElectroTrace today. The
software therefore must not silently infer that a new recording is in-domain.

For an unseen database or laboratory domain:

- Do not assume the MIT-BIH-trained RF suppressor transfers because the signal is
  an ECG or because the sampling rate is similar.
- Do not silently change the RF threshold, Stage-1 scale, polarity gates,
  channel-selection rule, or beat-symbol policy after inspecting held-out
  outcomes.
- If the domain has not prospectively validated the frozen ElectroTrace
  configuration, treat the RF path as an experimental detector.
- For research workflows requiring a conservative established baseline on an
  unvalidated domain, run a certified/reference detector such as WFDB
  `gqrs`/appropriate registered baseline alongside ElectroTrace rather than
  presenting the RF output as the default truth.
- Any database-specific calibration or adaptation must be developed on a
  declared development partition and frozen before evaluation on a separate
  held-out partition or untouched database.
- A development-data improvement must remain labeled development evidence. It
  cannot be promoted to external validation by rerunning on the same exposed
  records.
- If a frozen configuration performs poorly on a prospective external set,
  preserve the negative result. A changed configuration requires a new
  prospective cohort.

This is a scientific reporting policy, not a clinical deployment policy.
ElectroTrace is not a clinical device and no current result establishes clinical
safety, population generalization, or regulatory suitability.

## Reproducible evidence

Primary artifacts and scripts include:

- `validation_reports/mitdb_two_stage_locked_1.8.1.json`
- `validation_reports/incart_two_stage_external_full_windowed_std_2026-09-09.json`
- `validation_reports/experiments/2026-09-incart-complete/ARCHIVED_RESULT.json`
- `validation_reports/certified_wfdb_incart_2026-09-11.json`
- `validation_reports/incart_domain_shift_analysis_2026-09-12.json`
- `validation_reports/experiments/2026-09-incart-fp-forensics/`
- `validation_reports/experiments/2026-09-recalibration/threshold_grouped_cv.json`
- `validation_reports/experiments/2026-09-polarity-width/mitdb_frozen_v4_evaluation.json`
- `scripts/recalibrate_threshold_grouped_cv.py`
- `scripts/analyze_incart_domain_shift.py`
- `scripts/incart_fp_offsets.py`
- `validation_reports/VALIDATION_STATUS.md`

The repository's validation ledger is authoritative when a historical document
or older issue conflicts with a later immutable result.
