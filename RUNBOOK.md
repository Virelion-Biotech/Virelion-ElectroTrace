# Step 5 runbook: frozen MIT-BIH, threshold CV, trace QA

> **Historical phase runbook — completed.** This file is retained to reproduce
> the Phase-5 reasoning path. It is no longer a current task list. The
> authoritative current evidence is `validation_reports/VALIDATION_STATUS.md`.
> Subsequent work completed the leakage-safe polarity audit, prospective EDB
> evaluation, certified baselines, QTDB characterization, and full 75-record
> INCART source-cohort characterization.


> **Polarity-gate update (2026-09-28):** for Issue #14, use
> `docs/POLARITY_GATE_DERIVATION_RUNBOOK.md`. The adaptive-polarity locked
> MIT-BIH result is a legacy non-regression audit because record 207/pooled
> MIT-BIH historically informed the mechanism. Do not use this older Step-5
> sequence to derive or manually tune the v2 gate.

This step closes the remaining evidence gaps around the current v4/windowed-std two-stage model without retraining the Random Forest or touching the locked 12-record MIT-BIH test split during threshold experiments.

## 1. Verify the frozen v4 model on locked MIT-BIH

Run:

    python -u scripts/evaluate_frozen_model_mitdb.py \
      --mitdb-dir .cache/physionet/mitdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --scale-method windowed_std

This evaluates the model file as-is on the exact 12 locked records. For adaptive
polarity, treat the result as legacy non-regression rather than prospective
held-out evidence. Use the dedicated polarity-gate runbook and a verified
derivation artifact for any non-default v2 gate.

Do not use this script to retune the model. A threshold override is diagnostic
only and is labeled as such in the generated report.

## 2. Run record-grouped threshold cross-validation

Run:

    python -u scripts/recalibrate_threshold_grouped_cv.py \
      --mitdb-dir .cache/physionet/mitdb \
      --incart-dir .cache/physionet/incartdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --folds 8

The corrected script fits threshold candidates only on the seven MIT-BIH calibration records and groups by record. The locked 12-record MIT-BIH test set is never read, and INCART labels are excluded from threshold fitting/model selection; INCART is evaluated descriptively only after the candidate is frozen.

Read these fields first:

- `grouped_cv.threshold_mean`
- `grouped_cv.threshold_std`
- `grouped_cv.pooled_summary_by_database`
- `current_fixed_threshold.summary_by_database`

A tight threshold spread across folds is more compatible with a stable global operating point. A wide spread indicates record-dependent calibration.

The candidate threshold remains a development diagnostic until evaluated externally. INCART labels are not used to choose it in the corrected workflow.

## 3. Inspect high-amplitude false positives

Run:

    python -u scripts/inspect_fp_traces.py \
      --incart-dir .cache/physionet/incartdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops

This samples false positives with amplitude ratio >= 0.6 from the high-error INCART records and writes:

- `validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.png`
- `validation_reports/experiments/2026-09-incart-fp-forensics/incart_fp_trace_samples.json`

Use the opposite-polarity samples first. This is manual QA, not a validation metric.

## 4. Interpret the combination

The current evidence already shows that resampling and robust scaling do not materially change mean INCART F1, while false-positive forensics show strong Stage-2 ranking AUC and a concentration of false positives in the post-QRS window.

That decision path is now closed for this phase. The frozen v4 configuration was subsequently evaluated prospectively on all 90 preregistered EDB records (F1 0.9341) without retraining or threshold recalibration. The negative/heterogeneous external tail was preserved rather than tuned away. Any **new** threshold or detector change would require a new untouched cohort.

## 5. Remaining scientific guardrails

INCART is development data for this model generation and remains development evidence even after the complete 75-record characterization. The seven malformed-edge source annotations, including I57, have now been resolved under a stricter two-certified-detector consensus rule; detector outputs validate the cleaned source annotation and do not replace labels.

Model files are intentionally not tracked in git. Record the exact .skops SHA-256 and runtime package versions with every experimental report.

## 6. Definition of done for this phase

Phase 5 is complete. The repository now contains:

1. frozen v4 MIT-BIH 12-record non-regression artifacts;
2. corrected grouped-CV threshold report;
3. trace-inspection PNG + manifest;
4. prospective 90-record EDB evaluation;
5. certified WFDB comparator artifacts;
6. QTDB 105-record delineation characterization;
7. complete 75-record INCART source-cohort characterization;
8. green CI/security and the released 1.9.0 package.

Future detector changes are new research work, not unfinished Phase-5 validation.
