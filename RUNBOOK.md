# Step 5 runbook: frozen MIT-BIH, threshold CV, trace QA

This step closes the remaining evidence gaps around the current v4/windowed-std two-stage model without retraining the Random Forest or touching the locked 12-record MIT-BIH test split during threshold experiments.

## 1. Verify the frozen v4 model on locked MIT-BIH

Run:

    python -u scripts/evaluate_frozen_model_mitdb.py \
      --mitdb-dir .cache/physionet/mitdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --scale-method windowed_std

This evaluates the model file as-is on the exact 12 locked records with adaptive polarity. Pay special attention to record 207. This is the missing matched MIT-BIH number for the v4 model.

Do not use this script to retune the model. A threshold override is available for a later, explicitly labelled analysis only.

## 2. Run record-grouped threshold cross-validation

Run:

    python -u scripts/recalibrate_threshold_grouped_cv.py \
      --mitdb-dir .cache/physionet/mitdb \
      --incart-dir .cache/physionet/incartdb \
      --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
      --folds 8

The script uses the seven MIT-BIH calibration records plus INCART and groups candidates by record. The locked 12-record MIT-BIH test set is never read.

Read these fields first:

- `grouped_cv.threshold_mean`
- `grouped_cv.threshold_std`
- `grouped_cv.pooled_summary_by_database`
- `current_fixed_threshold.summary_by_database`

A tight threshold spread across folds is more compatible with a stable global operating point. A wide spread indicates record-dependent calibration.

The `deployment_candidate_threshold` is a development diagnostic because it uses INCART labels. Do not adopt it from this run.

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

Therefore the next decision is:

- frozen MIT-BIH v4 result healthy + grouped CV threshold stable + trace QA consistent with T-wave/operating-point errors: test the candidate threshold on a genuinely fresh third database;
- frozen MIT-BIH v4 result degraded, especially record 207: fix the live regression before threshold work;
- grouped CV threshold unstable across records: do not force a single threshold; investigate record-conditional calibration/features;
- trace QA shows systematic non-T-wave false positives: revisit the feature/candidate design rather than only moving the threshold.

## 5. Remaining scientific guardrails

INCART is development data for this model generation. Any threshold selected using INCART labels remains development evidence until evaluated on a fresh database that was not used for model design or threshold selection.

I57 remains the annotation edge case requiring a second reference-detector check before treating all seven leading-negative-index records as equivalently repairable.

Model files are intentionally not tracked in git. Record the exact .skops SHA-256 and runtime package versions with every experimental report.

## 6. Definition of done for this phase

A clean Phase-5 closeout should contain:

1. frozen v4 MIT-BIH 12-record evaluation JSON;
2. grouped-CV threshold report;
3. trace-inspection PNG + manifest;
4. no CI regressions;
5. a fresh third-database evaluation plan before any deployment-threshold change.
