# ElectroTrace external validation protocol

## Purpose

This protocol evaluates the current heuristic R-peak candidate detector against independent annotated ECG data. It is an algorithmic validation exercise, not a clinical validation study.

## Primary benchmark

Use the MIT-BIH Arrhythmia Database (PhysioNet/WFDB) as the first external benchmark. Keep the dataset outside the repository and record the dataset/record list used for every run.

## One-command run

After installing ElectroTrace with its test dependencies:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75
```

The script downloads MIT-BIH into the local cache, runs the exact detector adapter used for ElectroTrace external validation, compares it against `.atr` beat annotations, and writes a JSON report. The ECG files are never committed to Git.

## What is measured

For every record and for the aggregate dataset:

- sensitivity/recall
- positive predictive value
- F1 score
- true positives, false positives, false negatives
- median absolute detection timing error
- 95th-percentile absolute timing error

Matching is one-to-one and only pairs detections within the configured timing tolerance.

## Required reporting

Every external-validation report should state:

1. ElectroTrace version/commit.
2. Dataset name and exact dataset version or release identifier when available.
3. Complete record list included and excluded.
4. Annotation extension and beat-symbol policy.
5. Detector parameters.
6. Matching tolerance in milliseconds.
7. Per-record metrics and aggregate metrics.
8. Any records with import, annotation, or detector failures.

Do not report only the pooled sensitivity/PPV. Per-record results are necessary to expose performance heterogeneity.

## Recommended sequence

1. Run a development subset while detector settings can still change.
2. Freeze the detector configuration.
3. Run the full held-out record set.
4. Save the JSON report as a release artifact or study supplement.
5. Separately evaluate waveform-boundary algorithms on the QT Database.
6. Separately evaluate long-duration robustness on an appropriate long-term ECG database.

## Interpretation

A high sensitivity or PPV on MIT-BIH does **not** establish clinical safety, generalization to other populations, or suitability for diagnosis. Performance should be compared with the intended use case and independently validated on the target experimental data.

The current detector is deliberately described as a candidate generator. Do not label it "clinical-grade" without a dedicated validation and regulatory program.
