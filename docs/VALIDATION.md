# ElectroTrace external validation protocol

## Purpose

This protocol evaluates the current heuristic R-peak candidate detector against independent annotated ECG data. It is an algorithmic validation exercise, not a clinical validation study.

## Primary benchmark

Use the MIT-BIH Arrhythmia Database (PhysioNet/WFDB) as the first external benchmark. Keep the dataset outside the repository and record the dataset/record list used for every run.

## Locked one-command run

After installing ElectroTrace with its test dependencies:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75 \
  --polarity adaptive \
  --bootstrap 2000 \
  --seed 42
```

The script downloads MIT-BIH into the local cache, runs the exact detector adapter used for ElectroTrace external validation, compares it against `.atr` beat annotations, and writes a self-contained JSON report. The ECG files are never committed to Git.

## What is measured

For every record and for the aggregate dataset:

- sensitivity/recall
- positive predictive value
- F1 score
- true positives, false positives, false negatives
- median absolute detection timing error
- 95th-percentile absolute timing error

The report additionally contains pooled metrics, macro record-level metrics, Student-t confidence intervals for record-level means, and record-bootstrap uncertainty intervals. Matching is one-to-one and only pairs detections within the configured timing tolerance.

## Experimental unit

External R-peak validation treats the **record** as the independent experimental unit for uncertainty estimation. Never bootstrap or report confidence intervals by resampling individual beats as though they were independent records.

For downstream phenotype and cohort analyses, declare the subject/record experimental unit and aggregate repeated beat-level observations before inferential testing unless the study design explicitly supports a different model.

## Required reporting

Every external-validation report should state:

1. ElectroTrace version/commit.
2. Dataset name and exact dataset version or release identifier.
3. Complete record list included and excluded.
4. Annotation extension and beat-symbol policy.
5. Detector parameters and polarity mode.
6. Matching tolerance in milliseconds.
7. Per-record metrics.
8. Pooled and macro record-level metrics.
9. Confidence/bootstrapping method, random seed, and number of replicates.
10. Any records with import, annotation, or detector failures.
11. A deterministic manifest hash linking the run to its dataset/configuration state.

Do not report only pooled sensitivity/PPV. Pooled values can hide record-level heterogeneity.

## Recommended sequence

1. Run a development subset while detector settings can still change.
2. Freeze the detector configuration and record manifest.
3. Run the full held-out record set.
4. Archive the JSON validation report and manifest hash as release/study artifacts.
5. Separately evaluate waveform-boundary algorithms on the QT Database with an explicit boundary-event protocol.
6. Separately evaluate long-duration robustness on an appropriate long-term ECG database.
7. Validate phenotypes on the intended scientific cohort.
8. Benchmark downstream ML with subject/record-level splits and an untouched external test set.

## Interpretation

A high sensitivity or PPV on MIT-BIH does **not** establish clinical safety, generalization to other populations, or suitability for diagnosis. Performance should be compared with the intended use case and independently validated on the target experimental data.

The current detector is deliberately described as a candidate generator. Do not label it "clinical-grade" without a dedicated validation and regulatory program.
