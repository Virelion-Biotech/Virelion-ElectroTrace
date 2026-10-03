# Rigorous research validation workflow

ElectroTrace's validation workflow is designed around the experimental unit, not the individual beat. The default unit for external ECG detector validation is the recording. The default unit for downstream cohort inference is the subject/recording unit declared by the study manifest.

## Locked external detector validation

For MIT-BIH, freeze the detector configuration before evaluating the final held-out record set. The reproducible command is:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75 \
  --polarity adaptive \
  --bootstrap 2000 \
  --seed 42
```

The dataset remains outside Git. The generated report contains the complete record list, detector configuration, software version/commit when supplied by the runtime, annotation policy, matching tolerance, per-record metrics, pooled metrics, macro record-level estimates, bootstrap confidence intervals, and any requested records without a result.

## Why both pooled and macro metrics are required

Pooled sensitivity/PPV/F1 answer the question for all beats treated as one population and can therefore be dominated by long recordings or subjects with many beats. Macro record-level estimates give each recording equal weight. Both are retained so performance heterogeneity is visible rather than hidden.

## Uncertainty and reproducibility

Mean confidence intervals use a two-sided Student-t interval because the number of independent records can be small. Bootstrap intervals resample complete records, never individual beats, preserving the intended experimental unit.

Every `DatasetManifest` is canonicalized and hashed with SHA-256. The manifest includes dataset identity/version, record list, annotation policy, preprocessing, detector configuration, split information, calibration records, software version, and software commit. The hash should be archived with any paper, supplement, model card, or downstream dataset derived from the run.

## Phenotype quality control

`electrotrace.phenotype_validation.quality_report` checks mathematical and structural integrity without imposing disease-specific thresholds. It detects missing/non-finite fields, duplicate R indices, non-monotonic time, invalid RR intervals, and inconsistent heart-rate calculations.

`aggregate_by_unit` is provided for downstream analyses that would otherwise treat beats as independent observations. Aggregate within the declared experimental unit first, then perform statistical inference across those units.

## Recommended scientific release sequence

1. Finish detector and threshold changes on development data only.
2. Freeze detector configuration and dataset record manifest.
3. Execute the full MIT-BIH benchmark and archive its JSON report.
4. Validate waveform-boundary algorithms independently on QT Database.
5. Validate long-duration robustness on a suitable long-term ECG database.
6. Validate phenotype reproducibility on the intended scientific cohort.
7. Benchmark downstream ML with subject/record-level splits and an untouched external test set.
8. Archive manifests, reports, software commit, and parameter files together.

A strong detector score is not a clinical validation claim. ElectroTrace remains research software until an appropriate clinical validation and regulatory program has been completed.
