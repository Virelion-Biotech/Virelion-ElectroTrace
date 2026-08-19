# ElectroTrace external validation protocol

## Purpose

This protocol evaluates the current heuristic R-peak candidate detector against independent annotated ECG data. It is an algorithmic validation exercise, not a clinical validation study.

## Primary benchmark

Use the MIT-BIH Arrhythmia Database (PhysioNet/WFDB) as the first external benchmark. Keep the dataset outside the repository and record the complete dataset/record list used for every run.

## Locked definitive protocol

The definitive run is intentionally fail-closed. Every requested record must load, annotate, detect, and score successfully. A run with any failed record is **incomplete** and exits non-zero after writing its diagnostic report. Successful-record metrics must not be promoted as the final benchmark when records are missing.

Primary locked settings:

- record set: complete MIT-BIH record list returned by the dataset release
- primary channel: channel index 0 only; no annotation-informed channel selection
- detector: `electrotrace.validation_detectors:detect_r_peaks`
- polarity: adaptive unless a separate prespecified positive/negative experiment is declared
- minimum candidate spacing: 250 ms
- primary matching tolerance: 75 ms
- bootstrap replicates: 2000
- bootstrap unit: record
- seed: 42

The definitive runner also performs a prespecified tolerance sensitivity analysis at **50, 75, 100, and 150 ms** using the same detector configuration. The 75-ms result is the primary result; the other tolerances are sensitivity analyses, not post-hoc optimization.

The command is:

```bash
python scripts/validate_mitdb.py \
  --cache-dir .cache/physionet/mitdb \
  --output validation_reports/mitdb_rpeak_validation.json \
  --tolerance-ms 75 \
  --channel 0 \
  --polarity adaptive \
  --bootstrap 2000 \
  --seed 42
```

The script downloads MIT-BIH into the local cache, runs the exact detector adapter used for ElectroTrace external validation, compares it against `.atr` beat annotations, writes a self-contained JSON report, and returns non-zero if any requested record fails. ECG files are never committed to Git.

## Cryptographic provenance

The definitive report fingerprints the **actual downloaded input files**, not merely the logical dataset name/version. Each record-associated input file is SHA-256 hashed and stored in the manifest; the complete input-file hash map participates in the deterministic manifest SHA-256.

The report also records the **executed Git `HEAD`**. A supplied `--software-commit`, `GITHUB_SHA`, or `ELECTROTRACE_GIT_COMMIT` value is accepted only when it exactly matches the checked-out `HEAD`. Definitive validation therefore cannot intentionally report a different code revision from the code that actually ran.

For cohort/ML studies, `DatasetManifest` additionally supports an explicit record→subject mapping and validates that it covers the record set exactly when supplied. Do not represent subject identity as an unvalidated parallel array.

## Detector-output integrity

Validation never coerces malformed detector outputs into valid predictions. Detector outputs must be finite, integer-valued, non-negative sample indices in strictly increasing order with no duplicates. Reference annotation indices are subjected to the same structural checks.

This prevents silent truncation, duplicate removal, negative-index dropping, or type coercion from improving reported performance.

## Reference definition

The primary reference population is all `.atr` annotations whose symbols belong to the frozen ElectroTrace beat-symbol whitelist recorded in the report manifest. This is deliberately a heterogeneous beat benchmark, not a normal-beat-only benchmark.

The report must preserve the exact whitelist. Secondary analyses should stratify performance by beat/rhythm class where scientifically relevant rather than interpreting one aggregate F1 as uniform performance across all morphologies.

## What is measured

For every record and for the aggregate dataset:

- sensitivity/recall
- positive predictive value
- F1 score
- true positives, false positives, false negatives
- mean and median signed detection timing error
- timing SD
- mean and median absolute timing error
- 95th-percentile absolute timing error
- maximum absolute timing error

The report additionally contains pooled metrics, macro record-level metrics, Student-t confidence intervals for record-level means, **record-bootstrap confidence intervals for the macro record mean**, tolerance sensitivity, and the adaptive-polarity audit. Matching is one-to-one and only pairs detections within the locked primary timing tolerance.

## Retrospective scope

The detector uses record-level centering/scaling. The benchmark is therefore explicitly **retrospective full-record evaluation**. It makes no real-time or causal-streaming claim.

The 250-ms minimum candidate spacing is a deliberate physiological operating assumption. Events closer than this are unresolved by design and must not be interpreted as evidence of detector failure outside the detector's defined operating regime without separate analysis.

## Full-record vs standardized test-period analysis

The primary engineering analysis evaluates the complete records. Conventional benchmark comparisons may use standardized test-period exclusions. Such a comparison must be run as a **separate frozen protocol and manifest**; it must not be silently substituted for the primary full-record analysis.

## Experimental unit

External R-peak validation treats the **record** as the independent experimental unit for uncertainty estimation. Never bootstrap or report confidence intervals by resampling individual beats as though they were independent records.

For downstream phenotype and cohort analyses, declare the subject/record experimental unit and aggregate repeated beat-level observations before inferential testing unless the study design explicitly supports a different model. When subjects have multiple records, preserve an explicit record→subject mapping and perform subject-level separation where subject independence is required.

## Required reporting

Every external-validation report should state:

1. ElectroTrace version and executed Git `HEAD`.
2. Dataset name and exact dataset version or release identifier.
3. Complete record list and a complete record-failure list.
4. Exact SHA-256 for every input file used and the resulting deterministic manifest hash.
5. Annotation extension and exact beat-symbol policy.
6. Detector parameters, polarity mode, and polarity decisions.
7. Channel-selection policy.
8. Minimum candidate spacing.
9. Matching tolerance in milliseconds.
10. Per-record metrics.
11. Pooled and macro record-level metrics.
12. Confidence/bootstrapping method, random seed, and number of replicates.
13. Explicit bootstrap estimand (macro record mean versus pooled metric).
14. Tolerance sensitivity analysis.
15. Full-record versus any separately defined test-period analysis.

Do not report only pooled sensitivity/PPV. Pooled values can hide record-level heterogeneity.

## Recommended sequence

1. Freeze the detector configuration, dataset record list, channel, symbol policy, tolerance, and report schema.
2. Run the definitive full-record benchmark.
3. Inspect record-level heterogeneity and polarity decisions without changing the locked detector.
4. Archive the JSON validation report, input-file hash map, and manifest hash as the primary study artifacts.
5. Separately evaluate any standardized test-period protocol with its own frozen manifest.
6. Separately evaluate waveform-boundary algorithms on the QT Database with an explicit boundary-event protocol.
7. Separately evaluate long-duration robustness on an appropriate long-term ECG database.
8. Validate phenotypes on the intended scientific cohort.
9. Benchmark downstream ML with subject/record-level splits and an untouched external test set.

## Interpretation

A high sensitivity or PPV on MIT-BIH does **not** establish clinical safety, generalization to other populations, or suitability for diagnosis. Performance should be compared with the intended use case and independently validated on the target experimental data.

The current detector is deliberately described as a candidate generator. Do not label it "clinical-grade" without a dedicated validation and regulatory program.
