# ElectroTrace

[![PyPI](https://img.shields.io/pypi/v/electrotrace.svg)](https://pypi.org/project/electrotrace/)
[![Python](https://img.shields.io/pypi/pyversions/electrotrace.svg)](https://pypi.org/project/electrotrace/)
[![Tests](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/actions/workflows/test.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/actions/workflows/test.yml)
[![Release](https://img.shields.io/github/v/release/Virelion-Biotech/Virelion-ElectroTrace)](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/releases/latest)
[![License](https://img.shields.io/github/license/Virelion-Biotech/Virelion-ElectroTrace)](LICENSE)

**Current release: 1.10.0**

A reproducible ECG/electrophysiology annotation, benchmarking, and research-validation toolkit.

ElectroTrace is built around a simple principle: **the evidence is the product**. It helps researchers detect beats, batch recordings, compare detectors under declared protocols, preserve record/subject-level statistics, and carry source hashes, software versions, detector configuration, and provenance alongside results.

> ElectroTrace is research software, not a clinical device. The current release does not claim clinical validation, real-time/streaming validation, population-wide generalization, or universal detector superiority.


<!-- VALIDATION_PRODUCT_STATUS:BEGIN -->
## Validation and product completion status

**Reviewed 2026-10-15; implementation base `bd3d3da98a12`.** **Status: External algorithmic characterization exists; clinical/streaming claims unestablished.** This is an evidence-backed scope assessment, not a percentage-complete estimate or a declaration that the product is finished.

| Area | Current evidence / remaining gate |
|---|---|
| Software / operational checks | LUDB v1 Colab failed after six patients because its annotation-triplet assumption rejected incomplete references. All 24 artifact hashes and 256 pinned source files verified. Repaired notebook completed all 200 patients / 2,400 leads locally; all 411 recovery artifact hashes verified. Source/input/result-verified checkpoints passed full and partial resume checks. |
| Scientific or domain evidence | Version 1.10.0 preserves cross-database detector evidence, exposure boundaries and QTDB QRS-boundary tolerance results. Previously exposed cohorts are development/regression evidence, not fresh external tests. LUDB recovery scored 21,843 paired QRS references and excluded 122 with audited reference defects; both-boundary success at 20 ms is 69.0% with patient-clustered uncertainty. Manual-center assistance and reference exclusions limit this exploratory endpoint. |
| Remaining for the declared product scope | Predeclare and evaluate a genuinely unused cohort with subject-level uncertainty and failure accounting; qualify lead-selection benefit prospectively; validate streaming latency/robustness if shipping a streaming product; freeze intended-use/device boundaries and operational acceptance criteria. |

Evidence and limits: [LUDB failure/recovery review](validation_reports/COLAB_LUDB_REVIEW_2026-10-15.md) and [repository report](validation_reports/VALIDATION_STATUS.md). The [organization status index](https://github.com/Virelion-Biotech/Virelion-HeartTwin/blob/main/docs/STACK_VALIDATION_STATUS.md) separates software, numerical, data-readiness and empirical qualification. Code/schema execution or a completed download is not a substitute for endpoint-specific validation.
<!-- VALIDATION_PRODUCT_STATUS:END -->

## Install

Python 3.10+ is required.

Minimal installation:

```bash
python -m pip install electrotrace==1.10.0
```

Install common optional format/model support:

```bash
python -m pip install "electrotrace[all]==1.10.0"
```

Useful extras can also be installed separately:

```bash
python -m pip install "electrotrace[wfdb]==1.10.0"    # WFDB / PhysioNet records
python -m pip install "electrotrace[edf]==1.10.0"     # EDF files
python -m pip install "electrotrace[models]==1.10.0"  # .skops model artifacts
```

Confirm the installation:

```bash
electrotrace --help
electrotrace list
```

## 60-second demo

No external ECG dataset is required for the first run. Download the repository's small example CSV and detect R peaks:

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/Virelion-Biotech/Virelion-ElectroTrace/v1.10.0/sample_data/sample_ecg.csv','sample_ecg.csv')" && electrotrace detect sample_ecg.csv --detector pan-tompkins --channel 0 -o peaks.csv
```

That creates:

```text
peaks.csv
peaks.csv.manifest.json
```

`peaks.csv` contains detected sample indices and times. The adjacent manifest records the detector configuration, input hash, ElectroTrace version, channel selection, and provenance metadata.

The sample is an onboarding fixture only; it is **not** validation evidence and should not be used to infer clinical performance.

## Use your own recording

ElectroTrace currently accepts:

- CSV with a monotonic time column named `time`, `t`, `timestamp`, `time_s`, or `seconds`, plus one or more numeric signal columns;
- EDF with the `edf` extra installed;
- WFDB records with the `wfdb` extra installed.

Detect one recording:

```bash
electrotrace detect recording.edf \
  --detector pan-tompkins \
  --channel 0 \
  -o peaks.csv
```

Batch a directory:

```bash
electrotrace batch data/ \
  --detector pan-tompkins \
  --workers 4 \
  -o results/
```

Batch output is deliberately analysis-friendly:

```text
results/
├── beats.csv
├── records.csv
├── subjects.csv
├── failures.csv
├── manifest.json
├── batch_state.json
└── peaks/
```

Subject identity is never inferred from filenames. When subject-level aggregation is appropriate, provide a two-column `record,subject_id` CSV with `--subject-map`.

## Validation and benchmarking

Validate a local WFDB record against its annotations:

```bash
electrotrace validate .cache/physionet/mitdb/100 \
  --detector pan-tompkins \
  --tolerance-ms 75 \
  -o validation.json
```

Compare detectors over a local WFDB collection:

```bash
electrotrace bench .cache/physionet/mitdb \
  --detectors pan-tompkins,hamilton \
  --tolerance-ms 75 \
  -o bench.json
```

Render any JSON result as a self-contained HTML report:

```bash
electrotrace report bench.json -o bench.html
```

Primary comparative statistics should be interpreted at the record or subject level rather than treating individual beats as independent biological replicates.

## Python API

```python
from electrotrace import load_recording, detect_r_peaks

record = load_recording("recording.csv")
signal = record.signals[next(iter(record.signals))]

peaks = detect_r_peaks(
    signal,
    record.sampling_rate_hz,
    polarity="adaptive",
)

print(peaks[:10])
```

## EP calibration handoff

ElectroTrace can now turn a measured ECG into a calibration-grade, provenance-linked observation for CardiEP/CardiInfer:

```bash
electrotrace calibration prepare recording.csv \
  --entity-id subject-001 \
  --kind ecg \
  --detector stage1 \
  --units mV \
  --output-dir calibration/ \
  -o calibration-handoff.json
```

The existing handoff envelope remains `contract_version: 1.0` / `schema_version: electrotrace-ep-calibration-v1` for HeartTwin compatibility; its referenced artifact is explicitly versioned `electrotrace-ecg-measurement-v2`. Consumers should read `summary.schema_version` or artifact metadata before interpreting the artifact schema. The ECG handoff now writes the versioned `electrotrace-ecg-measurement-v2` artifact. It preserves the decoded raw waveform and original-unit values, source-file SHA-256, acquisition/device/filter declarations, lead names and order, all detected R-peak timestamps, selected beat-specific multi-lead waveforms, ectopic/paced annotations when supplied, RR/rhythm screening, QRS measurements, beat-to-beat variability, and a robust template with an explicit fit-eligibility gate. The flat v1 fields remain as a compatibility layer for existing handoff consumers. HeartTwin exposes the same path as `electrical.prepare_calibration`.

See [ECG measurement bundle v2](docs/ECG_MEASUREMENT_BUNDLE_V2.md) for schema contents, unit/linkage metadata, QC semantics, and the frozen-set acceptance plan. Pre-aligned EAM activation, activation-map, and repolarization-map files can also be registered as calibration observations, but ElectroTrace deliberately does not invent spatial registration: callers must supply the coordinate frame and units.

The baseline-noise and template-residual values are scales, **not probabilities of measurement correctness**. The v2 bundle deliberately leaves the global acceptance gate as `not_demonstrated` until it is run on an independent, manually adjudicated frozen set. QRS measurement uncertainty is not passed to CardiInfer as a Gaussian noise term unless the caller supplies an independently calibrated measurement-error scale and its source. This handoff remains an inverse-model input contract, not evidence that the downstream EP model or inferred patient parameters are clinically valid.

## Why ElectroTrace exists

Most ECG software answers “which detector can I run?” ElectroTrace is aimed at a stricter research question:

> **Under one explicit protocol, how did this detector behave, and can someone else reproduce the answer?**

The toolkit emphasizes:

- record/subject-level rather than beat-level primary statistics;
- one-to-one matching under a declared tolerance;
- locked splits with explicit seeds;
- bootstrap uncertainty over records;
- SHA-256 input hashes;
- software/git provenance;
- detector and model metadata;
- explicit evidence boundaries and non-claims.

## Evidence snapshot

ElectroTrace ships a research detector as well as benchmarking infrastructure. The detector evidence is intentionally reported with its limitations rather than compressed into one headline score.

| Protocol | Detector / generation | Records | Sensitivity | PPV | F1 | Evidence status |
|---|---|---:|---:|---:|---:|---|
| Locked MIT-BIH | two-stage v3 | 12 | 0.9924 | 0.9879 | 0.9902 | historical locked model comparison |
| MIT-BIH leakage-safe gate audit | frozen v4 | 12 | 0.9390 | 0.9913 | 0.9644 | legacy non-regression |
| INCART complete source cohort | frozen v4 | 75 | 0.8977 | 0.7633 | 0.8251 | exposed development characterization |
| INCART same-cohort baseline | WFDB gqrs | 75 | 0.9372 | 0.9310 | 0.9341 | certified reference baseline |
| European ST-T | frozen v4 | 90 | 0.9056 | 0.9644 | 0.9341 | preregistered prospective external evaluation |
| SVDB | selector v2 | 78 | 0.9680 | 0.9878 | 0.9778 | prospective selector transfer |
| Zymed LTSTDB subset | selector v3 | 18 | 0.8718 | 0.9650 | 0.9160 | prospective null-switch evaluation |

Important boundaries:

- INCART informed v4 development and is not fresh external validation.
- Historical MIT-BIH adaptive-polarity evidence remains legacy/non-regression where prior mechanism exposure applies.
- The EDB, LTAFDB, SVDB, and Zymed cohorts became exposed after their respective first runs.
- Selector v3 made zero prospective switches on the 18-record Zymed subset; that run therefore does not demonstrate a prospective starvation-rescue benefit.
- QTDB supports QRS-boundary characterization, not a conventional full-beat detector benchmark.
- No result supports a clinical-device, regulatory, streaming, population-generalization, or universal-superiority claim.

For the complete evidence history, hashes, workflow IDs, and interpretation boundaries, see `validation_reports/VALIDATION_STATUS.md`.

## Documentation

- Hosted documentation: https://virelion-biotech.github.io/Virelion-ElectroTrace/
- Quickstart: `docs/QUICKSTART.md`
- Reproducibility: `docs/REPRODUCIBILITY.md`
- Cross-database policy: `docs/CROSS_DATABASE_POLICY.md`
- Limitations: `docs/LIMITATIONS.md`
- Validation status: `validation_reports/VALIDATION_STATUS.md`

## Citation

Software citation metadata is provided in `CITATION.cff`. Release metadata is provided in `.zenodo.json` for Zenodo GitHub integration.

GitHub release: https://github.com/Virelion-Biotech/Virelion-ElectroTrace/releases/tag/v1.10.0

## Contributing

See `CONTRIBUTING.md`. Detector plugins can register through the `electrotrace.detectors` entry-point group.

## License

AGPL-3.0-or-later.
