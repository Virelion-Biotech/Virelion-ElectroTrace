# ElectroTrace

[![PyPI](https://img.shields.io/pypi/v/electrotrace.svg)](https://pypi.org/project/electrotrace/)
[![Python](https://img.shields.io/pypi/pyversions/electrotrace.svg)](https://pypi.org/project/electrotrace/)
[![Tests](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/actions/workflows/test.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/actions/workflows/test.yml)
[![Release](https://img.shields.io/github/v/release/Virelion-Biotech/Virelion-ElectroTrace)](https://github.com/Virelion-Biotech/Virelion-ElectroTrace/releases/latest)
[![License](https://img.shields.io/github/license/Virelion-Biotech/Virelion-ElectroTrace)](LICENSE)

**Current release: 1.9.0**

A reproducible ECG/electrophysiology annotation, benchmarking, and research-validation toolkit.

ElectroTrace is built around a simple principle: **the evidence is the product**. It helps researchers detect beats, batch recordings, compare detectors under declared protocols, preserve record/subject-level statistics, and carry source hashes, software versions, detector configuration, and provenance alongside results.

> ElectroTrace is research software, not a clinical device. The current release does not claim clinical validation, real-time/streaming validation, population-wide generalization, or universal detector superiority.

## Install

Python 3.10+ is required.

Minimal installation:

```bash
python -m pip install electrotrace==1.9.0
```

Install common optional format/model support:

```bash
python -m pip install "electrotrace[all]==1.9.0"
```

Useful extras can also be installed separately:

```bash
python -m pip install "electrotrace[wfdb]==1.9.0"    # WFDB / PhysioNet records
python -m pip install "electrotrace[edf]==1.9.0"     # EDF files
python -m pip install "electrotrace[models]==1.9.0"  # .skops model artifacts
```

Confirm the installation:

```bash
electrotrace --help
electrotrace list
```

## 60-second demo

No external ECG dataset is required for the first run. Download the repository's small example CSV and detect R peaks:

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/Virelion-Biotech/Virelion-ElectroTrace/main/sample_data/sample_ecg.csv','sample_ecg.csv')" && electrotrace detect sample_ecg.csv --detector pan-tompkins --channel 0 -o peaks.csv
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

Software citation metadata is provided in `CITATION.cff`. Release metadata is also prepared in `zenodo.json`.

GitHub release: https://github.com/Virelion-Biotech/Virelion-ElectroTrace/releases/tag/v1.9.0

## Contributing

See `CONTRIBUTING.md`. Detector plugins can register through the `electrotrace.detectors` entry-point group.

## License

AGPL-3.0-or-later.
