# Quickstart

ElectroTrace 1.9.0 is available from PyPI and requires Python 3.10+.

## Install

Minimal:

```bash
python -m pip install electrotrace==1.9.0
```

With common optional dependencies:

```bash
python -m pip install "electrotrace[all]==1.9.0"
```

Check the installation:

```bash
electrotrace --help
electrotrace list
```

## First run with the example ECG

Download the repository's onboarding CSV and run Pan-Tompkins:

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/Virelion-Biotech/Virelion-ElectroTrace/v1.9.0/sample_data/sample_ecg.csv','sample_ecg.csv')" && electrotrace detect sample_ecg.csv --detector pan-tompkins --channel 0 -o peaks.csv
```

Outputs:

```text
peaks.csv
peaks.csv.manifest.json
```

The example CSV is an onboarding fixture, not validation evidence.

## Supported recording formats

### CSV

CSV input needs a strictly increasing time column named one of:

`time`, `t`, `timestamp`, `time_s`, or `seconds`.

All other columns are treated as signal channels and must be numeric.

```bash
electrotrace detect recording.csv --detector pan-tompkins --channel 0 -o peaks.csv
```

### EDF

Install EDF support first:

```bash
python -m pip install "electrotrace[edf]==1.9.0"
electrotrace detect recording.edf --detector pan-tompkins --channel 0 -o peaks.csv
```

### WFDB

Install WFDB support:

```bash
python -m pip install "electrotrace[wfdb]==1.9.0"
```

Then validate a local annotated record:

```bash
electrotrace validate .cache/physionet/mitdb/100 --detector pan-tompkins -o validation.json
```

## Batch processing

```bash
electrotrace batch data/ --detector pan-tompkins --workers 4 -o results/
```

The batch directory contains per-beat, per-record, failure, provenance, restart-state, and optional subject-level outputs.

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

ElectroTrace never infers subject identity from filenames. Supply `--subject-map` when subject-level aggregation is scientifically appropriate.

## Benchmarking

```bash
electrotrace bench .cache/physionet/mitdb \
  --detectors pan-tompkins,hamilton \
  --tolerance-ms 75 \
  -o bench.json
```

Generate an HTML report:

```bash
electrotrace report bench.json -o bench.html
```

## What to read next

Before interpreting detector comparisons, read:

- [Validation evidence](validation/README.md)
- [Limitations](LIMITATIONS.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Cross-database policy](CROSS_DATABASE_POLICY.md)

ElectroTrace is research software and is not a clinical device.
