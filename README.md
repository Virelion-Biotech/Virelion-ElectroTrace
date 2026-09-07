# Virelion-ElectroTrace

ElectroTrace is a Python/REST research toolkit for ECG and electrophysiology signal import, annotation, beat segmentation, R-peak detection, phenotype extraction, and leakage-aware evaluation.

## What it contains

- CSV, EDF/EDF+, and WFDB ZIP import.
- Signal validation and non-destructive preprocessing.
- Interactive multi-channel annotation.
- Adaptive-polarity R-peak candidate generation.
- Two-stage Random Forest false-positive suppression.
- Beat-level feature and phenotype extraction.
- Subject/record-level validation and experimental-unit-aware statistics.
- REST endpoints and a local web interface.
- Reproducible dataset/software provenance.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[test]'
```

## Usage

Start the local server:

```bash
python server.py
```

The development server listens on `127.0.0.1:5000`.

Python:

```python
from electrotrace import detect_r_peaks

peaks = detect_r_peaks(signal, fs=250, polarity="adaptive")
```

The REST API exposes recording, filtering, R-peak detection, beat segmentation, ML, phenotype, statistics, and benchmark operations under `/api/`.

## Inputs and outputs

**Inputs:** ECG/electrophysiology recordings in supported formats, sampling rate and channel metadata, optional annotations, preprocessing parameters, and optional trained detection models.

**Outputs:** validated recordings, annotations, R-peak locations, beat segments, extracted features/phenotypes, statistical comparisons, benchmark reports, and provenance records.

## Validation

The repository contains a locked MIT-BIH held-out validation protocol and an INCART external pilot. Validation documentation and reports are under `docs/` and `validation_reports/`. The current INCART work is an external pilot, not population-level external validation.

Run software tests with:

```bash
pytest -q
```

## Limitations

ElectroTrace is research software, not a clinical device or validated clinical algorithm. MIT-BIH/QTDB results do not establish population generalization. ECG beats are not automatically independent biological replicates. The current two-stage benchmark is retrospective full-record evaluation and should not be interpreted as real-time performance.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
