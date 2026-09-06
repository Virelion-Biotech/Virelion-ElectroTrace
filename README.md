# Virelion-ElectroTrace

ElectroTrace is a Python/REST research toolkit for ECG and electrophysiology signal import, annotation, beat segmentation, R-peak detection, phenotype extraction, and leakage-aware evaluation.

## Scope

- CSV, EDF/EDF+, and WFDB ZIP import;
- signal validation and non-destructive preprocessing;
- interactive multi-channel annotation;
- adaptive-polarity R-peak candidate generation;
- two-stage Random Forest false-positive suppression;
- beat-level feature and phenotype extraction;
- subject/record-level validation and experimental-unit-aware statistics;
- reproducible dataset/software provenance;
- REST endpoints and a local web interface.

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

## Run

```bash
python server.py
```

The development server listens on `127.0.0.1:5000`. For non-local deployment, configure `ELECTROTRACE_API_KEY` and use an appropriate TLS-capable reverse proxy.

## R-peak detection

```python
from electrotrace import detect_r_peaks, detect_r_peaks_two_stage

peaks = detect_r_peaks(signal, fs=250, polarity="adaptive")
```

For the two-stage research pipeline, use `detect_r_peaks_two_stage` with a trained `CandidateSuppressor` and validate it on the target recording domain before use.

## API endpoints

- `POST /api/analyze`
- `POST /api/recording`
- `GET /api/recording/<id>/window`
- `POST /api/filter`
- `POST /api/detect/r-peaks`
- `POST /api/beats`
- `POST /api/segment`
- `POST /api/ml/train`
- `POST /api/ml/suggest`
- `POST /api/phenotype`
- `POST /api/statistics/compare`
- `POST /api/statistics/fdr`
- `POST /api/benchmark`

## Validation status

The repository contains a locked MIT-BIH held-out validation protocol and an INCART external pilot. The current INCART result documents domain shift and is not population-level external validation. Remaining validation work includes full INCART evaluation, certified WFDB baseline comparison, QTDB delineation analysis, and additional independent datasets.

Validation commands and exact protocols are documented under `docs/` and `validation_reports/`.

## Scientific limitations

ElectroTrace is research software, not a clinical device or validated clinical algorithm. MIT-BIH/QTDB results do not establish population generalization. ECG beats are not automatically independent biological replicates. The current two-stage benchmark is retrospective full-record evaluation and should not be interpreted as real-time performance.

## Testing

```bash
pytest -q
```

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and the exact validation protocol/dataset versions used in a study.
