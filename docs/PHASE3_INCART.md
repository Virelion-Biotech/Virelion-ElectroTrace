# Phase-3 INCART validation

This phase tests the three cheap explanations for the current cross-database performance gap while keeping the historical MIT-BIH-trained model frozen.

## What is tested

- raw INCART signal;
- resampling to 360 Hz;
- robust per-record median/MAD scaling;
- resampling plus robust scaling;
- exploratory RF threshold sweeps.

No retraining is performed. No deployment threshold is selected from INCART labels.

The script is scripts/phase3_incart_ablation.py.

## Colab

The historical pickle model is retrieved from the 201096a source snapshot, while the hardening branch uses skops for new model artifacts. skops is designed for safer scikit-learn model persistence and supports auditing unknown serialized types before loading.

```python
# Cell 1: clone the hardening branch
!git clone -b release-hardening-v1.9 https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
%cd Virelion-ElectroTrace

# Cell 2: install validation dependencies
!python -m pip install -e ".[all,dev]"

# Cell 3: retrieve the historical model from the exact 1.8.1 source snapshot
!wget -q -O /content/incart_mitbih_model_windowed_std_2026-09-09.pkl \
  https://raw.githubusercontent.com/Virelion-Biotech/Virelion-ElectroTrace/201096a/validation_reports/incart_mitbih_model_windowed_std_2026-09-09.pkl
!sha256sum /content/incart_mitbih_model_windowed_std_2026-09-09.pkl

# Cell 4: convert trusted legacy model to skops
!python scripts/convert_model_pickle_to_skops.py \
  /content/incart_mitbih_model_windowed_std_2026-09-09.pkl \
  /content/incart_mitbih_model_windowed_std_2026-09-09.skops \
  --allow-pickle

# Cell 5: download INCART if it is not already cached
import wfdb
from pathlib import Path
incart_dir = Path("/content/incartdb")
incart_dir.mkdir(exist_ok=True)
if not list(incart_dir.glob("*.hea")):
    wfdb.dl_database("incartdb", dl_dir=str(incart_dir))

# Cell 6: run the full frozen Phase-3 ablation
!python scripts/phase3_incart_ablation.py \
  --incart-dir /content/incartdb \
  --model /content/incart_mitbih_model_windowed_std_2026-09-09.skops \
  --output validation_reports/incart_phase3_ablation.json

# Cell 7: inspect summaries
import json
from pathlib import Path
report = json.loads(Path("validation_reports/incart_phase3_ablation.json").read_text())
print(json.dumps(report["summary_by_transform"], indent=2, sort_keys=True))
print("skipped:", len(report["skipped_records"]))
```

For a strict historical-model conversion, use the scikit-learn version associated with the original artifact before conversion. PyPI has scikit-learn 1.9.0 and 1.9.1; the latter was released September 10, 2026.

After the run, copy the JSON back as a new dated evidence artifact. Do not overwrite the existing locked reports.