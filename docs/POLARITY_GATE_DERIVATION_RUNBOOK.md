# Leakage-safe polarity-gate derivation — fresh runtime

This runbook assumes a new machine/Colab with no repository checkout, no cached
PhysioNet files, and no installed ElectroTrace environment.

The goal is narrow: derive the adaptive-polarity confidence gates without using
the 12 locked MIT-BIH labels or INCART labels, freeze the resulting development
artifact, and run a non-regression audit. It does **not** make adaptive-polarity
MIT-BIH results prospective held-out evidence, because record 207 and pooled
MIT-BIH behavior historically informed the mechanism itself.

## 1. Clean setup

```bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

Do not proceed to data derivation unless the repository tests are green.

### Colab editable-install note

In an already-running Colab/IPython kernel, `pip install -e` can succeed while
a subsequent notebook Python cell still cannot import `electrotrace`. Editable
installs use a `.pth` file, and an already-running interpreter may not
re-process that newly-created file until restart.

For notebook Python cells, either restart the runtime after the editable install
or explicitly add the source tree once:

```python
import sys
from pathlib import Path

repo_src = Path("/content/Virelion-ElectroTrace/src").resolve()
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))

import electrotrace
print(electrotrace.__file__)
```

This changes only the current kernel's import path; it does not alter or install
another ElectroTrace distribution.

## 2. Restore the historical v4 model locally

The model artifacts are intentionally not stored on current `main`. The
trusted legacy pickle is still present in repository history at commit
`201096a`. Reconstruct the safe `.skops` + `.skops.json` pair from that
exact object. **Do not bypass the checksum before unpickling.**

```bash
MODEL_PKL=/content/incart_mitbih_model_windowed_std_2026-09-09.pkl
MODEL_SKOPS=validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops
EXPECTED_PKL_SHA=8732aa47051c76b043bd56e4d9dc82dc01307e0d9bcc6c5ed046fca19a27f5bb

git cat-file -e 201096a^{commit} 2>/dev/null || git fetch origin 201096a
git show 201096a:validation_reports/incart_mitbih_model_windowed_std_2026-09-09.pkl > "$MODEL_PKL"

echo "$EXPECTED_PKL_SHA  $MODEL_PKL" | sha256sum -c -

mkdir -p validation_reports
python scripts/convert_model_pickle_to_skops.py \
  "$MODEL_PKL" \
  "$MODEL_SKOPS" \
  --allow-pickle

test -s "$MODEL_SKOPS"
test -s "$MODEL_SKOPS.json"
sha256sum "$MODEL_SKOPS" "$MODEL_SKOPS.json"
```

Then verify semantic equivalence of the trusted pickle and reconstructed safe
artifact in the same runtime:

```python
from pathlib import Path
import numpy as np

from electrotrace.candidate_suppressor import CandidateSuppressor

pkl = Path("/content/incart_mitbih_model_windowed_std_2026-09-09.pkl")
skops = Path("validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops")

legacy = CandidateSuppressor.load(pkl, allow_pickle=True)
safe = CandidateSuppressor.load(skops)

assert legacy.metadata.to_dict() == safe.metadata.to_dict()
assert legacy.feature_names == safe.feature_names

expected = {
    "model_version": "rf-candidate-suppressor-v4",
    "feature_schema_version": "candidate-features-v4",
    "target_recall": 0.995,
    "threshold": 0.2751648051220502,
    "n_training_candidates": 102115,
    "n_positive_candidates": 63526,
    "n_negative_candidates": 38589,
    "random_seed": 42,
    "n_estimators": 150,
    "calibration_candidates": 24841,
    "calibration_method": "locked_MITBIH_calibration_records",
}
for key, value in expected.items():
    assert getattr(safe.metadata, key) == value, (key, getattr(safe.metadata, key), value)

rng = np.random.default_rng(20260928)
X = rng.normal(size=(64, safe.model.n_features_in_))
np.testing.assert_allclose(
    legacy.predict_proba(X),
    safe.predict_proba(X),
    rtol=0.0,
    atol=0.0,
)
print("Model reconstruction verified.")
print("features:", safe.model.n_features_in_)
print("threshold:", safe.metadata.threshold)
print("sklearn_version recorded in sidecar:", safe.metadata.sklearn_version)
```

The historical Phase-5 `.skops` artifact had SHA-256
`4b4a7c6dfc7b53edc30bf8224c6f93210b79db27db6bc4a9924e176ff405e575`.
A fresh conversion can produce different serialized bytes under a different
runtime, so that old `.skops` hash is informative but is **not** the acceptance
criterion. The checksum-gated source pickle plus metadata/prediction
equivalence checks above are the reconstruction guardrail.

## 3. Download the complete MIT-BIH Arrhythmia Database

```python
import wfdb
wfdb.dl_database("mitdb", dl_dir=".cache/physionet/mitdb")
```

Optional human check:

```bash
ls .cache/physionet/mitdb/*.hea | wc -l
# expected: 48
```

The derivation script itself is stricter than this count: in normal mode it
requires `.hea`, `.dat`, and `.atr` for every one of the 36 standard
MIT-BIH records outside the locked 12-record split. Missing/corrupt records
abort the run; there is no silent skip-to-a-smaller-pool behavior.

## 4. Derive the gates

```bash
python -u scripts/derive_polarity_thresholds_mitdb_extended.py \
  --mitdb-dir .cache/physionet/mitdb \
  --output validation_reports/experiments/2026-09-polarity-thresholds/mitdb_extended_derivation.json
```

Read these fields before looking at the selected grid point:

1. `freeze_eligible` must be `true`.
2. `protocol.development_pool_size` must be 36.
3. `protocol.locked_heldout_labels_used` must be `false`.
4. `protocol.incart_used` must be `false`.
5. Inspect `non_identifying_dimensions`.
6. Use **`recommended_thresholds`**, not `selected_result`.
7. Inspect `parameter_status` and `engagement_summary` to see what the data
   actually identified and which records changed final polarity.

If a dimension is non-identifying, the report automatically keeps its
historical default in `recommended_thresholds` rather than exposing a
tie-broken edge value as the frozen recommendation.

## 5. Run the locked non-regression audit

Do not manually copy threshold numbers if you can avoid it. Give the evaluator
the derivation artifact so it can verify provenance and load the frozen values
itself:

```bash
python -u scripts/evaluate_frozen_model_mitdb.py \
  --mitdb-dir .cache/physionet/mitdb \
  --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
  --polarity-threshold-report \
    validation_reports/experiments/2026-09-polarity-thresholds/mitdb_extended_derivation.json \
  --output validation_reports/experiments/2026-09-polarity-thresholds/mitdb_locked_nonregression.json
```

A non-default `--v2-gate-confidence` is rejected unless the same
freeze-eligible derivation report is supplied. If a CLI threshold is supplied
alongside the report, it must exactly agree with the report's frozen value.

## 6. Interpret the result correctly

For a full 12-record adaptive-polarity run, the evaluator should report:

```text
evidence_status = legacy_validation_non_regression
```

That label is deliberate. The clean development-only threshold derivation
prevents **further** held-out tuning, but it cannot erase the fact that the
adaptive-polarity mechanism was historically motivated using record 207 and
pooled MIT-BIH behavior.

Therefore:

- use the result as a reproducible non-regression/audit result;
- do not describe it as prospective held-out validation of adaptive polarity;
- do not retune either gate after reading the 12-record result;
- keep Issue #14 open until the real-data derivation/audit artifact exists and
  the historical limitation is documented;
- prospective adaptive-polarity validation requires an independent dataset
  that was not used to design or tune this mechanism. In particular, neither
  MIT-BIH nor INCART is prospective for the combined adaptive/width mechanism:
  MIT-BIH informed the count/v2 polarity work, while INCART informed the width
  override mechanism.

## Diagnostic subsets

`--records ... --allow-diagnostic-subset` exists only for tests/diagnostics.
Such reports set `freeze_eligible=false` and the locked evaluator refuses to
use them to authorize a non-default v2 gate.
