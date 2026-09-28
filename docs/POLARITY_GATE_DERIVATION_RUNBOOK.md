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

## 2. Download the complete MIT-BIH Arrhythmia Database

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

## 3. Derive the gates

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

## 4. Run the locked non-regression audit

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

## 5. Interpret the result correctly

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
  that was not used to design or tune this mechanism.

## Diagnostic subsets

`--records ... --allow-diagnostic-subset` exists only for tests/diagnostics.
Such reports set `freeze_eligible=false` and the locked evaluator refuses to
use them to authorize a non-default v2 gate.
