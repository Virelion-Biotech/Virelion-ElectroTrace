# One-shot prospective EDB validation runbook

This runbook executes Issue #26: the first genuinely prospective external
evaluation of the frozen v4/windowed_std ElectroTrace detector on the European
ST-T Database (EDB).

The protocol is already preregistered in
`validation_protocols/edb_prospective_v1.json`. The evaluator verifies the
protocol file SHA-256 before scoring.

## Scientific status

EDB was not used to design or tune the current adaptive-polarity mechanism,
width override, RF suppressor, model threshold, or Stage-1 scale selection.
That makes the **first scored EDB run** prospective for this frozen model
generation.

After the first EDB result is inspected, EDB is exposed. Any model/protocol
change motivated by EDB must treat EDB as development data and use a different
untouched database for the next prospective validation.

## Locked primary protocol

- all 90 canonical EDB records
- channel 0 only
- `atr` reference annotations
- ElectroTrace `DEFAULT_BEAT_SYMBOLS`
- 75 ms matching tolerance
- frozen `rf-candidate-suppressor-v4`
- stored model threshold only
- `windowed_std` Stage-1 scale
- adaptive polarity
- v2 and width gates loaded from the verified v3 polarity derivation artifact
- expected v2 gate = 0.0
- expected width gate = 0.0
- recovery disabled
- no retraining
- annotation policy = `error`
- any missing/failed record aborts the primary run
- no EDB-based lead, threshold, polarity, scale, architecture, or model selection

There is no post-hoc performance pass threshold. Report the frozen result
completely, whether good or bad.

## 1. Update to the preregistered evaluator

```bash
cd /content/Virelion-ElectroTrace
git fetch origin
git checkout main
git pull --ff-only origin main
python -m pip install -e ".[test]"
pytest -q tests/test_edb_prospective.py tests/test_evaluate_v2_gate.py
```

## 2. Restore/verify the frozen model if necessary

Use `docs/POLARITY_GATE_DERIVATION_RUNBOOK.md` section 2. The expected local
safe model path is:

```text
validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops
```

The evaluator independently checks the model's semantic metadata against the
preregistered protocol and requires the `.skops.json` sidecar.

## 3. Keep the completed v3 polarity artifact

The evaluator requires the corrected v3 artifact from Issue #14, for example:

```text
validation_reports/experiments/2026-09-polarity-thresholds/mitdb_extended_derivation_v3.json
```

It must verify as schema
`electrotrace.mitdb_polarity_thresholds_extended_derivation/v3` and recommend
exactly v2=0.0, width=0.0 for this frozen model generation.

## 4. Download the complete European ST-T Database

```python
import wfdb
wfdb.dl_database("edb", dl_dir=".cache/physionet/edb")
```

Human sanity check only:

```bash
wc -l .cache/physionet/edb/RECORDS
# expected: 90
```

The evaluator is stricter than the count. It requires the exact ordered
preregistered RECORDS list and every `.hea`, `.dat`, and `.atr` file.

## 5. Run the one-shot prospective evaluation

**Do not run this until the frozen model, v3 derivation artifact, and complete
EDB download are present.**

```bash
python -u scripts/evaluate_frozen_model_edb_prospective.py \
  --edb-dir .cache/physionet/edb \
  --model validation_reports/incart_mitbih_model_windowed_std_2026-09-09.skops \
  --polarity-threshold-report \
    validation_reports/experiments/2026-09-polarity-thresholds/mitdb_extended_derivation_v3.json \
  --confirm-first-scored-run
```

The output path is intentionally fixed and cannot be overridden:

```text
validation_reports/experiments/2026-09-edb-prospective/edb_frozen_v4_prospective_first_run.json
```

The runner refuses to overwrite an existing first-run artifact.

## 6. What to return

Return the untouched JSON file above for review.

Do not run alternate channels, change gates, turn on recovery, alter the model
threshold, or inspect an alternative configuration before the primary result
has been archived. Such experiments can be useful later, but they are
development analyses and must not replace the one-shot prospective result.
