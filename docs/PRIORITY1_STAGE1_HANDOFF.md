# Priority 1 handoff — INCART Stage-1 candidate generation

## Context (read these on `main`)

| Artifact | Path |
|---|---|
| ElectroTrace INCART external | `validation_reports/incart_two_stage_external_full_2026-09-09.json` (+ `.csv`) |
| Certified gqrs/sqrs on INCART | `validation_reports/certified_wfdb_incart_2026-09-11.json` |
| Comparison summary | `validation_reports/incart_wfdb_vs_electrotrace_comparison_2026-09-11.json` |
| Domain-shift analysis script | `scripts/analyze_incart_domain_shift.py` |
| Stage-1 implementation | `src/electrotrace/validation_detectors.py` |
| INCART runner (how stage-1 is called) | `scripts/run_incart_external_full.py` |

Regenerate the per-record join table anytime:

```bash
python scripts/analyze_incart_domain_shift.py
```

## Aggregate scoreboard (68 records, 75 ms, channel 0)

| System | Sens | PPV | F1 |
|---|---:|---:|---:|
| gqrs | 0.932 | 0.926 | **0.929** |
| sqrs | 0.749 | 0.951 | 0.838 |
| ElectroTrace two-stage (locked MIT-BIH model) | 0.324 | 0.968 | **0.485** |

**Conclusion:** INCART is detectable (gqrs F1 ~0.93). ElectroTrace failure is **not** "impossible data".

## Failure-mode taxonomy (from published fields)

| Mode | n | Definition |
|---|---:|---|
| **candidate_generation_failure** | **22** | `stage1_candidates / reference_count < 0.5` |
| mixed_or_other | 19 | ET F1 < 0.75, neither clean mode |
| over_suppression | 10 | `stage1/ref >= 0.8` and `suppression_rate > 0.7` |
| acceptable | 17 | ET F1 ≥ 0.75 |

**28 records** have gqrs F1 ≥ 0.9 **and** ElectroTrace F1 < 0.5.

Among pure candidate-failure records, mean `cand_vs_ref` ≈ **0.13**.  
Among ET F1 ≥ 0.85 records: mean `cand_vs_ref` ≈ **1.36**, mean suppression ≈ **0.29**.

## Priority-1 focus records (stage-1 starvation; gqrs still excellent)

| Record | ET F1 | ET sens | stage1/ref | suppress | gqrs F1 |
|---|---:|---:|---:|---:|---:|
| I56 | 0.004 | 0.002 | **0.009** | 0.75 | 0.994 |
| I24 | 0.004 | 0.002 | **0.010** | 0.77 | 0.991 |
| I03 | 0.005 | 0.002 | **0.011** | 0.68 | 0.994 |
| I40 | 0.016 | 0.008 | **0.026** | 0.67 | 0.996 |
| I19 | 0.015 | 0.008 | **0.096** | 0.92 | 0.986 |
| I25 | 0.028 | 0.014 | **0.053** | 0.65 | 0.984 |
| I28 | 0.032 | 0.016 | **0.049** | 0.57 | 0.983 |
| I60 | 0.032 | 0.016 | **0.026** | 0.38 | 0.970 |
| I05 | 0.044 | 0.023 | **0.040** | 0.44 | 0.991 |
| I66 | 0.031 | 0.016 | **0.054** | 0.68 | 0.958 |

Contrast with **working** ET records (do not break these):

| Record | ET F1 | stage1/ref | suppress | gqrs F1 |
|---|---:|---:|---:|---:|
| I52 | 0.998 | 1.84 | 0.46 | 0.959 |
| I71 | 0.992 | 1.90 | 0.47 | 0.986 |
| I70 | 0.959 | 2.10 | 0.52 | 0.951 |
| I02 | 0.948 | 0.99 | 0.08 | 0.967 |

## Where the code lives

### Stage-1 core (`src/electrotrace/validation_detectors.py`)

- `_candidate_set(z, fs_hz, scale, prominence_fraction=0.5)`
  - `scipy.signal.find_peaks`
  - `distance = max(1, round(fs_hz * 0.25))` → at 257 Hz ≈ **64 samples** (~250 ms)
  - `prominence = scale * prominence_fraction` with **default 0.5**
  - `scale = std(signal - median(signal))`

- `detect_r_peaks(...)` — median-center, optional polarity flip, call `_candidate_set`

- `select_signal_polarity(...)` — count-ratio rule + low-confidence fallback to `polarity_v2`
  - On INCART, polarity is almost always **positive** (65/68); **not** the main split

- `recover_stage1_candidates(...)` — gap recovery with lower prominence (0.25)
  - INCART external run used **`recovery=False`** in `detect_r_peaks_two_stage`

### How INCART calls it

`scripts/run_incart_external_full.py` with polarity `"adaptive"`, **`recovery=False`**, locked MIT-BIH-trained `CandidateSuppressor` (do **not** retrain on INCART for this priority).

## Constraints (do not violate)

1. **No INCART labels for model selection / threshold tuning / architecture search** until stage-1 coverage is fixed and documented.
2. Do **not** change the RF suppressor architecture for Priority 1.
3. Keep MIT-BIH held-out performance from regressing; any stage-1 change must be re-checked on MIT-BIH.
4. Prefer parameter/front-end fixes (prominence schedule, scale estimator, refractory distance, optional recovery) over new ML.

## Suggested investigation procedure

1. Load INCART records I56, I03, I24 (257 Hz, channel 0).
2. Plot: raw lead, median-centered ±z, atr references, ElectroTrace stage-1 peaks, gqrs `.qrs` peaks.
3. Log: `std(z)`, peak counts at `prominence_fraction ∈ {0.5, 0.35, 0.25, 0.15}`, pos vs neg counts, polarity decision.
4. **Hypothesis 1 (primary):** `prominence = 0.5 * global std` is too harsh on low-amplitude / nonstationary INCART segments (global scale dominated by noise or artifacts → true R peaks fall below threshold).
5. **Hypothesis 2:** fixed 250 ms distance interacts badly with rate/morphology (less likely given cand/ref ~0.01).
6. **Hypothesis 3:** `recovery=True` only fills *gaps between existing peaks*; if stage-1 returns ~10 peaks on a 2000-beat record, recovery cannot help — need primary detection density first.

## Acceptance criteria for Priority 1

On the 10 focus records above, without using INCART labels to fit a model:

- Raise median `stage1_candidates / reference_count` from ~0.01–0.05 toward **≥ 0.8** (candidate-coverage proxy), **or** document why not possible without label leakage.
- Re-run full INCART external script with the same locked stage-2 model.
- Show aggregate INCART sensitivity/F1 movement and MIT-BIH held-out non-regression.
- Over-suppression records (I62, I73) are **out of scope** until stage-1 is fixed.

## Out of scope for Priority 1

- Retraining `CandidateSuppressor` on INCART
- Changing F1 threshold selection using INCART calibration
- New network architectures
- Multi-lead fusion (stay on channel 0 for comparability)
