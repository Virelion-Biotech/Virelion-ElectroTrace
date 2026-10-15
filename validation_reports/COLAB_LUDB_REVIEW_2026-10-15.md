# LUDB Colab failure and recovery — 2026-10-15

The uploaded v1 notebook **failed**, after six patients, with `Unexpected LUDB annotation layout`. All 24 artifact hashes and 256 pinned source files match. Its original ZIP and status are retained in `colab_ludb_2026_10_15/`; it is not a completed LUDB validation result.

## Cause and safe correction

The original launcher assumed each lead annotation stream consisted entirely of `( p )`, `( N )`, `( t )` triples. Hash-verified LUDB 1.0.1 `data/7.v1` starts with `N, )`, has other missing QRS onsets, and includes `N, (, )` sequences. The failure is in the notebook adapter, not evidence that the frozen delineator itself is broken. Official data: https://physionet.org/content/ludb/1.0.1/ .

`scripts/validate_ludb.py` counts every N marker, extracts only complete adjacent ordered `( N )` references, and exports every excluded QRS index/sample/reason. It never invents missing boundaries or shifts triplet grouping after a defect. Nonadjacent, missing or reordered supports are excluded conservatively, not declared physiologically invalid. The estimand is explicitly limited to complete adjacent reference triplets; the reference coverage denominator and exclusions accompany the tolerance curves. Invalid array shapes, out-of-range/nonmonotonic samples and unknown symbols fail closed.

The existing delineator, manual-center construction, tolerance grid (20/40/60/80/100 ms), and patient-clustered bootstrap (2,000 draws, seed 1729) remain frozen. Each of the twelve leads stays nested within its patient. Missing predicted boundaries stay in the eligible denominator as failures. No performance thresholds are tuned from LUDB.

## What is established

The six complete uploaded patient results reproduce exactly. Patient 7 now completes and reports 18 excluded QRS references across its leads. The initial hash-verified subset audit covers 452 lead annotations: 4,017 QRS markers, 3,983 eligible paired references, 34 exclusions, and no parser errors. This subset is an engineering check, not the full 200-patient result or a representative performance estimate.

The recovery checks hashes of all required 2,802 dataset inputs against the pinned publisher manifest. Patient checkpoints bind source fingerprints, environment, input hashes and result hashes, and can continue on a new runtime. A full summary requires all 200 distinct patients. A fresh CPU Colab notebook will download the immutable repaired source and full public dataset, retain progress on Drive, and export a ZIP on completion, pause or failure.

## Remaining scientific gates

LUDB is available but is not fully fit for an unqualified all-QRS paired-boundary endpoint because some references are incomplete or ambiguous. Results from the recovery must disclose their reference coverage. Missing labels require expert adjudication; automated inference would contaminate the reference, so it is not done here.

Manual QRS centers make this oracle-assisted boundary characterization, not R-peak detection validation. Public-cohort exposure is unproven absent, so evidence is exploratory. P/T boundaries, device transport, lead-selector qualification, streaming performance and independent clinical qualification remain unevaluated. These gaps cannot be repaired by modifying a parser or relabeling a completed run.
