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

## Completed local recovery

The delivered standalone notebook was executed locally with its isolated pinned environment and immutable source archive `f431bb60a9f4a6625f4071fa7c342af879a8e720` (SHA256 `42b27fa5e5e7e7a3f391eb3737fdba28dd61977646d0284013dbdbac7f8ec33b`). Only Colab interface calls and filesystem roots were substituted. The real runner, dependency installation, source/data verification, delineation, bootstrap and export ran. A fresh hosted Colab execution remains unobserved. The [local recovery artifacts](colab_ludb_2026_10_15/local_recovery/) are separate from the original failed upload.

All 200 patients / 2,400 leads completed. The publisher manifest and 2,802 required input files were verified. All 411 exported artifact hashes verify. There are 21,965 N markers: **21,843 eligible paired QRS references and 122 excluded references (0.56%)**, distributed across six patients. Exclusion reasons overlap: 122 have missing/nonadjacent onset and 19 also have missing/nonadjacent offset.

| Tolerance | Both boundaries successful / eligible | Estimate | Patient-bootstrap 95% percentile interval |
|---|---|---|---|
| 20 ms | 15,071 / 21,843 | 69.00% | 65.76–71.89% |
| 40 ms | 20,060 / 21,843 | 91.84% | 89.21–94.25% |
| 60 ms | 21,172 / 21,843 | 96.93% | 95.32–98.32% |
| 80 ms | 21,700 / 21,843 | 99.35% | 98.60–99.81% |
| 100 ms | 21,826 / 21,843 | 99.92% | 99.83–99.98% |

These are beat-weighted success fractions with patients as resampling units; they are not means of independent lead percentages. Per-lead 20 ms both-boundary success ranges from 56.91% to 83.23%. Recorded signed error means across all eligible cases are +8.21 ms onset and −7.07 ms offset (including one unfound-onset fallback estimate; these are not conditional-on-found means). This suggests duration underestimation in this exploratory cohort; a pooled offset correction would be posthoc tuning, not an independently validated improvement. The delineator and gates are therefore unchanged. Development should use a predeclared split/reference-adjudication plan and test any correction on an unused cohort. No universal accuracy or clinical acceptance threshold was declared, so completion is not an acceptance-pass claim.

A second full notebook run reused all 200 patient results without rewriting them and reproduced the summary byte for byte. Actual runner tests verified budget-based pause/partial continuation and rejection of a tampered patient result. The notebook's HTTP resume alignment and complete-part recovery were also exercised. The notebook was validated against notebook schema. Local software tests: 375 passed; hosted Python 3.10–3.13 tests, packaging and security passed for the parser repair.
