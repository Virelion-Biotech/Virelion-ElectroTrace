# Validation closeout — historical checklist

> This file previously listed do-it-yourself validation steps. Those repository
> completion tasks are now finished. It is retained as a closeout index, not a
> current backlog.

## Completion status

| Item | Status | Primary evidence |
|---|---|---|
| MIT-BIH locked primary/model comparison | Complete | `validation_reports/mitdb_two_stage_locked_1.8.1.json` |
| Certified WFDB baselines on locked MIT-BIH | Complete | `validation_reports/experiments/2026-09-mitdb-certified-wfdb/ARCHIVED_RESULT.json` |
| Leakage-safe polarity-gate audit | Complete | `validation_reports/experiments/2026-09-polarity-width/` |
| Complete INCART source cohort | **75/75 complete** | `validation_reports/experiments/2026-09-incart-complete/ARCHIVED_RESULT.json` |
| Certified INCART reference-integrity audit | Complete | `validation_reports/experiments/2026-09-incart-reference-repair/ARCHIVED_RESULT.json` |
| QTDB delineation tolerance curve | **105/105 complete** | `docs/QTDB_VALIDATION.md` + archived workflow provenance |
| Prospective EDB external evaluation | **90/90 complete** | `validation_reports/VALIDATION_STATUS.md` |
| Prospective selector v1 / v2 / v3 transfer studies | Complete | LTAFDB / SVDB / Zymed LTSTDB sections in `VALIDATION_STATUS.md` |
| CI/security/release automation | Complete in repo | `.github/workflows/` |
| 1.9.0 package/citation/release metadata | Release candidate prepared | `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md` |

## Scientific closeout

The repository has no remaining P0/P1 validation issue required to support the
current bounded research claims. Historical results retain their original
evidence labels:

- MIT-BIH adaptive-polarity results are legacy non-regression where historical
  mechanism exposure applies.
- INCART is exposed v4 development data even though all 75 source records are
  now characterized.
- EDB, LTAFDB, SVDB, and the Zymed LTSTDB subset become exposed after their
  respective prospective first runs.
- QTDB is delineation characterization, not a fresh institutionally independent
  R-peak generalization cohort.
- No result supports a clinical-device, streaming, population-generalization,
  regulatory, or universal detector-superiority claim.

## Future research is not completion debt

A changed detector, threshold, polarity rule, or lead selector should be treated
as a new research generation: preregister it and evaluate it on another
untouched cohort. That is future scientific work, not an unfinished requirement
for ElectroTrace 1.9.0.

For current numbers, hashes, workflow IDs, and interpretation boundaries, use
`validation_reports/VALIDATION_STATUS.md`.
