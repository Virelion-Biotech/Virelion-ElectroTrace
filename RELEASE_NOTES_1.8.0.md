# ElectroTrace 1.8.0

## Validation (2026-08-29)

See `validation_reports/VALIDATION_STATUS.md`.

| Gate | Result |
|------|--------|
| Unit tests | 101 passed |
| MIT-BIH official (adaptive, bootstrap 2000) | sens 0.966 / PPV 0.672 / F1 0.792 |
| Two-stage MIT-BIH held-out | sens 0.938 / **PPV 0.993** / F1 0.965 |
| QTDB boundary (105/105) | matched 92.5%; onset median abs 8 ms |

## Changes

- **Default polarity** is `adaptive` for `detect_r_peaks` and `detect_r_peaks_two_stage`.
- **HTTP API** `/api/detect/r-peaks` uses the validated adaptive detector (`polarity` queryable).
- **Deployment guidance:** prefer two-stage FP suppression when PPV matters.
- **Hard case MIT-BIH 207:** adaptive count rule selects positive on this inverted lead; pass `polarity="negative"` for sens ≈ 0.92. Automatic morphology-score overrides were evaluated and **rejected** because they regress pooled MIT-BIH sensitivity (0.966 → ~0.83–0.92).
- Checkpointed QTDB harness: `scripts/validate_qtdb_resumable.py`.
- Optional CI workflow `validation-smoke.yml` (manual / weekly).
- Docs link to `VALIDATION_STATUS.md`.

## Recovery

Long-gap recovery remains **off by default**. Use `--recovery` on `benchmark_two_stage_mitdb.py` for experimental runs; treat results as exploratory until a locked recovery report is published.
