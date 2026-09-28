# Remaining validation work — do-it-yourself steps

## Already done in-repo

| Item | Status |
|------|--------|
| MIT-BIH held-out primary (1.8.1) | Locked |
| Provenance schema v7 | Locked |
| Pan–Tompkins / Hamilton protocol match | Locked |
| INCART external **pilot** (7 records) | `validation_reports/incart_external_pilot_2026-08-29.json` |

### INCART pilot takeaway (honest)

MIT-BIH-trained two-stage **does not transfer** well to this INCART subset
(sens collapses; PPV stays high). Hamilton generalizes best on this pilot.
Treat as evidence of **domain shift**, not a finished external study.

| Detector | Sens | PPV | F1 |
|----------|------|-----|-----|
| Hamilton | 0.975 | 0.912 | **0.943** |
| Pan–Tompkins | 0.817 | 0.983 | 0.892 |
| ElectroTrace Stage-1 | 0.638 | 0.883 | 0.741 |
| Two-stage (MIT-BIH model) | 0.333 | 0.990 | **0.499** |

---

## 1. Git tag `v1.8.1`

```bash
cd /path/to/Virelion-ElectroTrace
git fetch origin && git checkout main && git pull
git tag -a v1.8.1 -m "1.8.1: hybrid polarity, F1 threshold, locked validation + baselines"
git push origin v1.8.1
gh release create v1.8.1 --title "1.8.1" --notes-file validation_reports/VALIDATION_STATUS.md
```

---

## 2. Full INCART external study (75 records)

```bash
mkdir -p .cache/physionet/incartdb
python -c "import wfdb; wfdb.dl_database('incartdb', '.cache/physionet/incartdb')"
# Evaluate MIT-BIH-trained two-stage + baselines on all usable records
# Skip records with negative annotation sample indices
# Write validation_reports/incart_two_stage_external_full.json
```

Report per-record table, macro-record F1, failure analysis.

---

## 3. Certified WFDB baselines (`gqrs` / `sqrs`)

```bash
sudo apt-get install wfdb   # or build MIT-LCP wfdb-app-kit
which gqrs sqrs
gqrs -r .cache/physionet/mitdb/100
# Adapter: read .gqrs annotations → sample indices → same validate_record protocol
```

Emit locked JSON next to `mitdb_baseline_comparison_locked.json`.

---

## 4. Full QTDB delineation protocol

Prespecify: QRS onset/offset; tolerance curve {20,40,60,80,100} ms;
record-level bootstrap; optional `ecgpuwave` baseline.

Extend `scripts/validate_qtdb.py` → `validation_reports/qtdb_delineation_tolerance_curves.json`.

---

## 5. Prospective EDB external validation — completed

The one-shot European ST-T Database evaluation was preregistered in Issue #26
and completed on all 90 records on 2026-09-28. Frozen aggregate result:
sensitivity 0.9056, PPV 0.9644, F1 0.9341. EDB is now exposed and may only be
used for post-hoc development/diagnostics, not as prospective evidence for a
changed model.

## 6. Next untouched external database

Before making any EDB-motivated detector change, select and preregister a
different database that was not used in the current detector's development.
Do not use QTDB blindly as a prospective benchmark because it contains records
drawn from other databases, including historically exposed sources. Verify
source independence, annotation suitability, cohort definition, channel choice,
and scoring protocol before opening the next one-shot validation.

---

## Suggested paper wording

> On the locked MIT-BIH held-out split, ElectroTrace two-stage was competitive
> with a research Pan–Tompkins reimplementation (F1 0.990 vs 0.993). On an
> independent INCART pilot, a MIT-BIH-trained suppressor did not transfer
> (F1 0.50), indicating domain shift and motivating domain-specific calibration
> or classical detectors for cross-database use.
