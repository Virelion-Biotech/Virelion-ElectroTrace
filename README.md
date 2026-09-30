# ElectroTrace

**Current source version:** 1.9.0 (release candidate)

**A reproducible ECG/electrophysiology annotation and benchmarking workbench.**

ElectroTrace is built around a simple principle: **the evidence is the product**. It provides one place to run detectors, compare them under declared protocols, preserve record/subject-level statistics, and carry hashes, software versions, and provenance alongside results.

The repository also ships its own Stage-1 and two-stage Random Forest detector, but that detector is not the project's only or primary claim. The current evidence is deliberately mixed: the two-stage model performs strongly on its locked MIT-BIH split, and two successive model generations have both shown a cross-database drop on INCART. That gap is preserved and documented rather than hidden.

## Current evidence

**Two distinct model generations exist for the two-stage detector and must not be conflated** (see `validation_reports/VALIDATION_STATUS.md` for the full history). The original locked MIT-BIH row below is the `candidate-features-v3` model from 1.8.1. The current `candidate-features-v4` / windowed-std model now also has explicit MIT-BIH legacy non-regression audits, including the leakage-safe 0.0/0.0 polarity-gate result (F1 0.9644). Those audits do not become clean prospective evidence because MIT-BIH historically informed the adaptive-polarity mechanism.

| Protocol | Detector | Model generation | Records | Sensitivity | PPV | F1 |
|---|---|---|---:|---:|---:|---:|
| Locked MIT-BIH held-out | ElectroTrace two-stage | v3 (1.8.1 lock) | 12 | 0.9924 | 0.9879 | 0.9902 |
| MIT-BIH leakage-safe gate audit | ElectroTrace two-stage | v4 (legacy non-regression) | 12 | 0.9390 | 0.9913 | 0.9644 |
| Locked MIT-BIH held-out | Pan-Tompkins reimplementation | — | 12 | 0.9908 | 0.9954 | 0.9931 |
| Locked MIT-BIH held-out | Hamilton reimplementation | — | 12 | 0.9990 | 0.9303 | 0.9634 |
| Locked MIT-BIH held-out | ElectroTrace Stage-1 | — | 12 | 0.9931 | 0.7553 | 0.8580 |
| INCART external comparison | ElectroTrace two-stage | v3 | 68 | 0.3239 | 0.9679 | 0.4854 |
| INCART external comparison | ElectroTrace two-stage | v4 (windowed-std) | 68 | 0.9011 | 0.7594 | 0.8242 |
| INCART certified WFDB comparison | WFDB gqrs | — | 68 | 0.9324 | 0.9265 | 0.9294 |
| INCART complete source cohort | ElectroTrace two-stage | v4 frozen / development data | 75 | 0.8977 | 0.7633 | 0.8251 |
| INCART complete source cohort | WFDB gqrs | certified reference baseline | 75 | 0.9372 | 0.9310 | 0.9341 |
| INCART complete source cohort | WFDB sqrs | certified reference baseline | 75 | 0.7617 | 0.9537 | 0.8469 |
| European ST-T prospective external | ElectroTrace two-stage | v4 frozen | 90 | 0.9056 | 0.9644 | 0.9341 |

These values come from the repository's locked/archived validation artifacts. They are research results, not clinical validation and not evidence of universal detector superiority. **INCART informed v4 feature-scaling development and remains exposed development data**, so neither its historical 68-record row nor the completed 75-record row is a clean generalization estimate. The 75-record completion exists to finish the source cohort reproducibly, including the seven formerly malformed edge annotations under a two-certified-detector repair rule. A later one-shot prospective evaluation on all 90 European ST-T Database records provides independent external evidence for the unchanged frozen v4 configuration (F1 0.9341), but does not make INCART held-out or validate future post-hoc retuning. See `validation_reports/VALIDATION_STATUS.md` and `docs/CROSS_DATABASE_POLICY.md` for the evidence boundaries.

## Why ElectroTrace exists

Most ECG software answers "what detector should I use?" ElectroTrace is aimed at a different question:

> **Under one explicit protocol, how does this detector behave, and can someone else reproduce the answer?**

Primary statistics should treat records or subjects as the experimental unit rather than pretending individual beats are independent biological replicates. Validation artifacts can carry dataset hashes, detector configuration, software version, git commit, and protocol metadata.

### Primary use cases

**Preclinical electrophysiology:** batch animal recordings, retain explicit subject identifiers, and export record/subject-level tables for downstream statistics.

**Methods-heavy human ECG research:** evaluate an existing detector on held-out records and retain an auditable artifact for papers, reviews, and lab handoff.

**Detector development:** register a detector plugin and compare it against fixed baselines under the same matching rules.

ElectroTrace is research software. It is not a clinical device and the current models are not validated for clinical deployment.

## Install

Python 3.10+ is required.

**v1.9.0 is release-ready, but this repository does not claim PyPI availability until the tagged Trusted Publishing workflow succeeds.** Until then, install from source:

~~~bash
git clone https://github.com/Virelion-Biotech/Virelion-ElectroTrace.git
cd Virelion-ElectroTrace
python -m pip install -e ".[test,dev]"
~~~

## Five-minute start

~~~bash
electrotrace list
electrotrace detect recording.edf --detector pan-tompkins --channel 0 -o peaks.csv
electrotrace batch data/ --detector pan-tompkins --workers 4 -o results/
electrotrace report validation.json -o validation.html
~~~

Batch outputs:

~~~text
results/
├── beats.csv
├── records.csv
├── subjects.csv
├── failures.csv
├── manifest.json
├── batch_state.json
└── peaks/
~~~

Subject information is never inferred from filenames. Supply a two-column CSV with record and subject_id using --subject-map when subject aggregation is appropriate.

## Validation and benchmarking

~~~bash
electrotrace validate .cache/physionet/mitdb/100 --detector pan-tompkins -o validation.json
electrotrace bench .cache/physionet/mitdb --detectors pan-tompkins,hamilton --tolerance-ms 75 -o bench.json
~~~

Bench currently operates on local WFDB records. The repository does not vendor PhysioNet data.

The frozen experimental QRS boundary locator has also completed a 105-record QT
Database tolerance-curve study. Reference-centered joint onset/offset success
was 0.589 at 20 ms, 0.845 at 40 ms, 0.925 at 60 ms, 0.954 at 80 ms, and 0.994
at 100 ms, with record-level bootstrap uncertainty. This is delineation
characterization, not an end-to-end detector or clinical-performance claim;
see `docs/QTDB_VALIDATION.md` and `docs/QRS_DELINEATION.md`.

The detector plugin interface is already present. The next benchmark phase will add external package adapters and a regenerated multi-database leaderboard rather than a single scalar ranking.

## Model artifacts

The two-stage Random Forest is opt-in. New model persistence uses .skops plus a metadata sidecar. Legacy pickle models are migration-only and rejected by default.

A model records its feature schema and training scikit-learn major version. Loading stops when the runtime major version is incompatible or when the .skops file contains unknown serialized types.

The complete 75-record INCART result is part of the evidence surface: cross-database behavior must be measured instead of inferred from MIT-BIH. The research-use policy for unseen domains is documented in `docs/CROSS_DATABASE_POLICY.md`; ElectroTrace does not silently infer that a new database is in-domain or automatically promote the RF path over established reference baselines.

## Validation philosophy

- record-level rather than beat-level primary statistics;
- one-to-one matching under a declared tolerance;
- locked splits with explicit seeds;
- bootstrap confidence intervals over records;
- SHA-256 input hashes and software/git provenance;
- separate timing error, sensitivity, PPV, and F1;
- explicit limitations and non-claims.

## What is not claimed

ElectroTrace does not claim clinical performance, real-time streaming performance, population generalization, or superiority over established QRS detectors.

## Documentation and citation

See docs/, mkdocs.yml, and CITATION.cff. The repository includes Zenodo metadata for v1.9.0, but a DOI is not claimed until an actual tagged release is archived by Zenodo.

## License

AGPL-3.0-or-later at present. Any future core/server license split will be an explicit project decision.
