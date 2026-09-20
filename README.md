# ElectroTrace

**A reproducible ECG/electrophysiology annotation and benchmarking workbench.**

ElectroTrace is built around a simple principle: **the evidence is the product**. It provides one place to run detectors, compare them under declared protocols, preserve record/subject-level statistics, and carry hashes, software versions, and provenance alongside results.

The repository also ships its own Stage-1 and two-stage Random Forest detector, but that detector is not the project's only or primary claim. The current evidence is deliberately mixed: the two-stage model performs strongly on its locked MIT-BIH split, while the same trained model shows a large cross-database drop on INCART. That gap is preserved and documented rather than hidden.

## Current evidence

| Protocol | Detector | Records | Sensitivity | PPV | F1 |
|---|---|---:|---:|---:|---:|
| Locked MIT-BIH held-out | ElectroTrace two-stage | 12 | 0.9924 | 0.9879 | 0.9902 |
| Locked MIT-BIH held-out | Pan-Tompkins reimplementation | 12 | 0.9908 | 0.9954 | 0.9931 |
| Locked MIT-BIH held-out | Hamilton reimplementation | 12 | 0.9990 | 0.9303 | 0.9634 |
| Locked MIT-BIH held-out | ElectroTrace Stage-1 | 12 | 0.9931 | 0.7553 | 0.8580 |
| INCART external comparison | ElectroTrace two-stage | 68 | 0.3239 | 0.9679 | 0.4854 |
| INCART certified WFDB comparison | WFDB gqrs | 68 | 0.9324 | 0.9265 | 0.9294 |

These values are transcribed from the repository's locked validation artifacts. They are retrospective research results, not clinical validation and not evidence of general detector superiority.

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

**PyPI publication is prepared by CI but not yet performed for this hardening branch.** For the current code, install from source:



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

The detector plugin interface is already present. The next benchmark phase will add external package adapters and a regenerated multi-database leaderboard rather than a single scalar ranking.

## Model artifacts

The two-stage Random Forest is opt-in. New model persistence uses .skops plus a metadata sidecar. Legacy pickle models are migration-only and rejected by default.

A model records its feature schema and training scikit-learn major version. Loading stops when the runtime major version is incompatible or when the .skops file contains unknown serialized types.

The current INCART result is therefore part of the evidence surface: cross-database behavior must be measured instead of inferred from MIT-BIH.

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

See docs/, mkdocs.yml, and CITATION.cff. A DOI-bearing software release still requires final release/Zenodo configuration.

## License

AGPL-3.0-or-later at present. Any future core/server license split will be an explicit project decision.
