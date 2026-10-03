# Changelog

## 1.9.0 — 2026-10-01

### Added
- Public Python API exports plus the `electrotrace` CLI for detector listing, single-record detection, batch processing, local validation, benchmarking, and JSON-to-HTML reporting.
- Detector plugin discovery, subject-aware batch outputs, provenance manifests, restartable batch state, and publication-oriented record/subject tables.
- Safe `.skops` model persistence with metadata and scikit-learn major-version compatibility checks; legacy pickle loading remains explicit migration-only behavior.
- Certified WFDB `gqrs`/`sqrs` reference-binary benchmarks with pinned WFDB 10.7.0 source provenance.
- Leakage-safe polarity-gate derivation/audits and frozen-v4 non-regression tooling.
- One-shot prospective European ST-T Database evaluation, followed by preregistered prospective lead-selector evaluations on LTAFDB, SVDB, and the Zymed LTSTDB subset.
- QTDB 105-record QRS-boundary tolerance-curve characterization with record-bootstrap uncertainty and interobserver analysis.
- Complete 75-record INCART source-cohort characterization with strict two-certified-detector authorization for malformed edge-only source annotations.
- Permanent archived validation summaries under `validation_reports/experiments/` with workflow IDs, artifact IDs, hashes, and evidence-status boundaries.
- PR scientific-validation gates, transient PhysioNet retry handling, security auditing, packaging smoke checks, citation metadata, Zenodo metadata, Docker scaffolding, contributor templates, and release automation.

### Changed
- README and validation documentation now distinguish historical locked evidence, exposed development data, and genuinely prospective external runs.
- Cross-database policy explicitly forbids silently treating unseen domains as in-domain or relabeling development-data improvements as external validation.
- Standard CI supports Python 3.10–3.13.
- Release tags are checked against the package version before publication; successful PyPI publication is followed by an automatic GitHub Release.

### Scientific boundaries
- The core frozen v4 Random Forest detector is not retrained or post-hoc retuned by this release preparation.
- INCART remains exposed v4 development data even though the full 75-record source cohort is now characterized.
- Historical MIT-BIH adaptive-polarity results remain legacy non-regression evidence because the mechanism was historically informed by MIT-BIH behavior.
- ElectroTrace remains research software: no clinical-device, streaming, population-generalization, or universal detector-superiority claim is made.

## 1.8.1 — historical scientific freeze

The locked 1.8.1 artifacts remain immutable provenance records for the original
MIT-BIH two-stage benchmark and associated baselines. Later validation and
hardening work does not rewrite those historical JSON artifacts.
