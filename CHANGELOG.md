# Changelog

## 1.10.0 — 2026-10-06

### Added
- Calibration-grade ECG observation handoffs for downstream CardiEP/CardiInfer workflows, including aligned multi-lead median-beat templates, R-relative timing, QRS delineation summaries, robust residual-noise estimates, lead-quality metadata, source SHA-256 hashes, uncertainty/QC fields, typed artifact references, and likelihood hints.
- Registration of pre-aligned EAM activation, activation-map, and repolarization-map observations with explicit coordinate-frame and unit requirements; ElectroTrace does not invent spatial registration.
- `electrotrace calibration prepare` and HeartTwin `electrical.prepare_calibration` routing for the new calibration handoff contract.
- Regression coverage for release metadata, documentation onboarding, and calibration handoff behavior.

### Changed
- Public documentation is now centered on the released package and first-run sample while retaining explicit scientific-evidence boundaries.
- Release/citation metadata is synchronized at 1.10.0 and Zenodo metadata now uses the canonical `.zenodo.json` filename.
- Software citation authorship now credits Syed Umer Hannan with Virelion Biotech affiliation.
- The release workflow can bootstrap an explicitly named `release: vX.Y.Z` main-branch release commit, extract curated notes from this changelog, publish to PyPI with Trusted Publishing, and create or update the GitHub Release idempotently.

### Scientific boundaries
- This release adds software/integration capability; it does not retrain, retune, or upgrade the scientific evidence status of the frozen detector generations.
- Historical validation artifacts retain their original version labels and evidence classifications.
- Calibration handoffs are inverse-model input contracts, not evidence that downstream EP models, discrepancy functions, or inferred patient parameters are clinically valid.
- ElectroTrace remains research software: no clinical-device, streaming, population-generalization, regulatory, or universal detector-superiority claim is made.

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
