# Changelog

## Unreleased — hardening toward 1.9.0

### Added
- Documented top-level Python API exports.
- electrotrace CLI for listing detectors, single-record detection, batch processing, local validation, local benchmarking, and JSON-to-HTML reporting.
- Detector protocol and third-party entry-point discovery.
- Record/beat/subject batch outputs and provenance manifests.
- Ruff, formatting, packaging, import, and CLI smoke checks in CI.
- skops model persistence with metadata and scikit-learn major-version checks.
- Contributor, citation, documentation, issue-template, PR-template, and Docker scaffolding.

### Changed
- Legacy pickle model loading is now explicit opt-in migration behavior.
- README positioning leads with reproducibility, benchmarking, and evidence.

### Not changed
- Scientific detector behavior remains frozen in this hardening cycle. Cross-database detector changes require a separate validation protocol.
