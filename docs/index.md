---
layout: page
title: ElectroTrace
---

# ElectroTrace

ElectroTrace is a reproducible ECG/electrophysiology annotation, benchmarking, and research-validation toolkit.

## Start here

1. Install from PyPI with `python -m pip install electrotrace==1.9.0`.
2. Follow the [Quickstart](QUICKSTART.md) for a first detection run using the small example CSV.
3. Read [Validation evidence](validation/README.md) and [Limitations](LIMITATIONS.md) before interpreting comparative performance.
4. Use [Reproducibility](REPRODUCIBILITY.md) when preparing an experiment, manuscript artifact, or lab handoff.

## Core workflows

- Detect R peaks in CSV, EDF, and WFDB recordings.
- Batch recordings into tidy beat/record/subject tables.
- Validate detectors against local WFDB annotations.
- Benchmark multiple detectors under one declared matching protocol.
- Produce provenance manifests with source hashes and detector/software metadata.
- Render result JSON as a self-contained HTML report.

## Evidence-first design

ElectroTrace treats the evidence trail as part of the output. Results should travel with their protocol, source hashes, detector configuration, software version, and evidence status.

Historical locked results, development-data characterizations, and prospective external runs are deliberately kept distinct. See the validation pages for the current boundaries.

> ElectroTrace is research software, not a clinical device. No current result establishes clinical deployment, regulatory performance, real-time/streaming validity, population-wide generalization, or universal detector superiority.
