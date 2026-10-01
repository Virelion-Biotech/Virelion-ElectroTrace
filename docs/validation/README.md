---
layout: page
title: Validation evidence
---

# Validation evidence

This page is a compact orientation to the current evidence surface. The authoritative record of exact hashes, workflow IDs, model generations, and interpretation boundaries is `validation_reports/VALIDATION_STATUS.md` in the repository.

## MIT-BIH

Two model generations and multiple evidence statuses are preserved rather than conflated.

The historical locked 12-record two-stage v3 comparison reports sensitivity 0.9924, PPV 0.9879, and F1 0.9902.

The later leakage-safe frozen-v4 gate audit reports sensitivity 0.9390, PPV 0.9913, and F1 0.9644. Because the adaptive-polarity mechanism was historically informed by MIT-BIH behavior, this is labeled legacy non-regression rather than fresh prospective validation.

## INCART

INCART informed v4 development and is therefore exposed development data.

The completed 75-record source-cohort characterization reports frozen-v4 sensitivity 0.8977, PPV 0.7633, and F1 0.8251.

On the same 75 records, certified WFDB `gqrs` reports sensitivity 0.9372, PPV 0.9310, and F1 0.9341; `sqrs` reports sensitivity 0.7617, PPV 0.9537, and F1 0.8469.

These results characterize cross-database behavior. They are not a clean external generalization estimate for a detector developed with INCART information.

## European ST-T Database

A preregistered one-shot prospective evaluation of the unchanged frozen-v4 configuration completed all 90 records.

Aggregate sensitivity was 0.9056, PPV 0.9644, and F1 0.9341.

Performance was heterogeneous, including a small severe low-performing tail. The first-run outcomes were retained rather than tuned away. EDB became exposed after that run.

## Lead-selector transfer

The prospective selector-v2 evaluation on all 78 canonical SVDB records reported sensitivity 0.9680, PPV 0.9878, and F1 0.9778. Ten of 12 channel-1 switches improved record F1, while two regressed.

Selector v3 was subsequently evaluated prospectively on a preregistered 18-record Zymed LTSTDB subset. It reported sensitivity 0.8718, PPV 0.9650, and F1 0.9160, but selected channel 0 on all 18 records. The starvation-rescue condition therefore never fired; the run does not demonstrate a prospective rescue benefit.

## QT Database

The frozen QRS-boundary locator completed the 105-record QTDB tolerance-curve study.

Reference-centered joint onset/offset success was approximately 0.589 at 20 ms, 0.845 at 40 ms, 0.925 at 60 ms, 0.954 at 80 ms, and 0.994 at 100 ms.

This is QRS-boundary characterization, not a conventional full-beat detector benchmark or clinical validation.

## Evidence boundaries

Current results do **not** establish:

- clinical-device performance;
- regulatory performance;
- real-time or streaming validity;
- population-wide generalization;
- universal superiority over established detectors.

Any detector, threshold, polarity-rule, or selector change motivated by an exposed dataset creates a new research generation and requires appropriately untouched evidence before a new prospective claim.
