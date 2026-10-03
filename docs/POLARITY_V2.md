# Experimental Polarity Selector v2

> **Historical experiment.** This document records the early v2 polarity work
> that informed later leakage analysis. It is not the current selector or gate
> specification. The repository subsequently completed development-only gate
> derivation, locked non-regression audits, and prospective external evaluation;
> use `validation_reports/VALIDATION_STATUS.md` for current evidence.


## Status

Experimental only. The locked primary detector and MIT-BIH primary benchmark are unchanged.

## Design

Polarity v2 uses a QRS-first strategy:

1. Band-pass the signal in a QRS-oriented 5–20 Hz band.
2. Form an absolute-gradient envelope to identify QRS-like steepness events.
3. Examine the signed raw waveform inside each QRS window.
4. Count positive, negative, and ambiguous signed events and combine that evidence with RR regularity.
5. The standalone selector can choose either polarity without reference annotations.

This design takes ideas from established open ECG tooling. NeuroKit documents a detector that finds QRS complexes from the steepness of the absolute ECG gradient and then identifies the R peak within the QRS region; it also exposes multiple established QRS detectors including Pan-Tompkins, Hamilton, Christov, and Elgendi. SleepECG provides another open Python ECG heartbeat-detection implementation.

## Guarded hybrid

The more promising variant is a conservative hybrid:

- retain the existing ElectroTrace count-based polarity when its confidence is at least 0.10;
- invoke QRS-specific v2 only when the existing polarity decision is genuinely ambiguous.

This is intentionally an experimental rule. The 0.10 gate was not used to redefine the primary MIT-BIH benchmark and must not be promoted based solely on MIT-BIH performance.

## Exploratory MIT-BIH result

On the uploaded MIT-BIH 1.0.0 archive, using an independent local WFDB-format reproduction and the same 48 records / 75 ms matching policy:

| Selector | Sensitivity | PPV | F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Locked primary | 94.25% | 65.58% | 77.34% | 77.19% |
| Standalone QRS v2 | 95.60% | 63.23% | 76.12% | 76.25% |
| Guarded hybrid v2 | **95.56%** | **66.53%** | **78.44%** | **78.61%** |

The hybrid changed only record 207 under the 0.10 ambiguity gate. Its record-level F1 improvement on 207 was approximately +0.68. This is encouraging but is not sufficient evidence for promotion.

## Historical decision rule (superseded)

At the time of this experiment, the rule was not to modify the primary detector without official reproduction and independent external evaluation. Those follow-up steps were later executed under stricter leakage-aware protocols. The original 0.10 exploratory gate in this document is therefore not a current release parameter and must not be revived by reading this file in isolation.
