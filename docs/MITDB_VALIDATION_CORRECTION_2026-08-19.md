# MIT-BIH validation correction — 2026-08-19

The earlier `validation_reports/mitdb_rpeak_validation_local_reproduction_2026-08-19.json` report is **superseded**.

## Why

The uploaded MIT-BIH archive was decoded correctly for the record set, signal samples, and reference annotation counts, but the earlier report contained incorrect true-positive/false-negative matching for a subset of records. The detector candidate counts and reference counts were not the problem; the TP matching in the earlier local report was.

## Corrected locked baseline

Using the current ElectroTrace one-to-one matcher with an exact WFDB-compatible annotation decode over all 48 canonical MIT-BIH records, channel 0, adaptive polarity, and 75 ms tolerance:

- Sensitivity: **96.56%**
- PPV: **67.19%**
- F1: **79.24%**
- Macro-record sensitivity: **96.40%**
- Macro-record PPV: **68.95%**
- Macro-record F1: **79.14%**
- Record-bootstrap 95% CI for macro F1: **74.28–83.39%**

The exact reference count is 109,494 and the detector candidate count is 157,348.

## Guarded polarity v2 exploratory result

The guarded QRS-specific polarity selector changes only record 207 and gives:

- Sensitivity: **97.88%**
- PPV: **68.14%**
- F1: **80.34%**
- Macro-record F1: **80.56%**
- Record-bootstrap 95% CI for macro F1: **76.92–83.96%**

On record 207, F1 improves from **0.1266** to **0.8071**.

The v2 result remains **exploratory** and is not promoted to the primary detector until it is independently reproduced with the official `wfdb-python` runtime and then evaluated on an independent ECG dataset.

## Reproducibility note

The corrected benchmark is an official-style local reproduction from the uploaded MIT-BIH 1.0.0 archive. Network access was unavailable in the execution runtime, so a fresh `wfdb-python` installation could not be performed there. The annotation parser was independently aligned to the WFDB annotation-byte semantics and reproduces the canonical per-record reference counts, including an aggregate of 109,494 accepted beat annotations.

The uploaded archive SHA-256 is:

`47b26926927c11bd9174154d367c811afc1b186f650f8a205931ca8c520f0a87`

The detector source used for the corrected baseline is commit `615833eb1c13e9af531a9829fa32433f16fba40f`.
