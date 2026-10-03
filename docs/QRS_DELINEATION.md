---
layout: page
title: QRS delineation
---

# QRS Onset/Offset Delineation

ElectroTrace contains the experimental frozen signal-only QRS boundary
delineator `qrs-edge-energy-v1` in
`src/electrotrace/qrs_delineation.py`.

The module deliberately separates **delineation** from **event detection**. It
accepts a supplied R-peak-like center and searches a fixed ±160 ms window using
a 5–25 Hz band-limited waveform plus a 12 ms smoothed absolute-derivative
envelope. A boundary must remain below both a normalized edge-energy threshold
(15%) and a normalized band-amplitude threshold (10%) for three samples.

## Completed QTDB characterization

The frozen v1 delineator completed the prespecified QT Database v1.0.0 study on
September 29, 2026: **105/105 records**, no primary exclusions, 3,623 q1c
manual QRS boundary pairs, and a full 20/40/60/80/100 ms record-bootstrap
tolerance curve.

The primary analysis isolates delineation from R-peak detection by supplying
the midpoint of each manual QRS onset/offset pair as the center. Joint onset
and offset success was:

| Tolerance | Both boundaries within tolerance |
|---:|---:|
| 20 ms | 0.5893 |
| 40 ms | 0.8454 |
| 60 ms | 0.9249 |
| 80 ms | 0.9542 |
| 100 ms | 0.9945 |

The 95% record-bootstrap interval for the joint 60-ms endpoint was
0.8762–0.9683. The 11-record q1c/q2c inter-observer analysis reached joint
agreement 0.7698 at 20 ms and 0.9629 at 40 ms, illustrating that a very tight
boundary tolerance is partly constrained by observer variability.

The complete immutable result is Actions run `36592730416`, artifact
`11044358209`, JSON SHA-256
`e7c827b2df1cf808c9ccf717dcc54153d1bd425898e4c75576f4e38326cadb94`.
The frozen protocol is
`validation_protocols/qtdb_qrs_delineation_v1.json`.

## Scope

The delineator remains **experimental**. It has not been promoted into the
primary MIT-BIH R-peak detector and the QTDB result did not alter the locked
MIT-BIH detector.

The secondary QTDB analysis also demonstrates why event detection and boundary
localization must not be conflated: the raw Stage-1 detector matched 3,591 of
3,623 selected manual QRS events (sensitivity 0.9912) but produced 172,505
detections (PPV 0.0208) because QTDB manually delineates selected beats rather
than providing a conventional every-beat detection reference for this purpose.

No QTDB manual annotation was used to tune the frozen delineator parameters.
Any future parameter change constitutes a new delineator version and requires a
new declared development/validation sequence.

This is algorithmic waveform-boundary characterization, not clinical or
regulatory validation.
