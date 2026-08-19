# QRS Onset/Offset Delineation

ElectroTrace now contains an experimental frozen signal-only QRS boundary delineator: `qrs-edge-energy-v1` in `src/electrotrace/qrs_delineation.py`.

The module deliberately separates **delineation** from **event detection**. It accepts a supplied R-peak-like center and searches a fixed ±160 ms window using a 5–25 Hz band-limited waveform plus a 12 ms smoothed absolute-derivative envelope. A boundary must remain below both a normalized edge-energy threshold (15%) and a normalized band-amplitude threshold (10%) for three samples.

This design is inspired by established QRS detectors that first isolate QRS-like energy/steepness and then localize fiducial points, rather than treating raw ECG amplitude alone as a boundary definition.

## Scientific status

This module is **experimental**. It has not been promoted into the primary MIT-BIH R-peak detector and has not been used to alter the locked MIT-BIH benchmark.

QT Database validation must be run with official WFDB semantics before any boundary-performance claim is treated as confirmatory. In particular, QTDB headers may contain sampling frequency plus counter-frequency syntax such as `250/360`; the first value is the sampling frequency and the second is the counter frequency. Segment/base-time and annotation semantics must be handled by WFDB rather than a simplified byte parser.

No QTDB manual annotations may be used to tune these parameters before a confirmatory run. Any parameter changes constitute a new delineator version and a new validation experiment.
