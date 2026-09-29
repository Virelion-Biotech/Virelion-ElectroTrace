# QT Database QRS-boundary validation

## Purpose

The QT Database (PhysioNet v1.0.0) is used here for **waveform-boundary
characterization**, not as a fresh R-peak generalization cohort. QTDB contains
105 two-channel 15-minute ECG excerpts, many originating from other databases,
with manually marked waveform boundaries for selected beats.

The frozen study is defined in
`validation_protocols/qtdb_qrs_delineation_v1.json`.

## Manual references

The protocol uses only the manual second-pass annotations:

- `.q1c`: annotator 1; required for all 105 records and used for the primary
  and secondary analyses.
- `.q2c`: annotator 2; expected on exactly 11 records and used only for the
  confirmatory inter-observer analysis.

Automatic `.pu`, `.pu0`, and `.pu1` annotations are not used for model
selection or scoring.

WFDB waveform-onset `(` and waveform-end `)` annotations with `num=1`
identify QRS boundaries. The evaluator fails closed on unmatched, duplicate,
non-monotonic, negative, or out-of-support QRS boundary pairs.

## Prespecified tolerance grid

All boundary analyses use the complete prespecified grid:

`20, 40, 60, 80, 100 ms`

No single tolerance is selected after seeing the results.

Uncertainty is estimated by a 2,000-replicate percentile bootstrap with the
**record** as the resampling unit (seed 42). Beats are never treated as
independent bootstrap units.

## Primary endpoint: delineation-only tolerance curve

The primary endpoint isolates the boundary locator from R-peak detection.

For each manual `.q1c` QRS pair, the midpoint of the manual onset and offset
is supplied as the center to the frozen signal-only delineator
`qrs-edge-energy-v1`. The delineator does not receive the manual onset or
offset values.

At every tolerance, three success fractions are reported:

- onset within tolerance;
- offset within tolerance;
- both onset and offset within tolerance.

A boundary counts as successful only if the delineator actually reports that
the boundary was found and its absolute timing error is within the specified
tolerance. Fallback search-window edges do not count as successful boundaries.

This endpoint is intentionally **not an end-to-end detector score**. Manual
centers are supplied to remove event-detection misses and false positives from
the primary delineation question.

## Secondary endpoint: end-to-end signal-detector analysis

The secondary analysis runs the frozen Stage-1 signal detector on channel 0:

- detector: `electrotrace.validation_detectors:detect_r_peaks`;
- polarity: adaptive;
- Stage-1 scale: `windowed_std`;
- one-to-one event matching to manual QRS centers at 75 ms.

Event sensitivity and positive predictive value are reported separately from
boundary localization.

For each boundary tolerance and for onset, offset, and joint boundaries, the
report includes:

1. conditional boundary success among event-matched QRS complexes;
2. end-to-end boundary sensitivity relative to all manual QRS complexes;
3. end-to-end boundary positive predictive value relative to all detected
   events.

This separation prevents R-peak detection failures from being misreported as
pure delineation error.

## Confirmatory endpoint: inter-observer variability

For the 11 records with `.q2c`, annotator-2 QRS complexes are matched
one-to-one to annotator-1 QRS complexes using the same fixed 75 ms event-center
window. The report then applies the same 20/40/60/80/100 ms onset, offset, and
joint-boundary curve to `q2c - q1c` differences.

This is an observer-agreement reference, not an algorithm score.

## Frozen delineator

`qrs-edge-energy-v1` remains unchanged:

- search window: ±160 ms;
- band-pass: 5–25 Hz;
- derivative-envelope smoothing: 12 ms;
- normalized edge-energy threshold: 15%;
- normalized band-amplitude threshold: 10%;
- sustained inactive samples: 3.

QTDB manual annotations may not be used to tune these values during this
confirmatory study. A parameter change creates a new delineator version and
requires a separately declared development/validation sequence.

## Completeness and provenance

A completed primary artifact requires:

- the exact 105-record QTDB `RECORDS` cohort;
- 105 non-empty `.hea`, `.dat`, and `.q1c` inputs;
- exactly 11 available `.q2c` records;
- no excluded primary records;
- official WFDB Python parsing;
- SHA-256 for every input file and the protocol;
- executed Git `HEAD`;
- per-record results plus pooled ratio estimands with record-bootstrap
  intervals.

Any malformed required record aborts the run rather than silently dropping it.

## Interpretation

This study can characterize QRS boundary localization under the declared QTDB
protocol. It does **not** establish:

- clinical validation;
- regulatory compliance;
- population-wide generalization;
- fresh independent R-peak validation;
- superiority over established delineators.

QTDB includes excerpts from multiple previously existing source databases, so
its provenance must not be represented as a single institutionally independent
R-peak cohort.

## Source

PhysioNet QT Database v1.0.0:
https://physionet.org/content/qtdb/1.0.0/
