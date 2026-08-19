# QT Database Independent Waveform-Boundary Validation

## Purpose

The QT Database is the next independent scientific benchmark after MIT-BIH R-peak validation. It evaluates waveform boundaries rather than repeating the primary beat-detection task. The database contains 105 two-channel 15-minute ECG excerpts with expert manual waveform annotations, including QRS/P/T/U fiducial boundaries and peaks. This makes it appropriate for testing fiducialization and boundary extraction under diverse QRS and ST-T morphologies.

## Frozen protocol

Dataset: PhysioNet QT Database v1.0.0.

Primary analysis:

- complete QT Database record set;
- primary signal channel: channel 0;
- QRS boundary annotations from the final manual annotation files (`q1c`/`q2c` where available), with annotator handling declared before analysis;
- waveform onset `(` and waveform end `)` with annotation `num=1` for QRS;
- no use of automatic `pu`, `pu0`, `pu1`, or `ari` annotations for model selection;
- report onset and offset error separately;
- report signed error, absolute error, median, mean, SD, 95th percentile, and maximum;
- record-level uncertainty, never beat-level pseudoreplication;
- preserve all input-file SHA-256 values and the executed Git SHA in the manifest.

For the 11 records with two independent manual annotators, retain both annotator results and report inter-observer variability separately rather than averaging them into a single gold label. The database documentation explicitly provides these independent annotations for observer-variability analysis.

## Analysis rules

A QRS event is a pair of manual waveform-onset and waveform-end annotations with `num=1`. The adapter must reject malformed pairs, duplicate events, non-monotonic indices, and unmatched boundaries rather than silently repair them.

The detector under test should operate without access to the manual annotations. Manual boundaries are used only after detection for scoring.

The primary QRS boundary analysis should evaluate boundaries around detected QRS events matched one-to-one to manual QRS events under a predeclared event-matching tolerance. Boundary error must then be computed against the matched manual onset and offset.

No QT Database parameter may be tuned on the manual annotations before the locked primary analysis. Any algorithm refinement becomes a new version and is evaluated separately.

## Interpretation

This is an independent algorithmic validation, not clinical validation. Good boundary accuracy on the QT Database does not establish diagnostic or regulatory suitability.

## Source

PhysioNet QT Database v1.0.0: https://physionet.org/content/qtdb/1.0.0/
