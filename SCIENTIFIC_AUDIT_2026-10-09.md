# Scientific audit changes — 2026-10-09

## Behavior

Add event-aligned annotator consensus/disagreement summaries and prespecified detector-tolerance sweeps. Correct QRS likelihood handoff from sigma_ms to sigma with an explicit noise_unit of ms.

## Scope and remaining evidence

Annotator IDs and event alignment must be supplied by the caller. Observed disagreement is not a confidence interval for physiological truth. The routine does not generate independent annotations or EAM registration-error propagation.

## Implementation

- `src/electrotrace/annotation_uncertainty.py`
- `tests/test_annotation_uncertainty.py`
- `src/electrotrace/calibration.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
