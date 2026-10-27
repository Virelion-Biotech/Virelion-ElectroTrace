# ECGCalibrationBundle v2

**Schema:** `electrotrace-ecg-measurement-v2`  
**Default retention:** every complete beat window is retained; `max_beats` is an optional explicit cap. When set, omitted beats are listed in provenance.  
**Machine-readable schema:** [`schemas/ecg-measurement-bundle-v2.schema.json`](../schemas/ecg-measurement-bundle-v2.schema.json)

## Purpose

The `ECGCalibrationBundle` ECG measurement artifact is now designed to preserve the observed recording and the beat distribution, not only a pointwise median beat. The existing `build_ecg_calibration_bundle` remains available as the legacy `electrotrace-ep-calibration-v1` builder. The `prepare_ecg_calibration` path emits the v2 artifact and retains flat compatibility fields (including `beat_template`, `lead_names`, `qrs`, and `rr`) for existing handoff consumers.

The builder performs **no filtering, resampling, normalization, or voltage conversion**. The source-file SHA-256 identifies the original file bytes; the raw numeric arrays preserve the values decoded by the current loader. A source hash does not prove that the device labels or declared voltage units are correct, so unit provenance is stored separately.

## Contents

- **Acquisition:** reported and time-axis-inferred sampling frequency, disagreement check, duration, original lead order, requested order, expected/missing leads, per-lead voltage units and unit provenance, device/filter metadata, patient/encounter linkage, and primary lead.
- **Raw measurements:** source reference and SHA-256, source format/name, the source time axis, and the original decoded samples for every observed lead.
- **Beat analysis:** all detected R-peak sample indices/timestamps; complete selected beat windows; original-unit waveform arrays for each retained beat; supplied ectopic/paced annotations; annotation origin; RR intervals; and a descriptive regular/variable-RR screen.
- **Morphology:** per-beat QRS onset/offset and duration when the delineator returns boundaries; median and robust spread summaries; pointwise-median template; sample-wise beat-to-beat variability; and exploratory P/ST/T window proxies where those windows are estimable.
- **Measurement uncertainty:** per-lead baseline high-frequency difference-scale proxy, per-lead template residual scale, and beat-to-beat variability. QRS segmentation error remains explicitly **uncalibrated** until compared with independent manual annotations.
- **Provenance/QC:** algorithm and detector versions/settings, preprocessing declaration, source hash, omitted or truncated beats, unit origins, flags, and a quality status of `provisional_pass`, `qualified`, or `rejected`.

## Example

```python
from electrotrace import prepare_ecg_calibration

handoff = prepare_ecg_calibration(
    "record.csv",
    entity_id="case-identifier",
    output_dir="calibration-artifacts",
    params={
        "patient_id": "patient-identifier",
        "encounter_id": "encounter-identifier",
        "primary_channel": "II",
        "units": "mV",  # only declare this if it matches the original data
        "expected_leads": ["I", "II", "III", "aVR", "aVL", "aVF",
                           "V1", "V2", "V3", "V4", "V5", "V6"],
        "device_metadata": {"manufacturer": "known-device", "model": "known-model"},
        "filter_metadata": {"low_pass_hz": 150, "high_pass_hz": 0.05},
        # Optional source adjudications keyed by R-peak sample index:
        "beat_annotations": {
            1200: {"ectopic": True, "paced": False,
                   "source": "manual-adjudication", "adjudicated": True}
        },
    },
)
print(handoff["summary"])
```

The caller-supplied device/filter fields are recorded as declarations; ElectroTrace does not independently verify them. If patient or encounter linkage is unavailable, pass neither value rather than fabricating one. The resulting artifact will flag the missing linkage.

## Template selection and rhythm handling

All selected beat waveforms remain in `beat_analysis.beats`, even when the template excludes a beat. The template candidate set excludes beats explicitly annotated as ectopic or paced and beats flagged by a simple RR-outlier screen. The threshold is configurable. This screen is only a conservative heuristic: it is **not** an ectopy detector or rhythm diagnosis.

When too few stable beats remain, the bundle may retain a descriptive median of available beats, but marks it `descriptive_only_not_recommended_for_fit` and suppresses the ECG morphology likelihood hint. A valid-looking template is not evidence that multiple rhythms or conduction states can safely share one template.

## Uncertainty estimates and independent adjudication

The baseline noise and template residual estimates are scales, not calibrated error probabilities. Template residuals can include real beat-to-beat physiological differences; they must not automatically be interpreted as electronic noise or measurement error. The current QRS delineator does not produce a validated uncertainty interval.

`fit_uncertainty_calibration_map` fits a monotonic empirical scale-to-absolute-error map on a dedicated adjudicated calibration split. `apply_uncertainty_calibration_map` applies the frozen map. `evaluate_uncertainty_calibration` then accepts paired predicted uncertainty scales and absolute measurement errors from a separate independently adjudicated validation/test subset. All require an explicit measurement unit, adjudication source, subject and encounter identifiers, and `independent_reference=True`; at least three distinct subjects and encounters are required. It reports an equal-count empirical scale-to-error curve, Spearman association, and a binned median-scale/median-error difference. It rejects training-set references and caller-declared non-independent references. Independence and patient/encounter separation still require an auditable protocol outside this utility. The metrics are empirical calibration diagnostics—not probabilities of correctness.

Example:

```python
from electrotrace import evaluate_uncertainty_calibration

report = evaluate_uncertainty_calibration(
    predicted_uncertainty=[...],
    adjudicated_absolute_error=[...],
    adjudication_source="frozen-manual-review-v1",
    measurement_unit="ms",
    subject_ids=[...],      # one ID per measurement; independent split required
    encounter_ids=[...],    # one ID per measurement
    independent_reference=True,
    reference_split="independent_test",
)
```

Use one frozen subset for fitting any uncertainty-to-error mapping and a separate untouched subset to evaluate it. Prespecify event matching and tolerance windows, include multiple annotators or an adjudication protocol, and report performance by lead, noise level, rhythm type, and relevant demographic/acquisition strata where available.

## Acceptance gate status

The bundle records `acceptance_gate.status = "not_demonstrated"` by design. This implementation supplies the measurement data model, QC instrumentation, and acceptance-test scaffolding; it does **not** claim the requested frozen-set acceptance gate has passed. Before promotion beyond research use, execute a frozen corpus with independent annotations and confirm:

1. Lead labels/order, units, timebase, sample count, and source hash against source headers and known reference fixtures.
2. R-peak and QRS measurements against independent annotations with prespecified tolerances and error distributions.
3. Byte-reproducible artifacts from identical inputs and deterministic settings.
4. Correct qualification/rejection of noisy, irregular, ectopic/paced, missing-lead, truncated, and insufficient-beat recordings.
5. Uncertainty-scale correspondence to adjudicated errors on a held-out, patient/encounter-separated subset.

The existing repository’s rhythm/event and delineation validation sets may be useful starting points, but they must be checked for label suitability and independence before being called the frozen v2 acceptance set. A software test passing is not a clinical or patient-specific validation result.
