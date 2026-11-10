# Genotype-to-electrophysiology evidence contract

ElectroTrace should represent electrical phenotypes as measured observations linked
to a genetic context, not as deterministic consequences of a variant.

## Minimum linked entities

- Gene and variant: HGNC symbol, genome build, reference/alternate allele and
  transcript/protein consequence when available.
- Inheritance and evidence: zygosity, segregation, pathogenicity classification,
  classification source and date. Unknown fields remain explicitly unknown.
- Clinical context: phenotype, age/context where permitted, medications and
  acquisition context. Avoid storing direct identifiers in research exports.
- ECG acquisition: preserve the existing ECGCalibrationBundle provenance, source
  hash, sampling rate, units, lead order and device/filter metadata.
- Electrical phenotype: rhythm diagnosis, ectopy burden, exercise/catecholamine
  provocation context, QT/QTc methodology, ventricular arrhythmia events and
  adjudication status.
- Linkage: stable IDs to CardiAtlas product/target records and CardiSim phenotype
  profiles; every derived feature records algorithm/version and uncertainty.

## Initial cardiac-genetics profiles

CASQ2 and RYR2 are relevant to calcium-handling and CPVT phenotypes. TNNT2 is
relevant to contractile dysfunction and cardiomyopathy phenotypes. These are
prioritisation examples, not patient-level diagnostic rules.

## Safety and validity

- Do not infer genotype or pathogenicity from ECG morphology.
- Do not infer CPVT from an isolated ectopic beat or an uncontextualised exercise ECG.
- Preserve raw waveform and source annotation; derived labels must remain traceable.
- Distinguish automated detector output from expert-adjudicated rhythm labels.
- A valid measurement bundle is not a clinically validated diagnostic test.

## Acceptance criteria for future implementation

A genotype-linked phenotype record must include source provenance, assay/measurement
context, uncertainty, adjudication state and a clear distinction between observed,
derived and inferred fields. Missing clinical data must not be imputed as normal.
