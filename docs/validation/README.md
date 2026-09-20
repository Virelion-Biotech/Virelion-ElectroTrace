# Validation evidence

## Locked MIT-BIH

The repository contains a locked 12-record two-stage result with sensitivity 0.9924, PPV 0.9879, and F1 0.9902, plus the same-split classical baseline comparison.

## INCART

The certified WFDB comparison reports ElectroTrace two-stage sensitivity 0.3239, PPV 0.9679, F1 0.4854 across 68 records; WFDB gqrs reports sensitivity 0.9324, PPV 0.9265, F1 0.9294 under the same 75 ms matching rule.

The appropriate interpretation is a measured cross-database generalization gap. Follow-up experiments should address sampling-rate normalization, amplitude scaling, threshold transfer, polarity errors, and independent databases before changing the model architecture.
