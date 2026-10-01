---
layout: page
title: Limitations
---

# Limitations

ElectroTrace is research software, not a clinical device.

The locked MIT-BIH evidence uses 12 held-out records and should not be interpreted as population-level generalization. The two-stage Random Forest has a large external performance gap on INCART and is not presented as a universally transferable detector. For unseen databases, follow the explicit research-use rules in `CROSS_DATABASE_POLICY.md`; the repository has no validated automatic out-of-domain detector or clinical fallback policy.

The current benchmark architecture is retrospective full-record evaluation. Streaming/real-time performance is not established.

Comparative numbers depend on database, annotation policy, tolerance, signal channel, and detector implementation.

Subject-level inference requires explicit subject metadata. The CLI does not silently infer subjects from filenames.
