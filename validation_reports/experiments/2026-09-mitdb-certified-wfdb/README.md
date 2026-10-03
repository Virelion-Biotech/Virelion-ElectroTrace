# Certified WFDB locked MIT-BIH benchmark — 2026-09-29

This directory archives the completed certified/reference-binary comparison
requested by Issue #12.

Protocol: the historical locked 12-record MIT-BIH evaluation split
(`105 118 122 201 207 209 214 219 230 231 232 234`), channel 0, ElectroTrace
beat-symbol whitelist, 75 ms one-to-one matching. WFDB application source was
pinned to release **10.7.0**. No detector parameter was tuned on the locked
records.

| Detector | Sensitivity | PPV | F1 |
|---|---:|---:|---:|
| certified WFDB `gqrs`, default threshold 1.00 | 0.997066 | 0.982581 | 0.989771 |
| certified WFDB `sqrs`, default threshold 500 | 0.196992 | 0.197645 | 0.197318 |

The poor default-`sqrs` result is retained as observed rather than post-hoc
tuned. These are retrospective reference-binary comparisons, not clinical
validation or prospective evidence for ElectroTrace mechanisms.

Immutable source artifact:
- Actions run: `36605114898`
- head SHA: `8f960ec41cf80dce3ef8862198ccefacc16f10d1`
- artifact ID: `11050866718`
- artifact ZIP digest: `sha256:bca937fb2983407fa3f3e970cdb687dbdcdc27bfb83ae1c44254527a0ca541aa`
- full result JSON SHA-256: `57a922bfea41a8a5f799d739105d62d46dd91d31a3ca0f63ae0f0986c0da838e`

`ARCHIVED_RESULT.json` contains the compact permanent protocol/result record.
The Actions artifact contains the full per-record result JSON.
