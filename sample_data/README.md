# Example data

`sample_ecg.csv` is a small repository fixture for installation checks, documentation examples, and first-run CLI demonstrations.

It is **not** part of the scientific validation evidence and must not be used to infer detector accuracy, clinical performance, or population generalization.

The file follows ElectroTrace's CSV convention:

- a monotonic `time` column in seconds;
- one or more numeric signal columns (`Lead_I`, `Lead_II`).

Example:

```bash
electrotrace detect sample_ecg.csv --detector pan-tompkins --channel 0 -o peaks.csv
```

For benchmark or validation work, use a declared dataset/protocol and retain the generated provenance artifacts.
