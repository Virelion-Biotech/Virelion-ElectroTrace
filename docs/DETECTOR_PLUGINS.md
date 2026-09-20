# Detector plugins

ElectroTrace can discover third-party detectors through the Python entry-point group electrotrace.detectors.

## Contract

```python
import numpy as np

class MyDetector:
    name = "my-detector"
    version = "1.0.0"
    citation = "DOI or full reference"

    def __call__(self, signal: np.ndarray, fs_hz: float, **kwargs) -> np.ndarray:
        return np.asarray([], dtype=int)
```

Register it in your package:

```toml
[project.entry-points."electrotrace.detectors"]
my-detector = "my_package.detector:MyDetector"
```

ElectroTrace records the detector name, declared version, source, and citation in benchmark output.

The plugin registry does not make third-party dependencies mandatory. An unavailable optional plugin is reported as a warning and does not prevent built-in detectors from loading.

## Built-in reference adapters

The current registry includes the local Pan-Tompkins and Hamilton research reimplementations, ElectroTrace Stage-1, optional ElectroTrace two-stage detection when a model is supplied, and the WFDB Python gqrs adapter.

WFDB's Python processing API exposes gqrs_detect for one-dimensional ECG signals and sampling rates.

gqrs is treated as a reference adapter, not as a detector-superiority claim. Its citation and provenance should remain visible in comparative output.