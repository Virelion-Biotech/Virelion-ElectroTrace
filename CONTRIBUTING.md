# Contributing to ElectroTrace

Contributions are welcome, especially detector adapters, validation protocols, reproducibility fixes, and documentation.

## Scientific changes

Any change affecting detector outputs, matching, sampling-rate handling, thresholds, feature extraction, or experimental-unit aggregation must include a regression test and state whether historical validation artifacts remain comparable.

Do not overwrite a locked result to make a number look better. Add a new dated artifact and describe the protocol difference.

## Detector plugins

Third-party packages can register a callable under the electrotrace.detectors entry-point group. Adapters should declare a name, version, citation, and a callable accepting (signal, fs_hz) and returning sorted sample indices.

## Local checks

~~~bash
pip install -e ".[test,dev]"
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
python -m build
~~~
