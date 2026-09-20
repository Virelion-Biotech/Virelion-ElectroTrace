# Quickstart

Install:

~~~bash
python -m pip install electrotrace
~~~

Detect one file:

~~~bash
electrotrace detect recording.edf --detector pan-tompkins -o peaks.csv
~~~

Batch a folder:

~~~bash
electrotrace batch data/ --detector pan-tompkins --workers 4 -o results/
~~~

Validate a local WFDB record:

~~~bash
electrotrace validate .cache/physionet/mitdb/100 --detector pan-tompkins -o validation.json
~~~

Benchmark local WFDB records:

~~~bash
electrotrace bench .cache/physionet/mitdb --detectors pan-tompkins,hamilton -o bench.json
~~~
