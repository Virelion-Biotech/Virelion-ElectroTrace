"""ElectroTrace: research-grade ECG and electrophysiology software."""

__version__ = "1.8.0"

from .provenance import DatasetManifest, manifest_from_dict
from .research_validation import build_validation_report, summarize_records_rigorous, write_validation_report

__all__ = [
    "__version__",
    "DatasetManifest",
    "manifest_from_dict",
    "build_validation_report",
    "summarize_records_rigorous",
    "write_validation_report",
]
