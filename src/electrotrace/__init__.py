"""ElectroTrace: reproducible ECG/electrophysiology research and benchmarking tools."""

__version__ = "1.8.1"

from .candidate_suppressor import CandidateSuppressor
from .io import load_recording
from .provenance import DatasetManifest, manifest_from_dict
from .qrs_delineation import delineate_qrs
from .research_validation import build_validation_report, summarize_records_rigorous, write_validation_report
from .signal import apply_pipeline
from .validation import validate_record
from .validation_detectors import detect_r_peaks, detect_r_peaks_two_stage

__all__ = [
    "__version__",
    "CandidateSuppressor",
    "DatasetManifest",
    "manifest_from_dict",
    "load_recording",
    "apply_pipeline",
    "detect_r_peaks",
    "detect_r_peaks_two_stage",
    "delineate_qrs",
    "validate_record",
    "build_validation_report",
    "summarize_records_rigorous",
    "write_validation_report",
]
