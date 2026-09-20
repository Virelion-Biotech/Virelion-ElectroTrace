"""Detector interface and plugin discovery for ElectroTrace benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable, Protocol
import warnings

import numpy as np

from .baseline_detectors import hamilton_r_peaks, pan_tompkins_r_peaks
from .validation_detectors import detect_r_peaks, detect_r_peaks_two_stage


class Detector(Protocol):
    name: str
    version: str
    citation: str

    def __call__(self, signal: np.ndarray, fs_hz: float, **kwargs) -> np.ndarray: ...


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    version: str
    citation: str
    detector: Callable[[np.ndarray, float], np.ndarray]
    source: str = "builtin"


BUILTIN_CITATIONS = {
    "pan-tompkins": "Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm. IEEE T-BME. 1985;32(3):230-236.",
    "hamilton": "Hamilton PS, Tompkins WJ. Quantitative investigation of QRS detection rules. IEEE T-BME. 1986;33(12):1157-1165.",
    "stage1": "ElectroTrace Stage-1 heuristic candidate detector; see repository validation documentation.",
    "electrotrace-two-stage": "ElectroTrace two-stage Random Forest suppressor; see repository model and validation documentation.",
}


def _builtin_specs(model=None, polarity: str = "adaptive", scale_method: str = "windowed_std") -> dict[str, DetectorSpec]:
    specs: dict[str, DetectorSpec] = {
        "pan-tompkins": DetectorSpec(
            "pan-tompkins",
            "electrotrace-compatible",
            BUILTIN_CITATIONS["pan-tompkins"],
            pan_tompkins_r_peaks,
        ),
        "hamilton": DetectorSpec(
            "hamilton",
            "electrotrace-compatible",
            BUILTIN_CITATIONS["hamilton"],
            hamilton_r_peaks,
        ),
        "stage1": DetectorSpec(
            "stage1",
            "electrotrace-compatible",
            BUILTIN_CITATIONS["stage1"],
            lambda signal, fs: detect_r_peaks(signal, fs, polarity=polarity, scale_method=scale_method),
        ),
    }
    if model is not None:
        specs["electrotrace-two-stage"] = DetectorSpec(
            "electrotrace-two-stage",
            "electrotrace-compatible",
            BUILTIN_CITATIONS["electrotrace-two-stage"],
            lambda signal, fs: detect_r_peaks_two_stage(
                signal, fs, model, polarity=polarity, scale_method=scale_method
            ),
        )
    return specs


def discover_detectors(
    *, model=None, polarity: str = "adaptive", scale_method: str = "windowed_std"
) -> dict[str, DetectorSpec]:
    """Return built-ins plus third-party entry-point detectors."""
    specs = _builtin_specs(model=model, polarity=polarity, scale_method=scale_method)
    eps = entry_points()
    selected = (
        eps.select(group="electrotrace.detectors")
        if hasattr(eps, "select")
        else eps.get("electrotrace.detectors", ())
    )
    for ep in selected:
        try:
            obj = ep.load()
            obj = obj() if isinstance(obj, type) else obj
            if not callable(obj):
                continue
            name = str(getattr(obj, "name", ep.name))
            version = str(getattr(obj, "version", "external"))
            citation = str(getattr(obj, "citation", "No citation declared by plugin."))
            specs[name] = DetectorSpec(
                name, version, citation, obj, source=f"entry-point:{ep.value}"
            )
        except Exception as exc:
            warnings.warn(
                f"Could not load ElectroTrace detector plugin {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return specs


def get_detector(
    name: str, *, model=None, polarity: str = "adaptive", scale_method: str = "windowed_std"
) -> DetectorSpec:
    normalized = str(name).strip().lower().replace("_", "-")
    specs = discover_detectors(model=model, polarity=polarity, scale_method=scale_method)
    aliases = {
        "pantompkins": "pan-tompkins",
        "electrotrace-2stage": "electrotrace-two-stage",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"unknown detector '{name}'. Available detectors: {available}")
    return specs[normalized]
