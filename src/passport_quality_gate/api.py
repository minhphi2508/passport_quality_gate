from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any, Mapping, Optional, Union

import numpy as np

from .analyzer import Analyzer
from .config import load_config
from .localization import YoloLocalizer
from .types import GuideBoxLike, coerce_guide_box


QUALITY_POLICY = "FP2-GOLDEN-ACTUAL"
SDK_CANDIDATE_VERSION = "0.1.4"


def _project_root() -> Path:
    # Source-tree reference SDK layout: <root>/src/passport_quality_gate/api.py
    return Path(__file__).resolve().parents[2]


def _package_assets() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("defaults.yaml")

def _capture_viewport_config_path() -> Path:
    packaged = _package_assets() / "capture_viewport.yaml"
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("Packaged capture_viewport profile not found")

def _resolve_config_source(config: Optional[Union[str, Path, Mapping[str, Any]]]):
    if config is None:
        return _default_config_path()
    if isinstance(config, str) and config in {"capture_viewport", "v4"}:
        return _capture_viewport_config_path()
    return config

def _default_weights_path() -> Path:
    packaged = _package_assets() / "passport_detector_ver3_best.pt"
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("Packaged passport detector weights not found")

def _resolve_device(device: Union[str, int]) -> Union[str, int]:
    """Resolve 'auto' without imposing a deployment architecture on callers."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        # The quality package can still be imported/tested without torch; YOLO
        # construction will report the actual dependency problem if invoked.
        pass
    return "cpu"


def _normalize_final_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Repair final output aliases without changing the frozen decision.

    Normalize shared ACCEPT/RETAKE aliases across capture profiles.
    """
    out = dict(result)
    if out.get("mode") == "final" and out.get("state") in {"ACCEPT", "RETAKE"}:
        accepted = out["state"] == "ACCEPT"
        out["capture_allowed"] = accepted
        out["ready_for_capture"] = accepted
        out["capture_quality_state"] = (
            ("READY" if out.get("advisories") else "OPTIMAL")
            if accepted else "NOT_READY"
        )
    return out


def to_public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small JSON-ready contract intended for app/server consumers.

    Detailed diagnostics remain available from analyze_*(), while this compact
    field set is the external integration contract.
    """
    result = _normalize_final_result(result)
    timing = result.get("timing_ms") or {}
    return {
        **({"guidance_text": result["guidance_text"]} if "guidance_text" in result else {}),
        "capture_allowed": bool(result.get("capture_allowed")),
        "capture_quality_state": result.get("capture_quality_state"),
        "workflow_state": result.get("workflow_state"),
        "guidance_code": result.get("guidance_code"),
        "recommended_adjustment": result.get("recommended_adjustment"),
        "blocking_issues": list(result.get("blocking_issues") or []),
        "advisories": list(result.get("advisories") or []),
        "timing_ms": {
            "total": timing.get("total"),
        },
    }


class PassportQualityGate:
    """Passport capture-quality integration wrapper.

    This class does not open a camera, render UI, save images, or create logs.
    One instance should be used per live preview stream because FP2 preview
    state contains temporal/motion history. Call ``reset()`` when a session or
    document changes.

    Input frames follow the frozen engine contract: uint8 HxWx3 BGR arrays.
    """

    def __init__(
        self,
        *,
        device: Union[str, int] = "auto",
        weights: Optional[Union[str, Path]] = None,
        config: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        localizer: Any = None,
    ) -> None:
        config_source: Union[str, Path, Mapping[str, Any]]
        config_source = _resolve_config_source(config)
        # The public signature accepts any Mapping; the frozen loader accepts
        # dict specifically. Convert at this boundary, retaining its deep copy.
        self.config = load_config(dict(config_source) if isinstance(config_source, Mapping) else config_source)
        self.device = _resolve_device(device)
        self.weights = Path(weights) if weights is not None else _default_weights_path()

        if localizer is None:
            localizer = YoloLocalizer(
                self.weights,
                self.config["localization"],
                self.device,
            )
        self.localizer = localizer
        self._analyzer = Analyzer(localizer, self.config)
        # A missing-document final calls motion.reset() in the frozen engine.
        # Isolate final orchestration so it cannot erase live-preview history.
        # Both analyzers share the same localizer/model; no second model load.
        self._final_analyzer = Analyzer(localizer, self.config)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "sdk_candidate_version": SDK_CANDIDATE_VERSION,
            "quality_policy": self.config["capture_policy"]["profile"] if self.config.get("capture_policy", {}).get("enabled", False) else QUALITY_POLICY,
            "device": self.device,
            "production_validated": False,
        }

    def runtime_info(self) -> dict[str, Any]:
        """Return the same small runtime metadata mapping as ``metadata``.

        Kept as a convenience method for integration/acceptance scripts.
        It does not inspect or alter Golden quality behavior.
        """
        return dict(self.metadata)

    def reset(self) -> None:
        """Reset temporal/motion state before a new live capture session."""
        self._analyzer.reset()
        self._final_analyzer.reset()

    def analyze_preview(
        self,
        frame: np.ndarray,
        guide_box: GuideBoxLike,
        *,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """Analyze one live-preview frame using Golden FP2 preview policy."""
        return self._analyzer.analyze_frame(
            frame,
            coerce_guide_box(guide_box),
            mode="preview",
            timestamp=timestamp,
            capture_context="live_preview",
        )

    def analyze_final(
        self,
        frame: np.ndarray,
        guide_box: GuideBoxLike,
        *,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """Analyze one full camera frame using Golden FP2 final policy."""
        return _normalize_final_result(self._final_analyzer.analyze_frame(
            frame,
            coerce_guide_box(guide_box),
            mode="final",
            timestamp=timestamp,
            capture_context="full_frame_final",
        ))

    def analyze_document_crop(
        self,
        frame: np.ndarray,
        *,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """Analyze an already-cropped passport data-page image."""
        return _normalize_final_result(self._final_analyzer.analyze_frame(
            frame,
            guide_box=None,
            mode="final",
            timestamp=timestamp,
            capture_context="document_crop",
        ))

    def _require_viewport_profile(self):
        if not self.config.get('capture_policy',{}).get('capture_viewport',False):
            raise ValueError("Explicit ROI methods require config='capture_viewport'")

    def analyze_roi_preview(self, roi, *, timestamp=None, viewport_metadata=None):
        """Preferred V4 API: input consists ONLY of product-visible ROI pixels."""
        self._require_viewport_profile()
        # A changed physical viewport or ROI resolution starts fresh evidence.
        key=(tuple(roi.shape),repr(viewport_metadata))
        if getattr(self,'_viewport_key',key)!=key: self.reset()
        self._viewport_key=key
        result=self.analyze_preview(roi,(0.,0.,1.,1.),timestamp=timestamp)
        result['capture_viewport']=deepcopy(viewport_metadata) or {'analysis_frame_size':result['frame_size'],'input':'roi_pixels'}
        return result

    def analyze_roi_final(self, roi, *, timestamp=None, viewport_metadata=None):
        """Final-check the selected ROI itself, without recropping or resizing."""
        self._require_viewport_profile()
        result=self.analyze_final(roi,(0.,0.,1.,1.),timestamp=timestamp)
        result['capture_viewport']=deepcopy(viewport_metadata) or {'analysis_frame_size':result['frame_size'],'input':'roi_pixels'}
        return result

    def analyze_capture_preview(self, frame, capture_viewport, *, timestamp=None, preview_transform=None):
        """Convenience V4 path; explicitly crop BEFORE localization/analysis."""
        self._require_viewport_profile()
        roi,meta=capture_viewport.extract(frame,transform=preview_transform)
        return self.analyze_roi_preview(roi,timestamp=timestamp,viewport_metadata=meta)

    def analyze_capture_final(self, frame, capture_viewport, *, timestamp=None, preview_transform=None):
        self._require_viewport_profile()
        roi,meta=capture_viewport.extract(frame,transform=preview_transform)
        return self.analyze_roi_final(roi,timestamp=timestamp,viewport_metadata=meta)

    # Compact result helpers for app/server integrations.
    def analyze_preview_public(self, frame: np.ndarray, guide_box: GuideBoxLike, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_preview(frame, guide_box, timestamp=timestamp))

    def analyze_final_public(self, frame: np.ndarray, guide_box: GuideBoxLike, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_final(frame, guide_box, timestamp=timestamp))

    def analyze_document_crop_public(self, frame: np.ndarray, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_document_crop(frame, timestamp=timestamp))
