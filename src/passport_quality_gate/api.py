from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from .analyzer import Analyzer
from .config import load_config
from .localization import YoloLocalizer
from .types import GuideBoxLike, coerce_guide_box


QUALITY_POLICY = "FP2-GOLDEN-ACTUAL"
SDK_CANDIDATE_VERSION = "0.1.3"


def _project_root() -> Path:
    # Source-tree reference SDK layout: <root>/src/passport_quality_gate/api.py
    return Path(__file__).resolve().parents[2]


def _package_assets() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _default_config_path() -> Path:
    # Prefer the auditable source-tree Golden file when present. Fall back to
    # package assets so a wheel/install does not depend on repository layout.
    source = _project_root() / "configs" / "thresholds_v4.yaml"
    if source.is_file():
        return source
    packaged = _package_assets() / "thresholds_v4.yaml"
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("Default thresholds_v4.yaml not found in source tree or package assets")


def _default_weights_path() -> Path:
    source = _project_root() / "models" / "passport_detector_ver3_best.pt"
    if source.is_file():
        return source
    packaged = _package_assets() / "passport_detector_ver3_best.pt"
    if packaged.is_file():
        return packaged
    raise FileNotFoundError("Default passport detector weights not found in source tree or package assets")


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

    Raw Analyzer remains immutable. Only its final ACCEPT/RETAKE aliases need
    normalization; preview readiness still belongs entirely to Golden.
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

    The frozen engine emits many research diagnostics. Those remain available
    from analyze_*(), but only this compact field set is treated as the stable
    external contract for SDK 0.1.x.
    """
    result = _normalize_final_result(result)
    timing = result.get("timing_ms") or {}
    return {
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
    """Stable integration wrapper around the frozen FP2 Golden engine.

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
        config_source = _default_config_path() if config is None else config
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
            "quality_policy": QUALITY_POLICY,
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

    # Convenience methods for consumers that only want the documented stable
    # contract and do not need research diagnostics from the Golden engine.
    def analyze_preview_public(self, frame: np.ndarray, guide_box: GuideBoxLike, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_preview(frame, guide_box, timestamp=timestamp))

    def analyze_final_public(self, frame: np.ndarray, guide_box: GuideBoxLike, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_final(frame, guide_box, timestamp=timestamp))

    def analyze_document_crop_public(self, frame: np.ndarray, *, timestamp: Optional[float] = None) -> dict[str, Any]:
        return to_public_result(self.analyze_document_crop(frame, timestamp=timestamp))
