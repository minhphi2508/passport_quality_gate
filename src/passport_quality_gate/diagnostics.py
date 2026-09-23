from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from .api import QUALITY_POLICY, SDK_CANDIDATE_VERSION, _default_config_path, _default_weights_path


def runtime_diagnostics() -> dict[str, Any]:
    """Return lightweight deployment diagnostics; performs no inference."""
    data: dict[str, Any] = {
        "sdk_version": SDK_CANDIDATE_VERSION,
        "quality_policy": QUALITY_POLICY,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "default_config_exists": False,
        "default_weights_exists": False,
    }
    try:
        import numpy as np
        data["numpy"] = np.__version__
    except Exception as exc:
        data["numpy_error"] = type(exc).__name__
    try:
        import cv2
        data["opencv"] = cv2.__version__
    except Exception as exc:
        data["opencv_error"] = type(exc).__name__
    try:
        import torch
        data["torch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:
        data["torch_error"] = type(exc).__name__
        data["cuda_available"] = False
        data["cuda_device_count"] = 0
    try:
        import ultralytics
        data["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:
        data["ultralytics_error"] = type(exc).__name__
    try:
        cfg = Path(_default_config_path())
        data["default_config_exists"] = cfg.is_file()
        data["default_config_path"] = str(cfg)
    except Exception as exc:
        data["default_config_error"] = str(exc)
    try:
        weights = Path(_default_weights_path())
        data["default_weights_exists"] = weights.is_file()
        data["default_weights_path"] = str(weights)
    except Exception as exc:
        data["default_weights_error"] = str(exc)
    return data
