"""Installed-wheel acceptance check for the capture-viewport SDK."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameConfig, BestFrameSelector


def selector_smoke() -> dict:
    selector = BestFrameSelector(BestFrameConfig())
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def result(blur: float) -> dict:
        return {
            "capture_allowed": True,
            "capture_quality_state": "READY",
            "confidence": 0.9,
            "quality": {
                "blur_score": blur,
                "low_resolution_score": 0.05,
                "too_dark_score": 0.0,
                "too_bright_score": 0.0,
                "low_contrast_score": 0.0,
                "glare_score": 0.0,
                "noise_score": 0.0,
            },
            "raw_metrics": {"motion": {"score": 0.0}},
            "localization": {"bbox": [10, 10, 150, 110]},
        }

    t0 = time.monotonic()
    selector.push(frame, result(0.25), timestamp=t0)
    selector.push(frame, result(0.05), timestamp=t0 + 0.20)
    selector.push(frame, result(0.40), timestamp=t0 + 0.40)
    best = selector.select_recent(trigger_timestamp=t0 + 0.45)
    assert best is not None and best.timestamp == t0 + 0.20
    return best.metadata()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="Optional full product-visible ROI image")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    gate = PassportQualityGate(config="capture_viewport", device=args.device)
    out = {"runtime": gate.runtime_info(), "best_frame": selector_smoke()}

    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        preview = gate.analyze_roi_preview(image, timestamp=1.0)
        gate.reset()
        final = gate.analyze_roi_final(image, timestamp=2.0)
        out["image"] = {"preview": preview, "final": final}

    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print("ACCEPTANCE CHECK: PASS")


if __name__ == "__main__":
    main()
