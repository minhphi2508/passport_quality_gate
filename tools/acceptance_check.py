"""Fresh-environment acceptance helper for the installed SDK/wheel."""
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
    selector = BestFrameSelector(BestFrameConfig(window_ms=750, max_age_ms=900, max_frames=12, max_memory_mb=96))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    def r(blur: float):
        return {
            "capture_allowed": True,
            "capture_quality_state": "READY",
            "confidence": 0.9,
            "quality": {
                "blur_score": blur, "low_resolution_score": 0.05,
                "too_dark_score": 0.0, "too_bright_score": 0.0,
                "low_contrast_score": 0.0, "glare_score": 0.0, "noise_score": 0.0,
            },
            "raw_metrics": {"motion": {"score": 0.0}},
            "localization": {"bbox": [10, 10, 150, 110]},
        }
    t0 = time.monotonic()
    selector.push(frame, r(0.25), timestamp=t0)
    selector.push(frame, r(0.05), timestamp=t0 + 0.20)
    selector.push(frame, r(0.40), timestamp=t0 + 0.40)
    best = selector.select_recent(trigger_timestamp=t0 + 0.45)
    assert best is not None and best.timestamp == t0 + 0.20
    return {"buffered_frames": selector.buffered_frames, "buffered_mb": round(selector.buffered_megabytes, 3), "best": best.metadata()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--guide", nargs=4, type=float, default=(0.15, 0.20, 0.70, 0.60), metavar=("X", "Y", "W", "H"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    gate = PassportQualityGate(device=args.device)
    out = {"runtime": gate.runtime_info(), "best_frame": selector_smoke()}
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        guide = tuple(args.guide)
        preview = gate.analyze_preview_public(img, guide)
        gate.reset()
        final = gate.analyze_final_public(img, guide)
        out["image"] = {"preview": preview, "final": final}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("ACCEPTANCE CHECK: PASS")


if __name__ == "__main__":
    main()
