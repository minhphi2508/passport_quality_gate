"""Minimal source-tree example for the public API wrapper.

This example intentionally owns camera acquisition outside the SDK. Replace
OpenCV VideoCapture with Android/iOS/server input in the real integration.
"""
from __future__ import annotations

import argparse

import cv2

from passport_quality_gate.api import PassportQualityGate


GUIDE = (0.16, 0.18, 0.68, 0.64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    gate = PassportQualityGate(device=args.device)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Cannot open source")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = gate.analyze_preview(frame, GUIDE)
            print(
                result["capture_allowed"],
                result["guidance_code"],
                result["timing_ms"]["total"],
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        gate.reset()


if __name__ == "__main__":
    main()
