# Known Limitations — V4 Capture Viewport

- The SDK assumes the host supplies the correct product-visible ROI; incorrect UI-to-camera mapping invalidates geometry/completeness reasoning.
- Detector/corner reliability can degrade under extreme blur, rotation, perspective, occlusion or unusual backgrounds.
- Quality thresholds are not calibrated against a large OCR-linked real-world passport/device dataset.
- Small destructive glare or borderline partial truncation can still be imperfectly classified.
- Best-frame selection improves shutter-time robustness but does not guarantee the best downstream OCR result.
- The SDK does not verify authenticity and does not guarantee OCR correctness.
- Broad target-device latency, thermal and battery validation remains application/deployment work.
- Previously cropped digital passport images are outside the physical live-camera truncation guarantee unless separately validated as a supported input.
