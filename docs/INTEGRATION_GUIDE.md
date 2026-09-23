# Integration Guide

Recommended flow:

```text
camera/app preview frames
        ↓
PassportQualityGate.analyze_preview(...)
        ↓
UI maps guidance_code to localized text
        ↓
BestFrameSelector.push(...) [optional]
        ↓
user presses capture
        ↓
select_recent(...) or current frame
        ↓
PassportQualityGate.analyze_final(...)
        ↓
accepted image → OCR
```

The SDK does not open the camera or dictate preview resolution/FPS. The caller controls analysis cadence according to target hardware and latency budget.

For a new passport/session, call both `gate.reset()` and `selector.clear()`.

For server use, each concurrent stream needs independent temporal state. Detector/model sharing across sessions is a future optimization and is not part of v0.1.
