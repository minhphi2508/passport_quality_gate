# Capture viewport integration

The host application must provide the exact pixel region that is visible and usable for capture. Pixels outside that region are UI-only and must not influence localization, quality analysis, temporal state, best-frame selection, final validation, or the OCR handoff.

## Recommended flow

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

# Load the capture-viewport policy.
gate = PassportQualityGate(config="configs/capture_viewport.yaml", device="auto")
selector = BestFrameSelector()

# The app maps its on-screen capture rectangle into camera-buffer coordinates
# and supplies only those ROI pixels to the SDK.
preview = gate.analyze_roi_preview(roi_bgr, timestamp=t, viewport_metadata=metadata)
selector.push(roi_bgr, preview, timestamp=t)

selected = selector.select_recent(trigger_timestamp=t_click)
chosen = selected.frame if selected is not None else roi_bgr
chosen_meta = selected.result["capture_viewport"] if selected is not None else metadata

final = gate.analyze_roi_final(chosen, timestamp=t_click, viewport_metadata=chosen_meta)
if final["capture_allowed"]:
    passport_crop = extract_passport_page(chosen, final)
```

The final check must run on the exact selected ROI bytes. Do not re-crop a newer camera frame for final validation.

## Host-app responsibilities

The app owns camera lifecycle, preview rendering, the grey outside-mask, orientation, mirror handling, UI-to-camera coordinate transforms, session lifecycle, localized copy, storage/privacy policy, and downstream OCR invocation.

`CaptureViewport` and `map_preview_viewport()` are reference helpers for mapping a UI rectangle to an oriented camera buffer. The production app must use its real preview transform; the SDK cannot infer platform-specific scaling, lens correction, arbitrary UI transforms, or camera cropping.

## Decision hierarchy

The capture policy is stage-aware:

1. No reliable passport page: ask the user to place the passport in frame.
2. Confirmed physical truncation: directional move guidance.
3. Missing or incomplete bottom machine-readable text: block capture.
4. Motion / rotation / perspective.
5. Scale and resolution.
6. Lighting, glare, blur, contrast, and noise.
7. READY.

The SDK does not require all four paper corners to be visible. Small blank-margin loss may be acceptable, while confirmed loss of required content is not.

## MRZ policy

The packaged YOLO detector is authoritative for MRZ presence. Preview allows a short debounce for a transient miss; final validation does not. A detected MRZ that is visibly incomplete at a capture boundary is rejected even if the detector still returns a box.

The UI should not expose the term “MRZ”. Use plain-language guidance such as “Make sure the two lines of text at the bottom are visible.”

## Reference webcam harness

```bash
python examples/webcam_capture.py \
  --source 0 \
  --device auto \
  --viewport 0.16 0.18 0.68 0.64 \
  --metrics outputs/capture_metrics.jsonl
```

Keys:

- `D`: diagnostics
- `C`: product-style capture using the best recent eligible ROI
- `F`: final-check the current ROI directly
- `R`: reset session and selector
- `Q`: quit

Images are not saved unless `--record-images` is supplied.

## Validation scope

The automated suite covers ROI isolation, camera-coordinate mapping, MRZ presence/completeness, physical cut guidance, page-relative motion, glare behavior, best-frame selection, final validation, and crop handoff. Manual camera validation remains required for target-device behavior.

This SDK is a capture-quality gate. It does not perform authenticity verification and does not guarantee downstream OCR success.
