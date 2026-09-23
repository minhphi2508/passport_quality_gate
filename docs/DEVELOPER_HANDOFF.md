# Developer Handoff — Passport Quality Gate SDK 0.1.3

## Deliverable

SDK version:

```text
0.1.3
```

Quality policy:

```text
FP2-GOLDEN-ACTUAL
```

The 13-file Golden quality core remains unchanged from the approved baseline.

v0.1.3 adds integration-layer correctness fixes, recent-best-frame handling, and passport-page extraction for downstream OCR handoff.

## What the SDK owns

- passport/MRZ localization
- quality analysis and guidance codes
- preview temporal state
- final ACCEPT / RETAKE quality check
- bounded recent-best-frame selector
- perspective-corrected passport-page extraction
- packaged detector/config assets

## What the dev team owns

The consuming team must implement and validate:

1. **Camera integration**
   - camera lifecycle
   - preview resolution/FPS
   - autofocus/exposure strategy
   - orientation handling
   - frame conversion to BGR `uint8 HxWx3`

2. **UI/UX**
   - guide rectangle
   - localized strings for `guidance_code`
   - blocker/advisory presentation
   - capture/retake interaction
   - accessibility and product copy

3. **Session lifecycle**
   - one gate/selector state per live stream
   - `gate.reset()` + `selector.clear()` for new document/session and meaningful pause/resume
   - monotonic timestamps

4. **OCR/VLM handoff**
   - use the accepted in-memory passport crop as OCR input
   - invoke the downstream OCR/VLM
   - handle OCR errors/retries
   - do not assume SDK `ACCEPT` guarantees OCR correctness

5. **Storage/privacy/network**
   - image retention policy
   - encryption / secure transport
   - telemetry policy
   - consent / privacy handling
   - server/API transport if used

6. **Target deployment**
   - CPU/GPU/mobile runtime choice
   - performance profiling
   - memory/thermal/battery testing
   - packaging into the application architecture
   - concurrency/process model

## Minimal integration

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

gate = PassportQualityGate(device="auto")
selector = BestFrameSelector()

# Preview:
preview = gate.analyze_preview(frame_bgr, guide_box=(x, y, w, h), timestamp=t)
selector.push(frame_bgr, preview, timestamp=t)

# UI:
guidance_code = preview.get("guidance_code")
capture_ready = bool(preview.get("capture_allowed"))

# Capture:
best = selector.select_recent(trigger_timestamp=t_click)
selected_frame = best.frame if best is not None else current_frame_bgr

# Authoritative final check:
final = gate.analyze_final(
    selected_frame,
    guide_box=(x, y, w, h),
    timestamp=t_click,
)

if final["capture_allowed"]:
    passport_crop = extract_passport_page(selected_frame, final)
    send_to_ocr(passport_crop)
else:
    request_retake(final)
```

## Important technical notes

### Use full preview result with BestFrameSelector

Do not push `analyze_preview_public()` output into the selector. The selector uses quality/localization diagnostics available in the full result.

### Final must check selected pixels

If the selector returns an older recent frame, final analysis must run on **that exact frame**, not the current shutter frame.

### Crop comes after final ACCEPT

The passport crop is the preferred OCR handoff image. The full camera frame is only needed during quality analysis and may optionally be retained for debug according to the host app's privacy policy.

### No silent full-frame fallback

If `extract_passport_page()` fails after final acceptance, the caller must handle it explicitly. Do not silently send the full camera frame to OCR unless that fallback is separately approved and validated.

### Timestamps

`BestFrameSelector` rejects:

- NaN / infinity timestamps
- duplicate push timestamps
- decreasing push timestamps
- capture trigger timestamps earlier than the latest push

Use one monotonic clock source per session.

## Known limitations the dev team must not hide

- partial passport truncation can occasionally be missed
- completeness/corner evidence is not perfect
- small OCR-critical glare can occasionally pass
- strong motion can create preview instability
- the policy is not calibrated against a large OCR-linked real-world dataset
- ACCEPT is not an authenticity verdict and does not guarantee OCR success

The app should preserve a clear retake/error path.

## What is intentionally not implemented in v0.1.3

- production Android/iOS camera module
- React Native/Flutter/native mobile wrapper
- ONNX/TensorRT/CoreML/TFLite conversion
- server scaling/concurrency architecture
- document identity tracking across long sessions
- automatic OCR invocation
- OCR-result-based quality feedback loop
- production telemetry/analytics
- production privacy/storage policy
- broad multi-device calibration
- new completeness/glare model
- authenticity/fraud detection

These are downstream product/deployment tasks or future research, not missing Python-SDK bugs.

## Acceptance before app release

At minimum, the consuming team should test:

- good steady passport
- natural hand jitter
- strong shake
- far-but-readable passport
- very far / low-resolution passport
- too close
- partial crop on each edge
- glare on MRZ
- glare on VIZ/body text
- pause/resume
- orientation change
- new-document session reset
- shutter-time motion with BestFrameSelector
- crop extraction and OCR handoff
- low-memory / target-device thermal behavior

Record downstream OCR outcome for the exact selected/cropped image whenever possible.
