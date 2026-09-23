# Integration Guide — SDK 0.1.3

## Recommended flow

```text
camera/app preview frame
        ↓
gate.analyze_preview(...)
        ↓
UI maps guidance_code → localized instruction
        ↓
selector.push(frame, full_preview_result)
        ↓
user presses capture
        ↓
selector.select_recent(...) or current-frame fallback
        ↓
gate.analyze_final(exact_selected_frame, ...)
        ↓
      ACCEPT?
       /   \
     no     yes
     ↓       ↓
   RETAKE  extract_passport_page(...)
              ↓
       passport crop in memory
              ↓
         OCR / VLM
```

## Important integration rule

The final quality check must run on the **exact selected full frame** before the passport page is cropped.

Do not:

```text
crop first → final-check a different image → OCR another image
```

The selected pixels, final decision, and crop must refer to the same capture candidate.

## Reference integration

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

gate = PassportQualityGate(device="auto")
selector = BestFrameSelector()

# Preview loop
preview = gate.analyze_preview(frame_bgr, guide_box, timestamp=t)
selector.push(frame_bgr, preview, timestamp=t)

# UI can consume the compact contract if desired:
ui_result = {
    "capture_allowed": preview.get("capture_allowed"),
    "guidance_code": preview.get("guidance_code"),
    "recommended_adjustment": preview.get("recommended_adjustment"),
}

# Capture action
selected = selector.select_recent(trigger_timestamp=t_click)
chosen = selected.frame if selected is not None else current_frame_bgr

final = gate.analyze_final(chosen, guide_box, timestamp=t_click)

if not final["capture_allowed"]:
    # Keep the user in capture flow / request retake.
    ...
else:
    passport_crop = extract_passport_page(chosen, final)
    # Pass passport_crop directly to OCR/VLM.
```

## Session lifecycle

For a new passport, a new user/session, or a meaningful pause/resume boundary:

```python
gate.reset()
selector.clear()
```

Do not reuse one stateful gate across unrelated concurrent streams.

## Camera ownership

The SDK does not open or configure the production camera.

The app chooses:

- camera API/backend
- preview size/FPS
- autofocus/exposure behavior
- analysis cadence
- guide rectangle rendering
- shutter behavior

The Python webcam examples are reference harnesses only.

## Guidance UX

The SDK returns codes. The app owns:

- Vietnamese/English/etc. strings
- iconography
- message priority/presentation
- animations
- accessibility
- optional secondary advisory display

A useful product pattern is to keep a blocker as the primary instruction while showing a non-blocking positioning advisory separately.

## Failure handling

The app must define product behavior for:

- `RETAKE`
- no recent best frame
- passport crop extraction failure
- detector/model initialization failure
- camera interruption / app backgrounding
- timestamp/session reset
- OCR/network failure after an accepted capture

Do not silently treat unknown/error states as successful capture.

## Next integration phase

After mobile/app integration, evaluate:

- shutter frame vs selected recent frame
- accepted crop vs downstream OCR result
- OCR field accuracy / MRZ correctness
- time-to-ready and guidance stability
- target-device latency, RAM, battery and thermal behavior

Do not tune the Golden thresholds from anecdotal single-device failures without a controlled evaluation set.
