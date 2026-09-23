# Developer handoff — Passport Quality Gate SDK 0.1.2

## What the dev team owns
Camera lifecycle, preview resolution/FPS, UI/UX, localized strings, shutter button, app storage/networking and mobile/server deployment.

## What this SDK owns
Passport quality analysis, readiness/guidance codes, final quality check, and an optional recent-best-frame selector.

## Minimal integration
```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector

gate = PassportQualityGate(device="auto")
selector = BestFrameSelector()

result = gate.analyze_preview(frame_bgr, guide_box=(x, y, w, h))
selector.push(frame_bgr, result, timestamp=t)

# On shutter/capture action:
best = selector.select_recent(trigger_timestamp=t_click)
ocr_frame = best.frame if best is not None else frame_bgr
final = gate.analyze_final_public(ocr_frame, guide_box=(x, y, w, h))  # guide_box required for full-frame final
```

Use one `PassportQualityGate` instance per live session/stream and call `reset()` when switching documents/users.

## Stable fields
For application code prefer `analyze_*_public()` or `to_public_result()` and consume only:
`capture_allowed`, `capture_quality_state`, `workflow_state`, `guidance_code`, `recommended_adjustment`, `blocking_issues`, `advisories`, `timing_ms.total`.

## Important boundaries
- No fixed 1280x720 requirement.
- No requirement to run on CPU; `device="auto"` chooses CUDA when available.
- The SDK does not open a camera or render UI.
- The SDK does not save images/logs by default.
- The best-frame buffer is RAM-only and bounded.
- Quality policy is frozen as `FP2-GOLDEN-ACTUAL`; known limitations are documented separately.
