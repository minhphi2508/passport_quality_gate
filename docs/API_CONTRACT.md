# API Contract — SDK 0.1.3

## Input

Current Python reference input:

```text
numpy.uint8
BGR
H x W x 3
```

The engine does not require a fixed 1280×720 frame size.

A UI guide is supplied in normalized:

```text
(x, y, width, height)
```

coordinates.

## Main object

```python
PassportQualityGate(
    device="auto",
    weights=None,
    config=None,
)
```

Main methods:

- `analyze_preview(frame, guide_box, timestamp=None)` — full Golden preview result with integration diagnostics.
- `analyze_final(frame, guide_box, timestamp=None)` — authoritative full-frame final result.
- `analyze_document_crop(frame, timestamp=None)` — final analysis for an already-cropped passport data page.
- `analyze_preview_public(...)`, `analyze_final_public(...)`, `analyze_document_crop_public(...)` — compact JSON-ready result.
- `reset()` — reset both preview and final orchestration state for a new session/document.
- `metadata` / `runtime_info()` — runtime/version/device metadata.

Use one instance per live stream/session.

## Stable integration fields for SDK 0.1.x

The compact public result exposes:

- `capture_allowed`
- `capture_quality_state`
- `workflow_state`
- `guidance_code`
- `recommended_adjustment`
- `blocking_issues`
- `advisories`
- `timing_ms.total`

Other diagnostics are not guaranteed as a long-term app/server contract.

## Final result normalization in v0.1.3

The frozen Golden Analyzer internally uses final `ACCEPT` / `RETAKE` states. SDK v0.1.3 normalizes final integration aliases at the wrapper boundary so that:

```text
ACCEPT → capture_allowed=True
RETAKE → capture_allowed=False
```

This does not change the underlying frozen quality decision, quality scores, blockers, advisories, thresholds, or detector weights.

Final orchestration also uses separate analyzer state from live preview so a final call cannot unexpectedly reset preview temporal history.

## Full result vs public result

Use the compact public result for UI/server decision integration.

Use the **full result** when calling:

- `BestFrameSelector.push(...)`
- `extract_passport_page(...)`

because these utilities require localization and/or quality diagnostics omitted from the compact public contract.

## Best-frame selector

```python
selector.push(frame, full_preview_result, timestamp=t)
best = selector.select_recent(trigger_timestamp=t_click)
```

Requirements:

- timestamps must be finite
- push timestamps must strictly increase within one session
- `trigger_timestamp` must not precede the latest pushed timestamp
- clear the selector for a new document/session

The selector is RAM-only and bounded.

## Passport-page extraction

```python
from passport_quality_gate.capture_output import extract_passport_page

crop = extract_passport_page(selected_full_frame, full_final_result)
```

The function returns an in-memory perspective-corrected BGR `numpy.ndarray`.

Call it only on the exact selected frame that passed final analysis.

If extraction fails, the caller should explicitly handle that failure. The SDK does not silently substitute the full camera frame.

## Guidance strings

The SDK returns guidance **codes**, not user-facing localized text.

UI copy and localization belong to the consuming application.
