# API Contract — SDK 0.1.2

## Input
Current Python reference input is `numpy.uint8` BGR, shape `H x W x 3`. The engine does not require 1280x720. A UI guide is supplied in normalized `(x, y, w, h)` coordinates.

## Main object
```python
PassportQualityGate(device="auto", weights=None, config=None)
```

- `analyze_preview(frame, guide_box, timestamp=None)` — full Golden FP2 preview dictionary (stable decisions + research diagnostics).
- `analyze_final(frame, guide_box, timestamp=None)` — full-frame final policy. `guide_box` is required for full-frame analysis. Use `analyze_document_crop()` when the input is already a cropped passport page.
- `analyze_document_crop(frame, timestamp=None)` — already-cropped data page.
- `analyze_preview_public(...)`, `analyze_final_public(...)`, `analyze_document_crop_public(...)` — compact JSON-ready public contract.
- `reset()` — clear temporal/motion history before a new document/session.
- `metadata` / `runtime_info()` — small runtime/version/device metadata; both expose the same values.

Use one instance per live stream/session. Do not share one stateful instance across unrelated concurrent users.

## Stable integration fields for SDK 0.1.x
- `capture_allowed`
- `capture_quality_state`
- `workflow_state`
- `guidance_code`
- `recommended_adjustment`
- `blocking_issues`
- `advisories`
- `timing_ms.total`

The remaining engine diagnostics are intentionally not frozen as a long-term app/server contract.

## Best-frame selector
`BestFrameSelector` accepts analyzed frames, retains only capture-allowed candidates in a bounded RAM buffer, and returns the highest-ranked recent candidate on shutter action. Window, frame count, memory budget, geometry consistency, and recency weight are configurable. It never writes frames to disk.

## Guidance strings
The SDK returns codes, not user-facing Vietnamese/English text. UI localization belongs to the application layer.
