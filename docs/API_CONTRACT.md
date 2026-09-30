# API Contract — SDK 0.1.4

## Preferred production input

`numpy.uint8`, BGR, `H x W x 3`, containing **only the product-visible capture ROI**. The SDK does not own screen coordinates or UI layout.

```python
gate = PassportQualityGate(config="capture_viewport", device="auto")
preview = gate.analyze_roi_preview(roi_bgr, timestamp=t, viewport_metadata=metadata)
final = gate.analyze_roi_final(selected_roi_bgr, timestamp=t_click, viewport_metadata=metadata)
```

`viewport_metadata` is optional diagnostic metadata. It does not change pixel content.

## Full-camera convenience adapter

`CaptureViewport` can crop an oriented camera frame before analysis:

```python
from passport_quality_gate import CaptureViewport
viewport = CaptureViewport(0.16, 0.18, 0.68, 0.64)
result = gate.analyze_capture_preview(frame_bgr, viewport, timestamp=t)
```

The host is still responsible for mapping the actual UI rectangle through its preview transform.

## Public result

Use `to_public_result(...)` or the `*_public` helpers when only the stable integration fields are needed:

```text
capture_allowed
capture_quality_state
workflow_state
guidance_code
recommended_adjustment
blocking_issues
advisories
timing_ms.total
```

Keep the full result when using `BestFrameSelector`, debug diagnostics or `extract_passport_page`.

## Session lifecycle

One gate + selector per capture session. Use monotonic timestamps. On new document, meaningful pause/resume or viewport change: `gate.reset()` and `selector.clear()`.

## Compatibility

Legacy `analyze_preview(frame, guide_box)` / `analyze_final(frame, guide_box)` remain available. They use the legacy full-frame/guide contract and are not the recommended V4 product integration.
