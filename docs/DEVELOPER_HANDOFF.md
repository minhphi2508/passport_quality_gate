# Developer Handoff — Passport Quality Gate SDK 0.1.4

## What you receive

- installable `passport_quality_gate-0.1.4-py3-none-any.whl`
- source SDK ZIP
- SHA256 checksums
- capture-viewport integration docs

## Required app responsibility

The app must provide the exact product-visible capture ROI. If the UI greys pixels outside the rectangle, those pixels must not be analyzed as part of the capture image. The app owns screen-to-camera mapping, orientation/mirroring, camera lifecycle and UI/UX.

## Minimal integration

```python
from passport_quality_gate import PassportQualityGate, BestFrameSelector, extract_passport_page

gate = PassportQualityGate(config="capture_viewport", device="auto")
selector = BestFrameSelector()
```

Use ROI preview → selector → exact selected ROI final → crop after ACCEPT. See `CAPTURE_VIEWPORT.md`.

## Scope boundaries

The SDK does not implement authenticity/fraud detection, downstream OCR, network/storage policy, mobile UI, or a production camera stack. `production_validated` intentionally remains false until broader deployment validation is complete.
