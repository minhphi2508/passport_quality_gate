# Integration Guide — SDK 0.1.4

## Recommended production flow

1. App defines the visible capture rectangle.
2. App maps it through the real preview/sensor transform and extracts the exact oriented ROI.
3. SDK analyzes ROI-only pixels with `config="capture_viewport"`.
4. App maps `guidance_code` to localized UI/UX.
5. Push capture-allowed ROI frames into `BestFrameSelector`.
6. At shutter, final-check the selected ROI (or current ROI fallback).
7. Only after ACCEPT, call `extract_passport_page` on the same ROI.
8. Pass the in-memory crop to OCR/VLM.

Do not feed the full raw frame while merely drawing a grey mask in UI. Pixels the user cannot capture must not influence the model.

Use one gate/selector per session and monotonic timestamps. Reset both for a new document/session, meaningful pause/resume or changed viewport geometry.
