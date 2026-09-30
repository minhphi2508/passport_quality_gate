# Passport Quality Gate

Python reference SDK for pre-OCR passport capture quality assessment.

**SDK candidate:** `0.1.4`
**Capture policy:** `V4-CAPTURE-VIEWPORT`

The production contract is ROI-first: the host application maps its visible capture rectangle to camera pixels and sends only that effective capture ROI to the SDK. UI, camera lifecycle, preview rendering and localized user copy remain application responsibilities.

## Quick start

```python
from passport_quality_gate import PassportQualityGate, BestFrameSelector, extract_passport_page

gate = PassportQualityGate(config="capture_viewport", device="auto")
selector = BestFrameSelector()

preview = gate.analyze_roi_preview(roi_bgr, timestamp=t, viewport_metadata=metadata)
selector.push(roi_bgr, preview, timestamp=t)

selected = selector.select_recent(trigger_timestamp=t_click)
chosen = selected.frame if selected is not None else roi_bgr
chosen_meta = selected.result["capture_viewport"] if selected is not None else metadata

final = gate.analyze_roi_final(chosen, timestamp=t_click, viewport_metadata=chosen_meta)
if final["capture_allowed"]:
    passport_crop = extract_passport_page(chosen, final)
```

## SDK owns

- passport-page and MRZ localization
- completeness/positioning evidence within the supplied ROI
- motion, blur, exposure, glare, contrast, noise, rotation and perspective checks
- stage-aware guidance codes
- recent-best-frame selection
- final ACCEPT/RETAKE validation
- perspective-corrected passport crop for downstream OCR/VLM

## Host application owns

- camera lifecycle and preview
- the visible capture rectangle and grey outside-mask
- UI-to-camera coordinate mapping, orientation and mirroring
- user-facing text/UX
- storage/privacy/networking
- downstream OCR/VLM invocation and retry policy

## Validation

```bash
python tools/verify_release_core.py
python -m pytest -q
python -m compileall -q src examples tools tests
```

Manual camera acceptance was completed on the accepted V4 candidate before this release-prep pass. `production_validated` remains `False`: broad device/population validation and OCR-linked calibration are still future work.

See `docs/CAPTURE_VIEWPORT.md`, `docs/API_CONTRACT.md`, and `docs/DEVELOPER_HANDOFF.md`.
