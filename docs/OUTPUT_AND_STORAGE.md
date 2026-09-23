# Output and Storage Policy — SDK 0.1.3

Core SDK behavior is in-memory.

## No automatic production persistence

The following components do not write passport images to disk by themselves:

- `PassportQualityGate`
- `BestFrameSelector`
- `extract_passport_page`

Preview/final analysis returns Python objects. The selector stores a bounded rolling buffer in RAM. Passport-page extraction returns a BGR `numpy.ndarray`.

## Recommended production handoff

```text
selected full frame in RAM
        ↓
final ACCEPT
        ↓
passport crop in RAM
        ↓
OCR / VLM
```

The application should not need to save an intermediate JPEG simply to call OCR.

## Example scripts

`examples/webcam_demo.py` and `examples/best_frame_compare.py` may write images only when explicitly run in an opt-in test/debug mode.

The current webcam demo can save:

- accepted passport crop — OCR handoff candidate for manual validation
- selected full frame — debug/reference artifact
- JSON metadata

These example writes are not SDK runtime requirements.

## Production responsibility

The host application/team owns:

- whether images are persisted
- retention duration
- encryption
- access control
- deletion
- telemetry/logging
- consent/privacy policy
- secure network transfer

Passport images may contain sensitive personal data. Production persistence should be minimized and reviewed by the responsible product/security teams.
