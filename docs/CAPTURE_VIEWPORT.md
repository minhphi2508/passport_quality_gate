# Capture Viewport Contract

The model must evaluate the same pixels the product considers capturable. If the app greys everything outside a rectangle, those hidden pixels must not be supplied as usable model input.

## Integration boundary

```text
UI capture rectangle
        ↓
host maps through preview transform
        ↓
exact oriented camera ROI
        ↓
PassportQualityGate(config="capture_viewport")
```

The model treats ROI edges as the physical capture boundary. The dev team owns FIT/FILL, sensor rotation, mirroring and screen-to-buffer mapping.

## Capture flow

```text
ROI preview → analyze_roi_preview → BestFrameSelector.push
             ↓
          shutter
             ↓
select_recent (or current ROI) → analyze_roi_final
             ↓ ACCEPT
extract_passport_page(exact selected ROI, final_result) → OCR/VLM
```

Never final-check one pixel buffer and crop a different one. Never allow pixels hidden outside the product viewport to rescue page/MRZ evidence.

Previously cropped digital passport images shown inside the ROI are not the physical-truncation contract; uploaded-image completeness would be a separate product requirement.
