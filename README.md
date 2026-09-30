# Passport Quality Gate

Python reference SDK for pre-OCR passport capture quality assessment.

The current capture-viewport candidate analyzes only the camera ROI that the host application exposes to the user. Pixels hidden outside the product capture viewport must not affect the model.

## Responsibilities

The SDK owns:

- YOLO passport-page and MRZ localization
- physical completeness and positioning evidence
- motion, blur, exposure, glare, contrast, noise, rotation and perspective checks
- stage-aware capture guidance codes
- temporal preview state
- recent-best-frame selection
- final ACCEPT / RETAKE validation
- perspective-corrected passport-page extraction for OCR

The host application owns:

- camera lifecycle and preview rendering
- the on-screen capture rectangle and grey outside-mask
- UI-to-camera ROI mapping
- orientation / mirroring integration
- localized copy and product UX
- storage, privacy, networking and downstream OCR invocation

## Installation

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
```

The reference runtime supports Python 3.11 and 3.12.

## Capture-viewport usage

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

# The host app supplies only product-visible ROI pixels.
gate = PassportQualityGate(config="configs/capture_viewport.yaml", device="auto")
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

For details, see [`docs/CAPTURE_VIEWPORT.md`](docs/CAPTURE_VIEWPORT.md) and [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

## Reference webcam harness

```bash
python examples/webcam_capture.py --source 0 --device auto \
  --viewport 0.16 0.18 0.68 0.64 \
  --metrics outputs/capture_metrics.jsonl
```

The harness greys the area outside the viewport and submits only ROI pixels to the SDK.

## Validation

```bash
python -m pytest -q
python -m compileall -q src examples tools tests
```

The capture-quality policy is not an authenticity check and does not guarantee OCR correctness. Target-camera calibration and production-device performance validation remain required.

## Repository layout

```text
passport_quality_gate/
├── src/passport_quality_gate/
│   ├── api.py
│   ├── analyzer.py
│   ├── capture_policy.py
│   ├── capture_output.py
│   ├── decision.py
│   ├── frame_selector.py
│   ├── geometry.py
│   ├── localization.py
│   ├── motion.py
│   ├── quality.py
│   ├── readability.py
│   └── viewport.py
├── configs/
│   └── capture_viewport.yaml
├── docs/
├── examples/
├── tests/
└── tools/
```
