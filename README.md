# Passport Quality Gate

Pre-OCR passport capture quality assessment SDK for live camera capture.

**Current release:** `v0.1.4`  
**Capture policy:** `V4-CAPTURE-VIEWPORT`  
**Python:** `3.11 / 3.12`

---

## 1. What this project does

`passport_quality_gate` evaluates whether a passport image from a live camera is suitable for downstream OCR/VLM processing.

The SDK analyzes the visible passport capture area and decides whether the current image should be:

- accepted for capture;
- rejected and retried;
- or corrected with guidance such as:
  - move closer;
  - move left / right / up / down;
  - hold steady;
  - reduce rotation / perspective;
  - improve lighting;
  - make the bottom text area visible.

The SDK also keeps a short rolling buffer of recent good frames so that a slightly worse shutter-time frame can be replaced by a better frame captured shortly before it.

After final acceptance, the passport page can be perspective-corrected and passed directly to downstream OCR.

---

# 2. Current product assumption

The current implementation is built around the existing product UI:

```text
+------------------------------------------+
| grey / unused area                       |
|                                          |
|      +----------------------------+      |
|      |                            |      |
|      |   visible capture area     |      |
|      |                            |      |
|      +----------------------------+      |
|                                          |
| grey / unused area                       |
+------------------------------------------+
```

Only pixels inside the visible capture rectangle should be analyzed.

The expected production flow is:

```text
Camera frame
    ↓
App maps visible UI rectangle to camera pixels
    ↓
Exact oriented capture ROI
    ↓
Passport Quality Gate
```

The SDK treats the edges of that ROI as the actual capture boundaries.

Pixels hidden by the grey area must **not** be supplied to the model as usable image content.

See:

```text
docs/CAPTURE_VIEWPORT.md
```

for the full integration contract.

---

# 3. Technical approach

The runtime is intentionally lightweight.

There is only **one learned detection model**:

```text
YOLO
 ├── passport_page
 └── mrz
```

The same detector instance is reused throughout the capture session.

The remaining quality analysis mainly uses classical image processing, geometry, rule-based logic and temporal analysis rather than additional neural networks.

These checks include:

```text
Passport / MRZ localization
        ↓
Geometry / completeness
        ↓
Position / scale
        ↓
Rotation / perspective
        ↓
Exposure
        ↓
Blur / readability
        ↓
Glare
        ↓
Contrast / noise
        ↓
Motion
        ↓
Temporal stabilization
        ↓
READY / BLOCK + guidance
```

This design keeps runtime cost relatively low and avoids loading multiple ML models during live capture.

---

# 4. Repository structure

```text
passport_quality_gate/
│
├── src/passport_quality_gate/
│   ├── api.py
│   ├── analyzer.py
│   ├── capture_policy.py
│   ├── decision.py
│   ├── localization.py
│   ├── geometry.py
│   ├── quality.py
│   ├── readability.py
│   ├── motion.py
│   ├── viewport.py
│   ├── frame_selector.py
│   ├── capture_output.py
│   └── assets/
│       ├── passport_detector_ver3_best.pt
│       └── capture_viewport.yaml
│
├── configs/
│   └── capture_viewport.yaml
│
├── examples/
│   ├── webcam_capture.py
│   ├── analyze_image.py
│   └── runtime_check.py
│
├── tests/
├── tools/
├── docs/
│
├── RELEASE_CORE_SHA256.json
├── requirements.txt
├── pyproject.toml
├── VERSION
└── README.md
```

Detector weights and the production capture profile are packaged with the SDK.

No separate model download is required for the normal packaged release.

---

# 5. Fastest way to run the project

## First-time setup only

Clone the repository:

```bash
git clone https://github.com/minhphi2508/passport_quality_gate.git
cd passport_quality_gate
```

Use Python 3.11 or 3.12.

### Windows

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### Git Bash

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project once:

```bash
python -m pip install -e ".[dev]"
```

You do **not** need to reinstall dependencies every time you run the project.

On later sessions, simply activate the same `.venv`.

---

# 6. Verify the installation

Run:

```bash
python examples/runtime_check.py
```

Then verify the release fingerprint:

```bash
python tools/verify_release_core.py
```

Expected release:

```text
SDK: 0.1.4
Policy: V4-CAPTURE-VIEWPORT
```

The release verifier should report that the release-core files match the v0.1.4 fingerprint.

---

# 7. Run the webcam demo

The main manual test harness is:

```bash
python examples/webcam_capture.py --source 0 --device auto
```

If camera `0` is not the correct camera:

```bash
python examples/webcam_capture.py --source 1 --device auto
```

The default normalized capture viewport is:

```text
x = 0.16
y = 0.18
w = 0.68
h = 0.64
```

A custom viewport can be supplied:

```bash
python examples/webcam_capture.py \
  --source 0 \
  --device auto \
  --viewport 0.16 0.18 0.68 0.64
```

### Webcam controls

```text
D  → toggle diagnostics
C  → capture best recent READY frame
F  → final-check the exact current ROI
R  → reset capture session
Q  → quit
```

`C` represents the recommended capture flow because it allows `BestFrameSelector` to recover a better recent frame.

`F` is mainly useful for testing the exact current frame.

---

## OpenCV GUI note

The SDK itself depends on:

```text
opencv-python-headless
```

because the production SDK does not require a desktop GUI.

However:

```text
examples/webcam_capture.py
```

uses `cv2.imshow()` for the development/demo window.

If OpenCV reports that GUI support is unavailable, install a GUI-capable OpenCV build in your local development environment.

This is only required for the desktop webcam demo, not for the production SDK itself.

---

# 8. Recommended production integration

Instantiate the SDK:

```python
from passport_quality_gate import (
    PassportQualityGate,
    BestFrameSelector,
    extract_passport_page,
)

gate = PassportQualityGate(
    config="capture_viewport",
    device="auto",
)

selector = BestFrameSelector()
```

The input should be:

```text
numpy.uint8
BGR
H x W x 3
```

and should contain **only the product-visible capture ROI**.

---

## Preview loop

```python
preview = gate.analyze_roi_preview(
    roi_bgr,
    timestamp=t,
    viewport_metadata=metadata,
)

selector.push(
    roi_bgr,
    preview,
    timestamp=t,
)
```

The SDK returns whether capture is currently allowed and which corrective guidance should be shown.

---

## Shutter / capture

At shutter time:

```python
selected = selector.select_recent(
    trigger_timestamp=t_click
)
```

Use the selected recent frame when available:

```python
chosen = (
    selected.frame
    if selected is not None
    else roi_bgr
)
```

Preserve the metadata belonging to the same frame:

```python
chosen_meta = (
    selected.result["capture_viewport"]
    if selected is not None
    else metadata
)
```

Run the final validation on the **exact same pixels**:

```python
final = gate.analyze_roi_final(
    chosen,
    timestamp=t_click,
    viewport_metadata=chosen_meta,
)
```

Only after final acceptance:

```python
if final["capture_allowed"]:
    passport_crop = extract_passport_page(
        chosen,
        final,
    )
```

The resulting crop can then be passed directly to OCR/VLM.

---

# 9. Full capture pipeline

```text
Live camera
    ↓
UI capture rectangle
    ↓
UI → camera coordinate mapping
    ↓
Exact oriented ROI
    ↓
analyze_roi_preview()
    ↓
READY?
 ┌──┴───────────────┐
 │                  │
NO                 YES
 │                  │
guidance      BestFrameSelector.push()
                    │
                    ↓
                  shutter
                    │
                    ↓
        select_recent()
                    │
                    ↓
        analyze_roi_final()
             ┌──────┴──────┐
             │             │
          RETAKE         ACCEPT
             │             │
          guidance          ↓
                   extract_passport_page()
                           │
                           ↓
                        OCR / VLM
```

Important rule:

> The frame that passes final validation must be the same pixel buffer used for passport extraction.

Do not validate one frame and then crop another.

---

# 10. BestFrameSelector

`BestFrameSelector` stores only recent frames that were already considered capture-allowed.

Default behavior:

```text
recent window     ≈ 750 ms
maximum age       ≈ 900 ms
maximum frames    = 12
memory budget     ≈ 96 MB
```

Frames are ranked using existing quality evidence such as:

- blur;
- resolution;
- exposure;
- motion;
- contrast;
- glare;
- noise;
- localization confidence;
- recency.

The selector keeps frames in RAM only.

It does not write images to disk.

---

# 11. Session lifecycle

The analyzer contains temporal state.

Use:

```text
one PassportQualityGate
+
one BestFrameSelector
```

per live capture session.

Use monotonic timestamps.

Reset both when:

- a new passport/document starts;
- the camera session restarts;
- there is a meaningful pause/resume;
- the capture viewport changes.

```python
gate.reset()
selector.clear()
```

Do not share the same stateful instance across unrelated simultaneous capture sessions.

---

# 12. Public result

Typical integration fields include:

```text
capture_allowed
capture_quality_state
workflow_state
guidance_code
guidance_text
recommended_adjustment
blocking_issues
advisories
timing_ms.total
```

For a compact external result:

```python
from passport_quality_gate import to_public_result

public = to_public_result(result)
```

Keep the **full result** when using:

- `BestFrameSelector`;
- debug diagnostics;
- passport crop extraction.

See:

```text
docs/API_CONTRACT.md
```

for the API contract.

---

# 13. Responsibilities

## SDK responsibility

The SDK handles:

- passport-page localization;
- MRZ localization;
- capture completeness evidence;
- positioning;
- scale;
- blur;
- exposure;
- glare;
- contrast;
- noise;
- motion;
- rotation;
- perspective;
- temporal stabilization;
- stage-aware guidance codes;
- best recent frame selection;
- final ACCEPT / RETAKE;
- perspective-corrected passport extraction.

---

## Host application responsibility

The consuming application owns:

- camera lifecycle;
- camera API;
- screen UI;
- the visible capture rectangle;
- grey outside-mask;
- screen-to-camera coordinate mapping;
- FIT / FILL / center-crop handling;
- sensor rotation;
- mirroring;
- user-facing text;
- animation / UX;
- localization;
- storage;
- privacy;
- network communication;
- downstream OCR/VLM;
- retry/fallback UX.

A particularly important integration requirement is:

> The ROI seen by the SDK must correspond exactly to the area users believe they are capturing.

Incorrect UI-to-camera coordinate mapping will invalidate positioning and completeness decisions.

---

# 14. Testing

## Release fingerprint

```bash
python tools/verify_release_core.py
```

---

## Automated tests

```bash
python -m pytest -q
```

---

## Syntax / import check

```bash
python -m compileall -q src examples tools tests
```

---

## Runtime demo

```bash
python examples/webcam_capture.py --source 0 --device auto
```

Manual camera testing remains important because many capture problems depend strongly on real:

- lighting;
- reflections;
- backgrounds;
- cameras;
- preview transforms;
- physical passport positioning.

---

# 15. Current validation status

Version `v0.1.4` is the current integration candidate.

The V4 capture-viewport behavior has gone through repeated automated and manual testing.

Release-preparation checks include:

```text
Release-core verification      PASS
Python compile check           PASS
Wheel build                    PASS
Wheel installation smoke test  PASS
Manual camera testing          completed
```

See:

```text
docs/RELEASE_VALIDATION.md
```

for the detailed release record.

`production_validated` remains:

```python
False
```

because the project has not yet been validated across a sufficiently broad real-world combination of:

- passport samples;
- devices;
- cameras;
- lighting environments;
- backgrounds;
- OCR outcomes.

---

# 16. Known limitations

The current implementation should not be considered perfect or fully production-calibrated.

Important remaining areas include:

### Directional guidance

Some directional guidance cases, particularly certain:

```text
MOVE_LEFT
MOVE_DOWN
```

situations still require more real-camera testing and may require further tuning.

---

### Lighting and glare

Glare/exposure handling works for the current test cases, but difficult real-world lighting remains an area that requires additional validation.

Examples include:

- small localized reflections;
- glare crossing important text;
- mixed bright/dark areas;
- very reflective passport surfaces;
- different camera exposure behavior.

---

### Completeness

Borderline partial passport truncation can still be difficult.

Completeness reasoning depends strongly on:

- detector quality;
- visible page geometry;
- physical ROI boundary;
- background;
- camera viewpoint.

---

### Detector limitations

Passport-page and MRZ detection can degrade under:

- extreme blur;
- large rotation;
- extreme perspective;
- occlusion;
- unusual backgrounds;
- very poor illumination.

---

### Threshold calibration

Current quality thresholds have not yet been calibrated against a large real-world dataset linked directly to downstream OCR success.

The next useful validation phase is therefore:

```text
real devices
+
real passports
+
real capture conditions
+
OCR outcome
```

rather than continued tuning only on synthetic or monitor-based cases.

---

### Uploaded / already-cropped images

The V4 completeness contract is designed for **live camera capture**.

A passport image that was already digitally cropped before being placed inside the capture ROI is a different problem.

The current SDK does not claim reliable physical-completeness detection for arbitrary pre-cropped digital passport images.

---

### Authenticity

This SDK does **not** determine whether a passport is genuine.

It is a capture-quality gate only.

---

### OCR

An `ACCEPT` result means the image passed the current capture-quality policy.

It does **not** guarantee that downstream OCR will be correct.

---

# 17. Analyze a single image

A simple image utility is also included:

```bash
python examples/analyze_image.py passport.jpg --device auto
```

Optional JSON output:

```bash
python examples/analyze_image.py passport.jpg \
  --device auto \
  --json-out output.json
```

Note that the main V4 product contract is the live-camera ROI workflow described above.

---

# 18. CPU / GPU

Use:

```python
device="auto"
```

for normal operation.

The SDK will use CUDA when available and supported, otherwise CPU.

Example:

```python
gate = PassportQualityGate(
    config="capture_viewport",
    device="auto",
)
```

Force CPU:

```python
device="cpu"
```

The architecture intentionally avoids loading multiple vision models.

---

# 19. Release integrity

The accepted runtime/package files are fingerprinted in:

```text
RELEASE_CORE_SHA256.json
```

Verify them using:

```bash
python tools/verify_release_core.py
```

Do not modify the manifest simply to hide unexpected runtime changes.

When quality logic changes intentionally, treat that as a new validated release rather than silently replacing the current release fingerprint.

---

# 20. Main documentation

| Document | Purpose |
|---|---|
| `docs/CAPTURE_VIEWPORT.md` | Product-visible ROI contract |
| `docs/API_CONTRACT.md` | SDK input/output API |
| `docs/INTEGRATION_GUIDE.md` | Recommended integration flow |
| `docs/DEVELOPER_HANDOFF.md` | Developer handoff |
| `docs/TEAM_DEV_CHECKLIST.md` | Integration checklist |
| `docs/KNOWN_LIMITATIONS.md` | Current technical limitations |
| `docs/RELEASE_VALIDATION.md` | v0.1.4 validation record |
| `docs/INSTALLATION.md` | Installation notes |

---

# 21. Recommended next phase

The current SDK is ready for integration testing.

The recommended next work is:

```text
SDK integration
      ↓
real camera/device testing
      ↓
collect difficult cases
      ↓
compare ACCEPT/RETAKE with OCR outcome
      ↓
targeted calibration / fixes
```

Further threshold changes should preferably be driven by real failure cases rather than additional uncontrolled tuning.

---

# 22. Release

Current stable project checkpoint:

```text
v0.1.4
```

Repository:

```text
https://github.com/minhphi2508/passport_quality_gate
```

For reproducible review or integration, use the tagged release rather than an arbitrary future commit on `main`.