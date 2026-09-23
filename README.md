# Passport Quality Gate

**Pre-OCR passport capture quality assessment, capture guidance, recent-best-frame selection, and passport-page handoff.**

![SDK](https://img.shields.io/badge/SDK-v0.1.3-blue)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![Quality Policy](https://img.shields.io/badge/Policy-FP2--GOLDEN--ACTUAL-success)
![Status](https://img.shields.io/badge/Status-Dev%20Handoff%20Candidate-orange)

`passport_quality_gate` is a Python reference SDK that evaluates whether a passport camera frame is suitable for downstream OCR/VLM processing.

The current release is **v0.1.3**. Its quality behavior remains based on the frozen policy:

```text
FP2-GOLDEN-ACTUAL
```

v0.1.3 does **not** introduce a new quality policy. It keeps the 13-file Golden core unchanged and adds integration-layer correctness fixes plus a passport-page crop handoff for downstream OCR.

> This is an integration/reference SDK candidate. It does not perform document authenticity verification, does not guarantee OCR correctness, and has not yet been production-validated across the full target device/passport population.

---

## End-to-end role

```text
Camera / app preview
        │
        ▼
PassportQualityGate.analyze_preview(...)
        │
        ├── NOT READY → guidance_code → app UI
        │
        └── READY
               │
               ▼
       BestFrameSelector
               │
               ▼
      selected full frame
               │
               ▼
PassportQualityGate.analyze_final(...)
               │
        ├── RETAKE → app UI
        │
        └── ACCEPT
               │
               ▼
     extract_passport_page(...)
               │
               ▼
 perspective-corrected passport crop
               │
               ▼
        downstream OCR / VLM
```

The final quality check is run on the **exact selected full-frame pixels**. Only after `ACCEPT` is the passport page extracted for OCR handoff.

---

## What the SDK owns

- YOLO-based passport page and MRZ localization
- geometry / positioning / crop-risk evidence
- exposure, blur, readability, glare, noise and motion evidence
- live-preview decision and guidance codes
- final ACCEPT / RETAKE analysis
- bounded RAM-only recent-best-frame selection
- perspective-corrected passport-page extraction for OCR handoff
- CPU/CUDA device selection for the Python reference runtime
- packaged detector weights and default configuration

## What the SDK does not own

The consuming application/team remains responsible for:

- camera lifecycle and camera API
- preview resolution/FPS selection
- mobile UI/UX and guide rendering
- localization of guidance strings
- shutter/button behavior
- session lifecycle and concurrency
- app storage, retention, encryption and privacy policy
- networking/backend transport
- invoking the downstream OCR/VLM
- retry/product fallback behavior when final analysis or crop extraction fails
- target-device performance profiling and mobile/runtime optimization
- production telemetry, rollout and monitoring

---

## Main capabilities

| Capability | Description |
|---|---|
| Passport localization | YOLO-based passport page and MRZ localization |
| Geometry analysis | Position, scale, frame containment, crop risk and orientation evidence |
| Exposure analysis | Excessively dark/bright capture conditions |
| Blur/readability | Passport and MRZ detail-preservation evidence |
| Glare analysis | Local highlight/reflection evidence |
| Motion analysis | Stateful preview motion evidence |
| Decision engine | Separates blockers from advisory guidance |
| Temporal preview | Stabilizes live capture decisions across frames |
| Best-frame selection | Keeps recent eligible frames in a bounded RAM buffer |
| Final quality check | Re-checks the exact frame selected for capture |
| Passport-page extraction | Perspective-corrected crop from the accepted selected frame |
| CPU / CUDA selection | `device="auto"` selects CUDA when available, otherwise CPU |
| Packaged assets | Detector weights and default configuration ship with the SDK |

---

## Installation

### Source/development install

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Wheel install

The release builder produces an installable wheel in `dist/`.

For GPU deployment, install the PyTorch build appropriate for the target CUDA environment before installing the SDK wheel.

---

## Verify the frozen quality core

```bash
python tools/verify_golden_core.py
```

Expected:

```text
Golden core OK: 13 files match FP2-GOLDEN-ACTUAL
```

Then:

```bash
python -m pytest -q
```

Do not regenerate Golden hashes simply to hide a mismatch.

---

## Basic preview API

```python
from passport_quality_gate.api import PassportQualityGate

gate = PassportQualityGate(device="auto")
guide = (0.16, 0.18, 0.68, 0.64)

result = gate.analyze_preview(
    frame_bgr,
    guide_box=guide,
    timestamp=t,
)
```

Input frame contract:

```text
numpy.uint8, BGR, H x W x 3
```

The guide uses normalized:

```text
(x, y, width, height)
```

coordinates.

---

## Public result contract

For app/server integration that only needs the stable decision contract:

```python
public = gate.analyze_preview_public(frame_bgr, guide_box=guide)
```

Stable SDK 0.1.x fields:

```python
{
    "capture_allowed": False,
    "capture_quality_state": "NOT_READY",
    "workflow_state": "ALIGNING",
    "guidance_code": "MOVE_CLOSER",
    "recommended_adjustment": {...},
    "blocking_issues": [...],
    "advisories": [...],
    "timing_ms": {
        "total": 85.2
    }
}
```

When using `BestFrameSelector` or `extract_passport_page`, keep the **full analysis result** because those utilities need diagnostics/localization that the compact public result intentionally omits.

See [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

---

## Recommended capture flow

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

gate = PassportQualityGate(device="auto")
selector = BestFrameSelector()

# Preview loop
preview = gate.analyze_preview(frame_bgr, guide, timestamp=t)
selector.push(frame_bgr, preview, timestamp=t)

# Shutter / capture action
selected = selector.select_recent(trigger_timestamp=t_click)
chosen = selected.frame if selected is not None else current_frame_bgr

final = gate.analyze_final(chosen, guide, timestamp=t_click)

if final["capture_allowed"]:
    passport_crop = extract_passport_page(chosen, final)
    # Pass passport_crop directly to downstream OCR/VLM.
else:
    # Ask the user to retake according to final guidance.
    ...
```

Do not silently replace a crop failure with the full camera frame unless the product team explicitly chooses and validates that fallback.

---

## Best recent frame selection

`BestFrameSelector` is optional but recommended for shutter-time robustness.

Default behavior:

- retains only preview frames marked `capture_allowed=True`
- RAM-only and bounded
- default recent window approximately `750 ms`
- checks geometry consistency against the latest frame
- ranks using existing Golden quality metrics
- gives only a small recency preference
- writes nothing to disk

Timestamps must be finite and strictly increasing within a session. Clear the selector when starting a new session/document.

See [`docs/BEST_FRAME_SELECTION.md`](docs/BEST_FRAME_SELECTION.md).

---

## Passport-page extraction

After the selected full frame passes final analysis:

```python
from passport_quality_gate.capture_output import extract_passport_page

passport_crop = extract_passport_page(selected_full_frame, final_result)
```

The helper uses the final passport polygon and, when available, MRZ localization to orient the passport page. The returned value is an in-memory BGR `numpy.ndarray`.

The production handoff to OCR should use this in-memory crop. Writing JPEG files is only a demo/debug behavior in the example scripts.

---

## Session lifecycle

Live preview is stateful.

Use one `PassportQualityGate` instance per live capture stream/session. For a new document/session or after a meaningful pause/resume boundary:

```python
gate.reset()
selector.clear()
```

Do not share one stateful gate instance across unrelated concurrent users.

---

## Runtime information

```python
print(gate.runtime_info())
```

Example:

```python
{
    "sdk_candidate_version": "0.1.3",
    "quality_policy": "FP2-GOLDEN-ACTUAL",
    "device": "cpu",
    "production_validated": False
}
```

---

## Examples

Analyze one image:

```bash
python examples/analyze_image.py image.jpg --device auto
```

Runtime check:

```bash
python examples/runtime_check.py
```

Reference webcam integration:

```bash
python examples/webcam_demo.py --source 1 --device auto
```

Best-frame/crop validation harness:

```bash
python examples/best_frame_compare.py --source 1 --device auto
```

The webcam scripts are integration examples, not production camera implementations.

---

## Repository structure

```text
passport_quality_gate/
├── src/passport_quality_gate/
│   ├── api.py
│   ├── analyzer.py
│   ├── localization.py
│   ├── geometry.py
│   ├── quality.py
│   ├── readability.py
│   ├── motion.py
│   ├── decision.py
│   ├── frame_selector.py
│   ├── capture_output.py
│   └── assets/
├── configs/
├── models/
├── examples/
├── tests/
├── tools/
├── docs/
├── GOLDEN_CORE_SHA256.json
├── pyproject.toml
├── VERSION
└── README.md
```

---

## Frozen Golden policy

The current approved quality baseline is:

```text
FP2-GOLDEN-ACTUAL
```

Its core is fingerprinted by:

```text
GOLDEN_CORE_SHA256.json
```

SDK/API/examples/integration infrastructure may evolve without changing these fingerprints.

v0.1.3 keeps the Golden quality policy unchanged while adding integration-layer fixes and OCR handoff infrastructure.

---

## Known limitations

Current known limitations include:

- partial document truncation can occasionally pass readiness checks
- corner/completeness evidence can be imperfect under some backgrounds/viewing conditions
- small OCR-critical glare, including MRZ glare, can occasionally be under-detected
- strong motion can cause preview instability
- quality thresholds are not yet calibrated against a large real-world passport/OCR-success dataset
- detector/corner reliability may degrade with extreme blur, rotation, cropping or perspective
- target mobile-device latency/thermal behavior has not yet been broadly characterized
- the SDK does not establish document authenticity or guarantee downstream OCR correctness

These are documented limitations, not hidden assumptions.

See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

---

## Documentation

| Document | Purpose |
|---|---|
| [API Contract](docs/API_CONTRACT.md) | Stable SDK integration contract |
| [Integration Guide](docs/INTEGRATION_GUIDE.md) | End-to-end app integration |
| [Best Frame Selection](docs/BEST_FRAME_SELECTION.md) | Rolling-buffer selection behavior |
| [Output and Storage](docs/OUTPUT_AND_STORAGE.md) | In-memory output and persistence boundaries |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Current technical limitations |
| [Developer Handoff](docs/DEVELOPER_HANDOFF.md) | Team responsibilities and minimal integration |
| [Team Dev Checklist](docs/TEAM_DEV_CHECKLIST.md) | Work intentionally left to the consuming team |
| [Manager Review](docs/MANAGER_REVIEW_v0.1.3.md) | What changed, evidence, and review scope |
| [Release Validation](docs/RELEASE_VALIDATION.md) | Release verification status/checklist |

---

## Release status

| Item | Current state |
|---|---|
| SDK | `v0.1.3` |
| Quality policy | `FP2-GOLDEN-ACTUAL` |
| Golden core | Frozen; 13-file hash manifest |
| Integration fixes | Included |
| Best-frame selector | Included |
| Passport-page OCR handoff | Included |
| Production mobile integration | Dev-team responsibility |
| Downstream OCR integration | Next integration phase |
| Production validation | Not yet claimed |

The next phase should focus on application integration, OCR-linked evaluation, and target-device validation rather than further uncontrolled quality-threshold tuning.
