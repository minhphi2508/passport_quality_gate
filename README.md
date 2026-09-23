# Passport Quality Gate SDK — v0.1.2 Integration Candidate

Quality policy: **FP2-GOLDEN-ACTUAL** (frozen)  
SDK version: **0.1.2**

Python reference SDK for passport capture-quality assessment. It is an integration candidate, not a claim of production validation or document authenticity/OCR correctness.

## SDK boundary

The SDK analyzes frames and returns machine-readable readiness/guidance. It does **not** own camera capture, UI, localization strings, shutter controls, storage, networking, or a fixed input resolution.

```python
from passport_quality_gate.api import PassportQualityGate

gate = PassportQualityGate(device="auto")
result = gate.analyze_preview_public(frame, guide_box=(0.16, 0.18, 0.68, 0.64))
# compact, JSON-ready stable integration fields
```

Use one gate instance per live capture stream/session; call `reset()` when the document/session changes.

## Recent best-frame utility

```python
from passport_quality_gate.frame_selector import BestFrameSelector

selector = BestFrameSelector()
selector.push(frame, result, timestamp=t)
best = selector.select_recent(trigger_timestamp=click_t)
```

It keeps a bounded RAM buffer only; it never writes frames to disk. The default window is 750 ms and is configurable.

## Install

Use Python 3.11/3.12 for the current reference environment. For source-tree development, install a PyTorch build appropriate for the target CPU/GPU first, then install the package:

```bash
python -m pip install -e .
```

The installable wheel declares the YOLO runtime (`ultralytics`) as a dependency. On GPU/server targets, preinstall the desired PyTorch/CUDA build before installing the wheel so pip reuses that compatible build.

Do not install `environment_GOLDEN.txt` as requirements; it is an audit snapshot and contains an old editable path.

## Verify frozen policy

```bash
python tools/verify_golden_core.py
python -m pytest -q
```

## Examples

```bash
python examples/analyze_image.py image.jpg --device auto
python examples/webcam_demo.py --source 1 --device auto
```

The webcam program is a reference integration harness only; its resolution/backend defaults are not SDK requirements.

See `docs/` for the API contract, integration notes, deployment notes, best-frame behavior, and known limitations.


For handoff boundaries and deployment notes, see `docs/DEVELOPER_HANDOFF.md` and `docs/INSTALLATION.md`.
