# Release Validation — SDK 0.1.3

## Release identity

```text
SDK: 0.1.3
Quality policy: FP2-GOLDEN-ACTUAL
Git tag: v0.1.3
```

## Confirmed project-level status

- Golden quality policy intentionally unchanged from FP2.
- v0.1.3 includes public final-result alias correction.
- v0.1.3 isolates final analyzer state from preview temporal state.
- BestFrameSelector validates finite/monotonic timestamps.
- Mapping-style config input is normalized at the public wrapper boundary.
- Recent-best-frame selection remains bounded and RAM-only.
- Accepted selected frames can be converted to a perspective-corrected passport-page crop for OCR handoff.
- Manual local webcam inspection confirmed the new passport-page crops were visually correct on the tested samples.
- Crop/glare/completeness limitations remain documented; no new Golden quality policy is claimed.

## Required pre-handoff checks

Run from the v0.1.3 source tree:

```bash
python tools/verify_golden_core.py
python -m pytest -q
python examples/runtime_check.py
python tools/build_sdk_package.py
```

Expected Golden verification:

```text
Golden core OK: 13 files match FP2-GOLDEN-ACTUAL
```

Record the actual pytest result from the handoff machine rather than copying an old test count.

## Fresh-environment wheel acceptance

Create a fresh virtual environment outside the source tree, then install only the generated wheel.

For target GPU/CUDA environments, install the appropriate PyTorch build first.

Run:

```bash
python tools/acceptance_check.py
```

or, from the source/handoff bundle:

```bash
python tools/acceptance_check.py --image <test_passport_image>
```

Confirm:

- import resolves from the fresh environment's `site-packages`
- `runtime_info()` reports SDK `0.1.3`
- quality policy is `FP2-GOLDEN-ACTUAL`
- selector smoke passes
- optional image preview/final call executes
- source repository is not required at runtime
- packaged config/model assets are present

## Release artifacts

The build script should create:

```text
dist/
├── passport_quality_gate_sdk_v0.1.3.zip
├── passport_quality_gate-0.1.3-*.whl
└── SHA256SUMS.txt
```

Send all three together.

## Production-validation claim

```text
NO
```

The release is suitable for manager/dev integration review. It is not yet evidence of broad production validation across all target mobile devices, passport variants, lighting conditions, or downstream OCR outcomes.
