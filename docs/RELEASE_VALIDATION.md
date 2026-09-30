# Release Validation — v0.1.4

## Accepted behavior

The V4 capture-viewport candidate completed manual camera acceptance by the model owner before release preparation. After the cleanup pass, the owner also completed the requested smoke re-check on the same Python 3.12 development environment and reported completion without a new behavior issue.

No quality thresholds or decision policy were intentionally changed during release preparation. Release-prep changes are packaging, API-profile resolution, documentation, asset de-duplication and release verification.

## Release-prep checks in this build environment

- `python tools/verify_release_core.py`: PASS (28-file release fingerprint)
- `python -m compileall -q src examples tools tests`: PASS
- `git diff --check`: PASS
- regression subset: 164 passed, 1 skipped, 3 deliberately deselected
- wheel build: PASS
- wheel installed from the generated artifact with `--no-deps`: PASS
- packaged `capture_viewport.yaml`, `defaults.yaml` and detector weights: present
- installed runtime metadata: `0.1.4` / `V4-CAPTURE-VIEWPORT`

The three deselected checks are environment-specific here: two require Ultralytics, which is not installed in this container, and one synthetic MRZ-glare assertion has the previously observed OpenCV/environment discrepancy. They are not silently converted into passes; the model owner's Python 3.12 environment is the acceptance environment for the complete suite and camera behavior.

## Scope

`production_validated` remains `False`. The release is an integration candidate. It does not claim broad target-device/passport-population validation, document authenticity, or guaranteed OCR correctness.
