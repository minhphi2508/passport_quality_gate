# Release Checklist — v0.1.4

1. `python tools/verify_release_core.py`
2. `python -m pytest -q`
3. `python -m compileall -q src examples tools tests`
4. `git diff --check`
5. Confirm manual camera acceptance on the release candidate.
6. `python tools/build_sdk_package.py`
7. Verify `SHA256SUMS.txt`.
8. Inspect the wheel for detector weights, `defaults.yaml`, and `capture_viewport.yaml`.
9. Install the wheel once in an isolated acceptance environment and run `tools/acceptance_check.py`.
10. Merge the release-prep branch to `main`, then tag `v0.1.4`.

The historical FP2 Golden manifest remains in Git history. Do not recreate or rewrite it to disguise V4 changes.
