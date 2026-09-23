# Release checklist

Before handing a build to another team:

1. `python tools/verify_golden_core.py`
2. `python -m pytest -q`
3. `python examples/runtime_check.py`
4. `python tools/build_sdk_package.py`
5. In a **new virtual environment**, install only the generated wheel (preinstall target-specific PyTorch first only when GPU/CUDA selection requires it).
6. Run `python tools/acceptance_check.py` from the source/handoff bundle, optionally with `--image <passport_test_image>`.
7. Confirm the installed package path points to the new environment's `site-packages` and the test still runs when the source repo is unavailable.
8. Verify generated SHA256 files.
9. Keep `FP2-GOLDEN-ACTUAL` known limitations attached to the release.
10. Do not modify/freeze new hashes merely to hide a Golden-core mismatch.
