# Changelog

## 0.1.2
- Golden quality policy remains byte-identical: `FP2-GOLDEN-ACTUAL`.
- Fix wheel dependency metadata: declare `ultralytics>=8.3,<9` so a fresh environment resolves the detector runtime.
- Keep deployment-specific PyTorch/CUDA selection documented; preinstalled compatible PyTorch is reused.
- Add `runtime_info()` as a convenience alias of `metadata` for integration/acceptance scripts.
- Correct API documentation: full-frame `analyze_preview` / `analyze_final` require `guide_box`; cropped-page analysis uses `analyze_document_crop`.
- Add repeatable release-metadata and fresh-environment acceptance checks.

## 0.1.1
- Golden quality policy remains byte-identical: `FP2-GOLDEN-ACTUAL`.
- Package default detector/config assets inside the installable Python package.
- Add compact public-result helpers while preserving full diagnostic results.
- Add runtime diagnostics for CPU/CUDA/package checks.
- Add opt-in best-frame comparison harness; no continuous disk output.
- Expand developer handoff, installation and release-checklist documentation.

## 0.1.0
- First reference SDK integration candidate around frozen FP2 Golden.
- Public API wrapper, device auto-selection, bounded recent best-frame selector, cleaned output behavior and reference examples.
