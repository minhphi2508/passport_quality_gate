# Changelog

## v0.1.4
### Added
- Capture-viewport profile packaged with the wheel and available as `config="capture_viewport"`.
- ROI-only preview/final APIs and viewport mapping helpers for host applications.
- Stage-aware guidance, MRZ presence/completeness checks, page-relative motion evidence and local MRZ glare handling from the accepted V4 candidate.
- Release-core fingerprint and wheel acceptance tooling.

### Integration contract
- The host application owns camera/UI/UX and must pass the exact product-visible capture ROI.
- Pixels hidden outside that ROI are not part of model input and must not influence the decision.
- Best-frame selection, final validation and OCR crop all operate on the same ROI pixel domain.

### Compatibility
- Legacy FP2 full-frame/guide APIs remain available for existing integrations.
- The historical FP2 Golden manifest remains available in Git history/tags; it is not repurposed as the V4 release manifest.

## v0.1.3
### Added
- Best-frame capture handoff now extracts the detected passport page for downstream OCR.
- Full selected frame remains available as optional debug/reference output.

### Fixed
- Correct final ACCEPT/RETAKE capture aliases in the public SDK wrapper.
- Final analysis no longer mutates live-preview temporal state.
- BestFrameSelector rejects invalid and out-of-order timestamps.
- Improved configuration Mapping compatibility.

### Quality policy
- No changes to FP2-GOLDEN-ACTUAL.
- All 13 Golden core fingerprints remain unchanged.

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
