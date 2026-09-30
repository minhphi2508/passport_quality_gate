# Dev Team Integration Checklist — SDK 0.1.4

- [ ] Verify `SHA256SUMS.txt`.
- [ ] Install the v0.1.4 wheel in the existing dev environment.
- [ ] Confirm `runtime_info()["sdk_candidate_version"] == "0.1.4"`.
- [ ] Instantiate with `config="capture_viewport"` and confirm policy `V4-CAPTURE-VIEWPORT`.
- [ ] Map the visible UI rectangle to the exact oriented camera ROI.
- [ ] Confirm pixels outside the UI viewport never enter model input.
- [ ] Use one gate + selector per capture session and monotonic timestamps.
- [ ] Push exact ROI pixels into `BestFrameSelector`.
- [ ] Final-check the exact selected ROI.
- [ ] Only crop after final ACCEPT; handle crop failure explicitly.
- [ ] Map `guidance_code` to localized copy; do not expose internal diagnostic terminology.
- [ ] Reset gate + selector on new document/session, meaningful pause/resume or viewport geometry change.
- [ ] Profile latency/RAM/thermal behavior on target devices.
- [ ] Keep a RETAKE/retry path because ACCEPT is not an OCR guarantee.
