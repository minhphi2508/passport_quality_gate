# Team Dev Integration Checklist — SDK 0.1.3

## Before integration

- [ ] Verify received `SHA256SUMS.txt`.
- [ ] Install the v0.1.3 wheel in a clean environment.
- [ ] Confirm `runtime_info()["sdk_candidate_version"] == "0.1.3"`.
- [ ] Confirm `runtime_info()["quality_policy"] == "FP2-GOLDEN-ACTUAL"`.
- [ ] Decide CPU/GPU/mobile deployment path.

## Camera/app integration

- [ ] Convert camera frames to BGR `uint8 HxWx3` for the Python reference SDK.
- [ ] Define normalized `(x, y, w, h)` guide rectangle.
- [ ] Choose analysis cadence appropriate for target hardware.
- [ ] Keep one gate + selector state per capture session.
- [ ] Use a monotonic timestamp source.
- [ ] Call `gate.reset()` + `selector.clear()` on new document/session and meaningful pause/resume.

## Preview/UI

- [ ] Run `gate.analyze_preview(...)`.
- [ ] Push the **full preview result** into `BestFrameSelector`.
- [ ] Map `guidance_code` to localized app copy.
- [ ] Distinguish blocker vs advisory presentation.
- [ ] Do not expose research severity numbers as probabilities.

## Capture

- [ ] On shutter, call `selector.select_recent(...)`.
- [ ] Use current frame only when there is no recent eligible candidate.
- [ ] Run final analysis on the exact selected full frame.
- [ ] If final RETAKE: keep user in capture flow.
- [ ] If final ACCEPT: call `extract_passport_page(...)`.
- [ ] Send the resulting in-memory passport crop to OCR/VLM.
- [ ] Explicitly handle crop extraction failure; do not silently fall back to full frame.

## OCR integration

- [ ] Record OCR result against the exact selected/cropped image during evaluation.
- [ ] Measure MRZ/field accuracy, not just SDK READY rate.
- [ ] Preserve retry flow when OCR fails after SDK ACCEPT.
- [ ] Do not interpret ACCEPT as authenticity verification.

## Privacy/storage

- [ ] Decide whether any passport image is persisted.
- [ ] Minimize full-frame retention.
- [ ] Define encryption/access/deletion policy.
- [ ] Review telemetry to avoid unnecessary personal-data logging.

## Target-device validation

- [ ] Good steady capture.
- [ ] Natural jitter.
- [ ] Strong shake.
- [ ] Far readable / extremely far.
- [ ] Too close.
- [ ] Left/right/top/bottom crop.
- [ ] MRZ glare and VIZ/body glare.
- [ ] Pause/resume.
- [ ] Orientation changes.
- [ ] Shutter-time motion.
- [ ] Low-memory/thermal/battery behavior.
- [ ] Different camera devices and lighting conditions.

## Do not change without review

- [ ] Do not modify the 13 Golden files casually.
- [ ] Do not regenerate `GOLDEN_CORE_SHA256.json` to hide a diff.
- [ ] Do not change detector weights/thresholds and still call the policy `FP2-GOLDEN-ACTUAL`.
- [ ] Do not add crop/glare heuristics based only on a few synthetic/manual examples.

Any future quality-policy change should be evaluated separately and versioned explicitly.
