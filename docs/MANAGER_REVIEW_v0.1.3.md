# Manager Review — Passport Quality Gate SDK v0.1.3

## Purpose

This package is the current handoff candidate for pre-OCR passport capture quality assessment.

The main goal is to prevent clearly poor camera captures from reaching OCR and to guide the user toward a better frame while preserving a small recent-frame buffer for shutter-time robustness.

## What changed since v0.1.2

v0.1.3 keeps the same frozen quality policy:

```text
FP2-GOLDEN-ACTUAL
```

No detector weights or Golden quality thresholds were intentionally changed.

Integration-layer changes:

- final `ACCEPT` now maps correctly to `capture_allowed=True`
- final analysis uses state isolated from live preview
- selector timestamp/config validation is stricter
- configuration accepts Mapping-like inputs safely
- recent-best selected frame is final-checked
- after final ACCEPT, the detected passport page can be perspective-cropped for OCR handoff
- full camera frame is no longer the preferred OCR handoff image

## Why the crop was added

The full camera frame contains unnecessary background. The preferred downstream input is now the detected passport data page extracted from the accepted selected frame.

The full frame can remain available only for temporary debug/reference use if product/privacy policy allows it.

## Current boundary

This SDK is not the final mobile application.

The dev team still needs to implement:

- real camera lifecycle
- UI/UX and localized guidance
- app session/reset behavior
- connection to the existing OCR/VLM
- storage/privacy/network policy
- target-device performance testing
- deployment/runtime optimization

See `DEVELOPER_HANDOFF.md` and `TEAM_DEV_CHECKLIST.md`.

## Known unresolved quality limitations

The research cycle intentionally stopped before adding more unvalidated heuristics.

Known limitations include:

- some partial crop/completeness false accepts
- occasional small OCR-critical glare misses
- motion-related preview instability in harder conditions
- lack of large OCR-linked real-world calibration

These should be measured during the next app/OCR integration phase rather than hidden by additional threshold tuning.

## Recommended review decision

Review v0.1.3 as:

```text
Frozen FP2 quality baseline
+ integration correctness fixes
+ bounded best-frame selection
+ passport-page OCR handoff
```

If approved, the dev team should integrate the wheel/reference flow first, then return OCR-linked and target-device evidence before any proposed Golden quality-policy change.
