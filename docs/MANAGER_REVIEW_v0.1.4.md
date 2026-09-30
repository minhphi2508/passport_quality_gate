# Manager Review — Passport Quality Gate v0.1.4

## Outcome

The capture-quality research branch has been reduced to a release candidate centered on one production contract: the application supplies the exact visible capture ROI, and the model evaluates only those pixels.

## Main capabilities

- passport-page + MRZ localization
- stage-aware guidance
- physical completeness within capture ROI
- MRZ presence/completeness checks
- motion/blur/exposure/glare/contrast/noise/geometry quality checks
- recent-best-frame selection
- exact selected-ROI final validation
- perspective-corrected OCR crop handoff

## Ownership boundary

Model/SDK: analysis, decision and guidance codes.
Application/dev team: camera, UI/UX, ROI mapping, user-facing copy, storage/networking and OCR invocation.

## Validation status

Manual camera acceptance and automated regression testing were completed during the V4 cycle. The release is an integration candidate, not a claim of broad production validation or document authenticity.
