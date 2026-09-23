# Known Limitations — FP2-GOLDEN-ACTUAL

The current policy is frozen because it has the best validated behavior from the present research cycle, not because it is perfect.

- partial document truncation can occasionally pass readiness checks;
- corner localization can be unreliable under some backgrounds/viewing conditions;
- MRZ glare can occasionally be under-detected;
- strong motion can cause READY / NOT_READY temporal fluctuation;
- thresholds are not calibrated against a large real-world passport/OCR-success dataset;
- current validation does not establish document authenticity or guarantee downstream OCR correctness.

These limitations are deliberately documented rather than hidden by additional unvalidated heuristics.
