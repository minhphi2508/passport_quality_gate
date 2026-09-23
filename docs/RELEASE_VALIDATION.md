# Release validation — SDK 0.1.2

- Golden core verification: **PASS** — 13 files match `FP2-GOLDEN-ACTUAL`.
- Clean-source suite in release build workspace: **106 passed, 1 skipped**. The one skipped test targets the intentionally omitted legacy `run_webcam_v3.py` harness, not SDK runtime logic.
- Packaged asset parity: **PASS** — packaged thresholds/model are byte-identical to Golden source-tree assets.
- Wheel dependency metadata: **PASS** — `ultralytics>=8.3,<9` is declared alongside NumPy/OpenCV/PyYAML.
- Wheel build: **PASS**.
- Previous v0.1.1 fresh-environment runtime/independence test: **PASS** after manually installing the missing detector dependency; v0.1.2 fixes that packaging gap.
- v0.1.2 fresh-environment install without manual dependency repair: **PENDING USER ACCEPTANCE**.
- Production validation claim: **NO**. Known FP2 quality-policy limitations remain documented in `KNOWN_LIMITATIONS.md`.
