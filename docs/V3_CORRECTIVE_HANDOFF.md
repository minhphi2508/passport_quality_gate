# V3 corrective handoff

Branch: `research/vnext-corrective-v3`, continuing V2 commit `93923f660d5effde2822652f683958815a9b72be`; original main `b0c3f4d6d4d09d684626fb9b9a8094add7b8c633`. Implements the supplied V3 PDF plus the stage-aware addendum. No restart, release/tag change, new dependency, model, OCR, or second YOLO pass.

## What changed

- **Localization prerequisites:** an absent or below-threshold page always gives `PLACE_PASSPORT_IN_FRAME`, with no secondary adjustment. Diagnostics remain available internally. Once the page is reliable, page-level geometry, motion and quality may supply actions. Full camera pixels remain the localization input; the guide is a composition target only.
- **YOLO authority:** MRZ presence requires the current YOLO confidence threshold. Classical two-row detection is telemetry only, never readiness authority or expected-boundary evidence. Preview holds previously established presence for at most 0.4 s with a geometry continuity check; reset, loss of reliable page, or a large jump invalidates the hold. Final has no hold and immediately RETAKEs an absent/below-threshold YOLO MRZ.
- **Debounce without stale diagnosis:** an isolated miss can retain previously observed text evidence within the same hold window. The current MRZ polygon stays absent; no MRZ-local glare/readability measurement or inferred expected bounds is made from a cached rectangle. `text_detail_current_observed` and `text_detail_held` distinguish the evidence. Current page defects still block. An expired miss prompts “Make sure the two lines of text at the bottom are visible,” unless a clearer upstream cause such as strong motion or a confirmed physical cut owns guidance.
- **Causal guidance:** low resolution maps to MOVE_CLOSER; strong/recent motion plus detail/MRZ loss maps to HOLD_STEADY. A lost/unreliable page still obeys the placement prerequisite. Independent strong content crop, harmful glare or severe exposure can outrank motion. Generic uncertainty is never the user-facing instruction. Plain `guidance_text` is added for opt-in consumers and used on the primary webcam overlay; technical codes remain under D/debug and in JSONL.
- **Small physical cuts:** multiple compact contrast/ink components that touch a real camera boundary contribute independent content-at-cut evidence. Fusion is `max(geometry_cut, content_cut)`, with semantic left/right/top/bottom labels. Blank margins and uniform border lines are negative controls. Existing global side thresholds and motion thresholds are unchanged.
- **Glare correction:** spatial MRZ overlap/component geometry must be supported by local optical damage. Cells require contrast and stroke loss, with adjacent text or temporal corroboration. Bright intact text cannot block solely from candidate geometry/reliability floors. Existing body glare paths remain.
- **Runtime metric:** Linux memory sampling uses `/proc/self/status` to avoid ambiguous PID namespaces, with portable fallbacks. This affects telemetry only.

Default FP2 processing, worker scheduling, BestFrameSelector, exact selected full-frame final validation, crop-after-ACCEPT and opt-in-only image recording remain. Display FPS is independent of analysis throughput. A preview frame cannot certify a newer unanalysed camera frame.

## Validation

`python -m pytest -q`: **189 passed in 37.34 s**. Includes all 126 original tests unchanged, 25 VNext cases updated only where V3 explicitly supersedes fallback/glare policy, and 38 corrective/addendum cases. Coverage includes isolated/sustained detection misses, final authority, fake fallback, page confidence prerequisites, held-evidence separation, causal motion, content cuts on all sides, blank controls, bright intact text, destructive glare, guide-only mismatch and public text mapping. Existing selected-frame/final/crop tests pass unchanged.

`python -m compileall -q src examples tools tests` and `git diff --check` pass. GOLDEN_CORE_SHA256.json, baseline configs/defaults, VERSION, release metadata and weights remain unchanged. The research branch intentionally differs in analyzer.py and decision.py from the frozen manifest; do not regenerate it. The latter change is inherited from V2.

Fixtures are deterministic generated proxies, not real passport photos or actual monitor/camera replays. No new opt-in captured frames were supplied. Camera acceptance is pending; this is not production validation.

## Runtime

Same harness: CPU, Python 3.12, Torch 2.14.0+cpu, OpenCV 5.0.0; Torch 2 threads, OpenCV 1; 5 warm-up and 40 measured synthetic 1024x768 frames. One actual packaged YOLO inference per frame. Full distributions and environment are in the adjacent three `research_v3_*.json` files.

| Workload | p50 ms | p95 ms | p99 ms | Analysis Hz |
|---|---:|---:|---:|---:|
| V2 reference, YOLO | 125.2 | 167.6 | 187.0 | 7.55 |
| FP2, YOLO | 120.5 | 137.7 | 153.8 | 8.10 |
| V3, YOLO | 136.1 | 174.4 | 191.6 | 7.03 |
| FP2, supplied localization | 42.4 | 47.7 | 49.7 | 23.04 |
| V3, supplied localization | 54.1 | 58.4 | 59.4 | 18.25 |

V3 YOLO sampled RSS was 499.6 MiB; sequential-process FP2 was 481.7 MiB. These are not isolated model footprints. V3 median local glare / side completeness / MRZ presence cost 11.00 / 1.25 / 0.17 ms, versus V2 10.33 / 1.10 / 0.26 ms. V3 total median was 10.85 ms above the same-session V2 reference; localization also changed from 72.24 to 81.70 ms across runs, so the total difference is not an isolated algorithm-overhead estimate. Forced MRZ-miss telemetry costs p50/p95/p99 2.40/3.07/4.09 ms. No GPU result. Small samples and scheduler variance limit tail comparisons. This CPU does not sustain the suggested 8–15 Hz range for V3; the worker improves display responsiveness, not inference speed.

## Run and manual acceptance

```bash
python examples/webcam_research_vnext.py --source 0 --device auto --profile vnext --metrics outputs/v3_metrics.jsonl
```

Use camera 1 if appropriate. D toggles diagnostics, R resets, C selects a recent eligible full frame and validates those exact pixels before cropping, Q quits. No images saved unless `--record-images outputs/opt_in_images` is explicitly added. For FP2 A/B use `--profile fp2` and another metrics path.

P0 order: clean/steady and clear monitor image; sustained bottom-two-lines physical loss; strong shake while page remains detectable; small left/right text cuts. Also confirm page absent/unreliable gives only placement. P1: large directional cuts, far MOVE_CLOSER, very close complete page, top blank versus content trim, harmful versus harmless shine, natural jitter, bad-to-good recovery, C/final/crop.

Move content outside the **physical camera view**, not merely outside the guide. After a bottom-loss trial, reset or wait until the previous eligible BestFrame window expires before C if the purpose is to test the current bad frame: a deliberate recent-good-frame selection can validly ACCEPT an earlier complete frame. Selection metadata identifies the exact frame checked.

## Remaining limits

Content-at-cut uses compact high-contrast strokes, not semantic understanding: low-contrast content, large solid details, rotated pages, loose/incorrect detector boxes and unusual layouts can still defeat it. YOLO itself can miss or mislocalize the page/bottom text. Optical damage is heuristic; non-clipped washout and unusual security backgrounds need physical-camera controls. Bright synthetic negatives do not establish robustness across real displays. Re-test the PDF checklist on the target camera and report scenario/time plus JSONL for residual failures. No motion sensitivity tuning was introduced.
