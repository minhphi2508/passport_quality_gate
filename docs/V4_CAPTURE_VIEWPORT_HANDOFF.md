# V4 capture viewport — execution handoff

Continues `research/vnext-corrective-v3` at `7c80023ab0436fd12eb5602931e953f265efc9be`, on `research/v4-capture-viewport`. The supplied V4 PDF supersedes V3's composition-only rectangle **only in the explicit V4 profile**. V3 and FP2 configurations/tests retain their original semantics.

## Contract and integration

`configs/research_v4.yaml` activates V4. Its preferred API accepts only ROI pixels:

```python
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.viewport import CaptureViewport
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

gate = PassportQualityGate(config='configs/research_v4.yaml')
selector = BestFrameSelector()
viewport = CaptureViewport(.16, .18, .68, .64)
roi, metadata = viewport.extract(oriented_camera_frame)
preview = gate.analyze_roi_preview(roi, timestamp=timestamp, viewport_metadata=metadata)
selector.push(roi, preview, timestamp=timestamp)
selected = selector.select_recent(trigger_timestamp=trigger)
if selected is not None:
    final = gate.analyze_roi_final(selected.frame, timestamp=trigger,
                                  viewport_metadata=selected.result['capture_viewport'])
    if final['capture_allowed']:
        page = extract_passport_page(selected.frame, final)
```

Convenience methods `analyze_capture_preview(frame, capture_viewport, ...)` and `analyze_capture_final(...)` explicitly crop before calling the same ROI path. `guide_box` is never silently reinterpreted. For V4 host integrations, use these explicit methods; do not pass an uncropped raw buffer to the ROI methods. Final must receive selected ROI bytes, not a newly cropped raw frame. Caller owns timestamps and viewport changes; clear its selector on changes. The included harness does so.

`CaptureViewport` is normalized `(x,y,w,h)` in the **oriented, unmirrored** camera buffer. Pixel rectangle `[x0,y0,x1,y1]` uses exclusive upper bounds. Bounds round inward (ceil start/floor end) and clamp to the buffer, rejecting ROIs below 8x8. This deliberately excludes fractional outside pixels. Extraction returns an independent contiguous copy. No resize, enhancement or exterior padding supplies evidence.

For screen coordinates, use `map_preview_viewport(screen_rect, preview_size, sensor_size, rotation_clockwise=..., mirrored=..., mode='fit'|'fill')`, then extract from `orient_camera(sensor_frame, rotation_clockwise)`. FIT/FILL scaling, centered letterboxing/cropping and preview-only mirroring are inverted explicitly. Rotation occurs before analysis. Host must supply its actual transform; this utility cannot infer platform camera transforms, lens distortion, non-centered crops or arbitrary CSS transforms. Coordinate fixtures cover 0/90/180/270, FIT/FILL and both mirror states. Metadata includes normalized/resolved rectangles, ROI/raw sizes and preview transform. A viewport/resolution change resets preview evidence.

Outside pixels never enter YOLO, geometry, quality, temporal buffers, selector ranking, selected final or crop in this flow. Debug overlays add the ROI origin back only for drawing. There is one YOLO inference per analyzed image and one shared model for preview/final.

## Evidence and guidance changes

- Page unreliable: placement only; secondary adjustments are suppressed.
- MRZ authority/debounce remains YOLO-only (0.4 s preview hold; no final hold). Classical fallback is off by default; explicit telemetry can be enabled without affecting decisions.
- A current MRZ candidate touching/crossing an ROI side is incomplete. Broad minimum width/page ratio and aspect reject clearly partial candidates. Vertical placement is diagnostic corroboration only. Incomplete candidates cannot seed debounce or MRZ-local quality; original candidate coordinates remain internal diagnostics. This is capture geometry, not authenticity/OCR validation.
- Confirmed ROI cuts outrank scale and quality. Clear sides produce opposite translation; multiple sides produce whole-page guidance. Absent/incomplete bottom text gets a plain-language instruction. Strong/recent motion explaining sudden downstream loss is the causal exception before missing-text uncertainty, while confirmed cuts still win.
- V3 content-at-cut now sees ROI boundaries. Paper-margin extrapolation alone cannot prove content loss: V4 requires independent observed shape/boundary support. Internal weak detector edges away from the ROI boundary cannot become directional viewport-cut evidence. These corrections address blank-margin false blocks without lowering global thresholds or requiring four visible page corners.
- Rotation/perspective/placement precede scale; scale precedes quality. Harmful-vs-bright glare rules remain V3. Current reliable MRZ localization is required for local MRZ diagnoses.
- V4 motion uses center displacement divided by mean consecutive page diagonal, and absolute log page-scale ratio, then the existing temporal filtering/coupling. Initial dead zones/speed thresholds convert V3 units using a stated reference page size (diagonal 0.80, area scale about 0.63 of frame); they are explicit research parameters, not claimed camera calibration. Equivalent page motion is invariant to ROI extent/resolution. Natural synthetic jitter stays below the dead zone; strong synthetic shake blocks. Revalidate physical hand motion.

V3 profile, baseline thresholds/defaults, model weights, VERSION and GOLDEN manifest remain unchanged. `motion.py` now has an opt-in branch, so Golden verification on this research checkout will report it in addition to the inherited analyzer/decision differences. Do not regenerate the manifest or move release tags.

## Camera command

```bash
python examples/webcam_capture_viewport_v4.py --source 0 --device auto --viewport 0.16 0.18 0.68 0.64 --metrics outputs/v4_metrics.jsonl
```

The separate V4 entrypoint makes changed semantics explicit. The existing `webcam_research_vnext.py --profile vnext` still runs V3; `--profile fp2` still runs FP2. New harness greys all outside pixels, submits only ROI to the background worker, and stores only analyzed ROIs. D overlays exact ROI bounds, ROI-local detections shifted for display, side/motion/glare state and timing. R resets; Q exits. C is product best-recent-ROI capture; F is forced-current-ROI final for diagnosis. Both crop only after ACCEPT. Optional `--rotation 90` or 180/270 orients the sensor buffer first. The native OpenCV harness itself uses an unmirrored buffer preview.

No image persistence by default. `--record-images outputs/opt_in_images` explicitly saves selected/current ROI and accepted crops on capture. JSONL stores diagnostics/coordinates, not image bytes or OCR text. UI frame rate and analysis frequency are separate; `--analysis-fps 10` is a cap, not a speed guarantee.

## Validation and runtime

`python -m pytest -q`: **232 passed in 44.58 s** (all 189 prior tests unchanged + 43 V4 tests). Compile checks and `git diff --check` passed. The real packaged YOLO was exercised in deterministic image replays, not just mocked evidence tests. `tests/fixtures/v4/non_document_proxy.png` is a generated invalid-document image with no real identity data; `replay.json` describes the replay cases. Black, bright, seeded-noise and shifted-text outside regions give identical semantic outputs when ROI pixels are identical. Actual camera/display captures were not supplied.

The original Golden manifest is preserved. Verification reports exactly analyzer.py, decision.py and motion.py as changed on this research branch; that is expected, not a clean Golden certification. No old tests were rewritten for V4.

Fresh CPU benchmark: 5 warm-up + 40 measured frames per profile, Torch 2 threads/OpenCV 1, one packaged YOLO call per frame. Raw 1024x768; V4 resolved ROI `[164,139,860,629]`, 696x490. Latency includes V4 ROI extraction. Profiles run sequentially in one process. These are synthetic workload measurements, not accuracy or target-device guarantees.

| Workload/profile | p50 ms | p95 ms | p99 ms | Analysis Hz |
|---|---:|---:|---:|---:|
| FP2, real YOLO | 115.96 | 131.22 | 151.63 | 8.45 |
| V3, real YOLO | 128.94 | 136.56 | 138.32 | 7.71 |
| V4, real YOLO | 111.14 | 116.28 | 117.00 | 8.97 |
| FP2, supplied localization | 41.32 | 43.00 | 44.90 | 23.91 |
| V3, supplied localization | 52.92 | 66.19 | 67.53 | 18.28 |
| V4, supplied localization | 51.26 | 55.91 | 57.83 | 19.15 |

V4 median stage costs (ms): localization 56.16; page glare 13.40; MRZ local glare 10.45; side completeness 1.20; MRZ presence/completeness 0.49; motion 0.042; viewport extraction 0.150. Full distributions are in `research_v4_runtime_cpu.json` and `research_v4_runtime_supplied.json`. V4 forced missing-MRZ telemetry disabled/enabled costs p50 0.025/2.145 ms; the toggle has no decision authority (regression-tested).

Sampled process RSS: FP2 491.7 MiB, V3 500.7 MiB, V4 511.5 MiB. Sequential process RSS includes retained allocations and is not isolated per-profile memory. Selector pixel storage per frame is 2,359,296 raw bytes versus 1,023,120 ROI bytes, a 56.6% reduction; 12 candidate images would be 27.0 versus 11.7 MiB, excluding result/object overhead. Selector count/age/memory caps are unchanged. No GPU benchmark. The throughput improvement in this run is measured; do not generalize it to other cameras or infer improved recognition accuracy from it.

Reproduce from repository root:

```bash
python -m pytest -q
python -m compileall -q src examples tools tests
python tools/benchmark_research_vnext.py --viewport-v4 --output outputs/v4_runtime_cpu.json
python tools/benchmark_research_vnext.py --viewport-v4 --supplied --output outputs/v4_runtime_supplied.json
```

## Manual acceptance and remaining limits

P0: page outside viewport only; bottom two lines outside viewport; partial detected bottom text touching each viewport edge; small left/right text cuts; cut plus far document (direction before move closer); varying outside glare/text/noise with unchanged ROI. Use F to test current ROI directly; C can correctly select an earlier complete ROI.

P1: steady clean passport and clear monitor image; natural hand jitter; strong shake; blank top trim versus text trim; far complete page; very close complete content; destructive bottom reflection; bad-to-good recovery; C/F/final/crop. Confirm the exact debug rectangle agrees with visible usable pixels. Record case/time and JSONL for residual failures.

Known limitations: image replays are generated proxies, not a real passport population or physical-camera calibration. No live camera was available here. Detection can still miss or localize only an internal fragment; a partial MRZ whose detector box stays away from the boundary and passes broad priors can evade the completeness guard. Boundary contact can conservatively reject tightly fitted complete MRZ candidates. Compact high-contrast content-at-cut can miss low-contrast or large solid details. Large rotations, unusual layouts, security features and non-clipped glare need manual controls. Page-relative motion thresholds are an initial unit conversion and require target-camera jitter/shake validation. Host viewport mapping remains a P0 integration responsibility; use the actual preview transform. No production-release or authenticity/OCR-success claim.
