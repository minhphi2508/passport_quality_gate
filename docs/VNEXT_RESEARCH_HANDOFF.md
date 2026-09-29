# VNext research: manual camera handoff

> Historical V2 handoff for commit 93923f6. For the current branch use [V3 corrective handoff](V3_CORRECTIVE_HANDOFF.md); the fallback-authority and glare policies described below are superseded.

This branch implements the supplied `PASSPORT_QUALITY_GATE_ASTRA_EXECUTION_HANDOFF_v2.pdf` against main `b0c3f4d6d4d09d684626fb9b9a8094add7b8c633`. Main matched the expected handoff exactly. Before quality edits: Golden verification passed all 13 files; baseline tests: **126 passed in 36.35 s**.

Research only; manual acceptance is pending. No new release, no production-validation claim.

Final validation: **151 passed in 46.38 s** (126 existing + 25 research cases). Compile checks and `git diff --check` passed. Post-edit Golden verification reports exactly the two expected research-touched core files: analyzer.py and decision.py. Manifest/release/weights hashes were not changed.

## Run

From an existing editable installation in this repository, after checking out `research/vnext-quality-gate`:

```bash
python examples/webcam_research_vnext.py --source 0 --device auto --profile vnext --metrics outputs/vnext_metrics.jsonl
```

Change `--source 0` to `--source 1` if the phone/webcam is camera 1. If needed, install the existing project dependencies with `python -m pip install -e ".[dev]"`. This branch adds no dependencies. A desktop OpenCV build is required for any webcam window (as for the existing examples); the headless build alone cannot display windows.

- D: detailed overlay. Q: quit and print runtime/flip summary.
- R: reset session, MRZ history and best-frame buffer.
- C: select a recent eligible full frame, check **those exact pixels** in final mode, then extract the page only after ACCEPT. Prints final/crop result; does not save images.
- `--record-images outputs/opt_in_images`: explicit opt-in to save selected full frames and successful crops when C is pressed. No images are saved otherwise.
- `--analysis-fps 10`: upper limit, not a throughput promise. One background analysis worker keeps camera display independent of model cadence; no queue of stale camera frames accumulates. Final/reset waits for the one in-flight analysis before using the shared model.

JSONL contains numeric/localization diagnostics and decisions, not recognized text or image bytes. It includes actual READY-exit blockers, motion, blur, raw/confirmed glare, local cells, semantic cut sides, MRZ state/fallback, stage timings and final selection metadata. UI preview cannot certify an unanalysed newer camera frame; C always performs the authoritative final check.

## Changes and rationale

| Area | Implementation |
|---|---|
| Boundary/completeness | Semantic camera-side color/normal-gradient samples plus existing edge evidence; frame proximity separated from cut. Two-cue fusion of frame contact, asymmetric boundary support, page shape and MRZ layout. Unobservable outside pixels never count as weak-edge proof. |
| Top/bottom policy | Upper-strip text-like content and substantial-trim prior gate top rejection. Bottom loss remains strict. One missing side gives a directional action; multiple sides may give SHOW_ALL_EDGES. |
| MRZ state | Confident/plausibly associated YOLO => STRONG; credible weak detector or two-row classical fallback => WEAK; no credible MRZ plus bottom-cut evidence => ABSENT. Unresolved detection uncertainty blocks as QUALITY_UNCERTAIN rather than claiming physical loss. |
| Fallback | Only when MRZ is weak/missing: lower-page black-hat, horizontal gradients, closing, two aligned wide text rows. No OCR and no second detector pass. Caller detections are copied, not changed. |
| Glare | Opt-in fix removes the global raw-score cap from candidate-derived MRZ components. Original-frame MRZ alone is warped to width 800; 2x22 cells measure candidate coverage, clipping, contrast and edge retention. MRZ_GLARE is exposed, with GLARE compatibility retained. Global/body confirmation stays in place. |
| Temporal glare | At most four 220x40 grayscale MRZ crops in RAM; phase-aligned highlight changes corroborate spatial damage. Gap, geometry jump, missing document/MRZ and reset clear history. Final does not use preview history. |
| Guidance | Independent arbitration favors confirmed cuts/MRZ glare and useful lighting actions, distinguishes focus from motion blur, uses neutral localization guidance, and retains an active message briefly unless a stronger cause appears. Recovered causes are removed immediately. |
| Jitter | **No threshold relaxation or extra READY grace enabled.** Mild synthetic jitter stays READY and severe motion still blocks immediately. The real handheld flicker cause is not known; new logs identify whether it is BLUR, HOLD_STEADY or another defect before a later justified adjustment. |
| Integration | Default profile is unchanged; explicit `configs/research_vnext.yaml` activates research. Metadata reports VNEXT-RESEARCH. Selected-frame/final/crop and public keys remain compatible. |

Files: `analyzer.py` and `decision.py` contain small opt-in hooks; `research.py` owns new evidence/guidance; `api.py` identifies the active profile; `runtime_metrics.py`, the dedicated webcam harness and benchmark provide measurements. Config and targeted tests are separate. Existing tests were not deleted or weakened.

## Runtime evidence

Linux x86_64, 9 logical CPUs exposed, CPU only, Python 3.12.14, Torch 2.14.0+cu130 (CUDA unavailable), OpenCV 5.0.0. Benchmark explicitly uses Torch 2 threads and OpenCV 1 thread. Forty measured frames per profile after five warm-up frames; 1024x768 non-document synthetic proxy. The existing packaged YOLO was run once on each measured frame and found the proxy page in all 40 frames. These are throughput measurements, not evidence of real-passport recognition accuracy.

| Workload/profile | p50 ms | p95 ms | p99 ms | Analysis Hz | Process memory MiB |
|---|---:|---:|---:|---:|---:|
| Real YOLO + FP2 | 194.1 | 245.6 | 353.7 | 4.91 | 930.3 |
| Real YOLO + VNext | 217.9 | 264.3 | 268.7 | 4.63 | 949.3 |
| Supplied localization + FP2 | 63.9 | 84.7 | 85.2 | 15.02 | 676.2 |
| Supplied localization + VNext | 81.9 | 103.4 | 108.3 | 11.74 | 680.7 |

VNext real-model median stages: localization 132.3 ms; readability 38.0 ms; page glare 13.1 ms; added MRZ glare 14.6 ms; side completeness 1.8 ms; MRZ presence 0.4 ms. Forced detector-miss fallback: p50/p95/p99 3.7/6.2/7.2 ms. Raw stage distributions are in `research_runtime_cpu.json` and `research_runtime_supplied.json`.

Gate construction in the already-imported benchmark process: FP2 268 ms, VNext 84 ms (second construction benefits from warm caches). A separate fresh-process VNext smoke measured ~1.93 s including lazy imports/model construction, and ~3.27 s for first inference/analysis. These startup figures are not comparable cold-start trials. Memory is sampled process RSS where available, Unix process high-water RSS fallback in this sandbox; both profiles share a sequential benchmark process, so values are not isolated per-model footprints. No VRAM result because CUDA was not exercised.

**This CPU did not meet the suggested 8–15 Hz with real YOLO.** Added classical work does not dominate latency; target-device profiling is still required. Worker scheduling improves display responsiveness, not detector throughput. Do not interpret the small-sample p99 comparison as proof of a speedup.

Reproduce:

```bash
python tools/benchmark_research_vnext.py --output outputs/runtime_cpu.json
python tools/benchmark_research_vnext.py --supplied --output outputs/runtime_supplied.json
python -m pytest -q
```

## Manual checklist

Run each scenario for about 5–10 seconds. Note the scenario and approximate timestamp in the JSONL when behavior is wrong; report whether READY or the guidance is wrong. D shows detailed diagnostics. Avoid treating the synthetic tests as a substitute for this checklist.

| # | Scenario | Expected |
|---|---|---|
| 1 | Good, steady | Stable READY/OPTIMAL, quiet guidance |
| 2 | Natural hand jitter | Mostly READY; record any repeated exit and its blocker |
| 3 | Strong shake | Prompt HOLD_STEADY/block |
| 4 | Far but readable | READY; framing advisory is acceptable |
| 5 | Extremely far | Block, resolution/move closer guidance |
| 6 | Very close, useful content complete | May READY; no repeated SHOW_ALL_EDGES solely for proximity |
| 7 | Cut left | Block, MOVE_RIGHT |
| 8 | Cut right | Block, MOVE_LEFT |
| 9a | Small top blank-margin trim | May READY |
| 9b | Top cut into important content | Block, MOVE_DOWN |
| 10 | Bottom/MRZ margin cut | Block, MOVE_UP |
| 11a | MRZ physically missing/cropped | Block |
| 11b | MRZ visible, weak YOLO | Classical fallback may avoid needless MRZ_NOT_FOUND |
| 12 | All useful content, weak edges/corners | Can READY; uncertainty alone is not incompleteness |
| 13 | Body glare | Block when materially harmful |
| 14 | Small reflection over MRZ characters | Block, REDUCE_REFLECTION_ON_MRZ |
| 15 | Clean bright page/hologram shine | Avoid false glare block |
| 16 | Bad to good | Recover without stale guidance/latch |
| 17 | Shutter disturbance | C can choose recent good frame; final checks selected pixels |
| 18 | Final ACCEPT/crop | Crop succeeds only after accepted exact-frame final |

## A/B and rollback

Use the same command with `--profile fp2 --metrics outputs/fp2_metrics.jsonl` to run the unchanged FP2 behavior on this branch. Use the same camera, lighting and guide for comparison. Profile thresholds otherwise inherit the baseline; the research CROPPED threshold is aligned with the side-MISSING threshold (0.55) so uncertainty below that level cannot trigger ambiguous crop guidance.

For the exact original source baseline in a separate checkout:

```bash
git worktree add ../passport_quality_gate_fp2 b0c3f4d6d4d09d684626fb9b9a8094add7b8c633
```

Or switch the existing clean checkout back to `main`. No history rewrite, no tag movement. `GOLDEN_CORE_SHA256.json`, `VERSION`, release metadata, packaged FP2 config and detector weights remain unchanged. After research edits, `verify_golden_core.py` intentionally reports analyzer.py and decision.py mismatches; do not regenerate hashes. Baseline verification belongs to the original checkout.

## Remaining limits

Real camera data, passport variants, MRZ fonts, security features and device latency remain unvalidated. Broad layout priors may confuse unusual layouts, large rotation/perspective, or a detector box around an internal fragment; two ordinary lower-page text lines may mimic the classical fallback. Top content uses local ink/geometry, not semantic OCR, and cannot certify that every required field exists. Glare is candidate-gated and can miss non-clipped washout or fail under bad localization; actual hologram false positives need manual controls. Temporal comparison is deliberately only corroborating evidence.

No extra motion grace is claimed: real jitter logs are needed. No heavy probe, learned model, full OCR, second YOLO pass, automatic image retention or broad production calibration was introduced.
