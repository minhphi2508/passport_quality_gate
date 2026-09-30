"""Reference capture-viewport webcam integration.

D toggles diagnostics, C captures the best recent ROI, F final-checks the
current ROI, R resets the session, and Q exits. Images are not written unless
--record-images is supplied.
"""
from concurrent.futures import ThreadPoolExecutor
from passport_quality_gate.capture_policy import guidance_text
from passport_quality_gate.viewport import CaptureViewport, orient_camera
import argparse
import json
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np
from passport_quality_gate.runtime_metrics import memory_mib

from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.capture_output import extract_passport_page
from passport_quality_gate.frame_selector import BestFrameSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--viewport', nargs=4, type=float, default=[.16,.18,.68,.64], metavar=('X','Y','W','H'))
    parser.add_argument('--rotation',type=int,choices=[0,90,180,270],default=0)
    parser.add_argument('--analysis-fps', type=float, default=10.)
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--record-images', type=Path)
    args = parser.parse_args()
    if args.analysis_fps <= 0:
        parser.error('--analysis-fps must be positive')
    viewport = CaptureViewport(*args.viewport)
    start = perf_counter()
    gate = PassportQualityGate(device=args.device, config="capture_viewport")
    print(json.dumps({'startup_ms': (perf_counter()-start)*1000, **gate.runtime_info()}))
    selector = BestFrameSelector()
    camera = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise RuntimeError('Cannot open camera; check --source')
    viewport_key = None
    log = None
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        log = args.metrics.open('w', encoding='utf-8')
    if args.record_images:
        args.record_images.mkdir(parents=True, exist_ok=True)
    latest = None
    last_analysis = -1e9
    debug = False
    timings = []
    peak_rss_mib = 0.
    flips = guidance_flips = 0
    started = monotonic()
    capture_index = 0
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='quality')
    pending = None
    analyzed_frame = None
    analyzed_time = None

    def record(result, observed_frame, stamp):
        nonlocal latest, flips, guidance_flips, peak_rss_mib
        previous = latest
        latest = result
        selector.push(observed_frame, latest, timestamp=stamp)
        timings.append(latest['timing_ms']['total'])
        peak_rss_mib = max(peak_rss_mib, (memory_mib() or 0.))
        if previous:
            flips += previous['capture_allowed'] != latest['capture_allowed']
            guidance_flips += previous['guidance_code'] != latest['guidance_code']
        latest['ready_exit_diagnostics'] = {
            'exited': bool(previous and previous['capture_allowed'] and not latest['capture_allowed']),
            'blocking_issues': latest['blocking_issues'],
            'motion': latest['raw_metrics'].get('motion'),
            'blur_score': latest['quality'].get('blur_score'),
        }
        if log:
            log.write(json.dumps({'event': 'preview', **latest}, allow_nan=False)+'\n')
            log.flush()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame=orient_camera(frame,args.rotation)
            roi,viewport_meta=viewport.extract(frame,transform={'rotation_clockwise':args.rotation,'mirrored':False,'mode':'buffer'})
            key_now=(tuple(viewport_meta['viewport_pixel_rect']),tuple(frame.shape))
            if viewport_key is not None and viewport_key!=key_now:
                if pending is not None: pending.result(); pending=None
                gate.reset(); selector.clear(); latest=None
            viewport_key=key_now
            now = monotonic()
            if pending is not None and pending.done():
                record(pending.result(), analyzed_frame, analyzed_time)
                pending = None
            if pending is None and now-last_analysis >= 1/args.analysis_fps:
                last_analysis = now
                analyzed_frame = roi.copy()
                analyzed_time = now
                pending = worker.submit(gate.analyze_roi_preview, analyzed_frame, timestamp=now, viewport_metadata=viewport_meta)
            display = np.full_like(frame,100)
            x0,y0,x1,y1=viewport_meta['viewport_pixel_rect']
            display[y0:y1,x0:x1]=roi
            cv2.rectangle(display,(x0,y0),(x1-1,y1-1),(80,210,250),2)
            if latest:
                color = (40, 220, 40) if latest['capture_allowed'] else (0, 190, 255)
                lines = [f"{latest.get('guidance_text', guidance_text(latest['guidance_code']))}", 'D debug | C best ROI | F current ROI | R reset | Q quit']
                if debug:
                    offset=np.array([x0,y0])
                    for name,color_poly in [('polygon',(0,255,0)),('mrz_polygon',(255,100,0))]:
                        polygon=latest['localization'].get(name)
                        if polygon is not None:
                            cv2.polylines(display,[(np.asarray(polygon)+offset).astype(np.int32)],True,color_poly,2)
                    lines += [f"ROI {viewport_meta['viewport_pixel_rect']} size {roi.shape[1]}x{roi.shape[0]}",
                              f"motion {latest['raw_metrics'].get('motion')}"]
                    lines += [f"{latest['state']} | {latest['guidance_code']}"]
                    lines += [f"block: {latest['blocking_issues']}", f"timing: {latest['timing_ms']['total']:.1f} ms"]
                    diagnostics = latest.get('capture_diagnostics', {})
                    lines += [f"cuts: {diagnostics.get('cut_scores', {})}", f"MRZ: {diagnostics.get('mrz', {}).get('state')} glare: {latest['quality'].get('mrz_glare_score')}"]
                for i, line in enumerate(lines):
                    cv2.putText(display, line, (12, 28+25*i), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            cv2.imshow('Passport capture viewport', display)
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key == ord('d'):
                debug = not debug
            if key == ord('r'):
                if pending is not None:
                    pending.result(); pending = None
                gate.reset(); selector.clear(); latest = None; last_analysis = -1e9
            if key in (ord('c'),ord('f')):
                trigger = monotonic()
                if pending is not None:
                    record(pending.result(), analyzed_frame, analyzed_time)
                    pending = None
                selected = selector.select_recent(trigger_timestamp=trigger) if key==ord('c') else None
                chosen = selected.frame if selected is not None else roi
                chosen_meta=selected.result['capture_viewport'] if selected is not None else viewport_meta
                final = gate.analyze_roi_final(chosen,timestamp=trigger,viewport_metadata=chosen_meta)
                crop = None
                error = None
                if final['capture_allowed']:
                    try:
                        crop = extract_passport_page(chosen, final)
                    except (ValueError, cv2.error) as exc:
                        error = str(exc)
                payload = {'event': 'capture', 'path':'best_roi' if key==ord('c') else 'forced_current_roi', 'selection': selected.metadata() if selected else None,
                           'final': final, 'crop_ok': crop is not None, 'crop_error': error}
                print(json.dumps(payload, allow_nan=False))
                if log:
                    log.write(json.dumps(payload, allow_nan=False)+'\n'); log.flush()
                if args.record_images:
                    capture_index += 1
                    cv2.imwrite(str(args.record_images / f'{capture_index:04d}_selected.jpg'), chosen)
                    if crop is not None:
                        cv2.imwrite(str(args.record_images / f'{capture_index:04d}_crop.jpg'), crop)
                gate.reset(); selector.clear(); latest = None; last_analysis = -1e9
    finally:
        worker.shutdown(wait=True)
        elapsed = monotonic()-started
        summary = {'event': 'summary', 'frames': len(timings), 'analysis_hz': len(timings)/max(elapsed, 1e-9),
                   'process_peak_sampled_rss_mib': peak_rss_mib, 'decision_flips_per_min': 60*flips/max(elapsed,1e-9),
                   'decision_flips': int(flips), 'guidance_flips': int(guidance_flips),
                   'latency_ms_p50_p95_p99': np.percentile(timings, [50,95,99]).tolist() if timings else []}
        print(json.dumps(summary))
        if log:
            log.write(json.dumps(summary)+'\n'); log.close()
        camera.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
