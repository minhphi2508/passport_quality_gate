"""Reference integration demo using only public SDK surfaces.

This is NOT the production camera layer. Resolution, backend and analysis rate
are demo options only. No frames/logs are saved by default.

When --save-dir is enabled and a final frame is accepted, the OCR handoff image
is a perspective-corrected passport-page crop produced from the exact selected
frame. The full selected frame is saved separately only as a debug artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, strftime

import cv2
import numpy as np

from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.capture_output import extract_passport_page
from passport_quality_gate.frame_selector import BestFrameConfig, BestFrameSelector

MESSAGES = {
    'PLACE_PASSPORT_IN_FRAME': 'Dua trang passport vao khung',
    'SHOW_ALL_EDGES': 'Can thay ro day du 4 canh passport',
    'MOVE_CLOSER': 'Dua passport lai gan hon',
    'MOVE_FARTHER': 'Dua passport ra xa hon',
    'MOVE_LEFT': 'Dich passport sang trai',
    'MOVE_RIGHT': 'Dich passport sang phai',
    'MOVE_UP': 'Dich passport len tren',
    'MOVE_DOWN': 'Dich passport xuong duoi',
    'ROTATE_PASSPORT': 'Xoay passport dung chieu',
    'HOLD_CAMERA_PARALLEL': 'Giu camera song song voi passport',
    'INCREASE_LIGHT': 'Tang anh sang',
    'REDUCE_DIRECT_LIGHT': 'Giam anh sang truc tiep',
    'HOLD_STEADY_OR_FOCUS': 'Giu yen / cho camera lay net',
    'AVOID_REFLECTION': 'Tranh anh sang phan chieu',
    'IMPROVE_LIGHTING': 'Dieu chinh anh sang',
    'SHOW_MRZ': 'Can thay ro ca hai dong MRZ',
    'HOLD_STEADY': 'Giu yen passport',
    'CAMERA_RESOLUTION_LOW': 'Tang do phan giai camera / doi camera',
    'READY': 'San sang chup',
    'CHECKING_STABILITY': 'Giu trong khung mot chut...',
}


def parse_guide(text: str):
    vals = tuple(float(x) for x in text.split(','))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError('guide must be x,y,w,h')
    return vals


def guide_poly(frame, box):
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    return np.array([
        [round(x*w), round(y*h)],
        [round((x+bw)*w), round(y*h)],
        [round((x+bw)*w), round((y+bh)*h)],
        [round(x*w), round((y+bh)*h)],
    ], np.int32)


def compact_result(result):
    return {
        'state': result.get('state'),
        'capture_allowed': result.get('capture_allowed'),
        'capture_quality_state': result.get('capture_quality_state'),
        'workflow_state': result.get('workflow_state'),
        'guidance_code': result.get('guidance_code'),
        'blocking_issues': result.get('blocking_issues'),
        'advisories': result.get('advisories'),
        'timing_ms': result.get('timing_ms'),
    }


def main():
    p = argparse.ArgumentParser(description='Passport Quality Gate SDK webcam reference demo')
    p.add_argument('--source', default='0')
    p.add_argument('--capture-backend', choices=['auto','dshow','msmf'], default='auto')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--analysis-fps', type=float, default=4.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--guide', type=parse_guide, default=(0.16,0.18,0.68,0.64))
    p.add_argument('--best-window-ms', type=float, default=750.0)
    p.add_argument(
        '--save-dir',
        type=Path,
        help='Opt-in: on C, save accepted passport crop + debug full frame + JSON',
    )
    p.add_argument('--log-jsonl', type=Path, help='Opt-in compact analysis log')
    a = p.parse_args()

    if a.analysis_fps <= 0:
        p.error('--analysis-fps must be positive')
    gate = PassportQualityGate(device=a.device)
    selector = BestFrameSelector(BestFrameConfig(window_ms=a.best_window_ms))
    source = int(a.source) if a.source.isdigit() else a.source
    backend = {'auto':cv2.CAP_ANY,'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF}[a.capture_backend]
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        p.error('Cannot open source')
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

    log = None
    if a.log_jsonl:
        a.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log = a.log_jsonl.open('w', encoding='utf-8')
    last_analysis = -1e9
    latest = None
    capture_index = 0
    final_text = 'C: select recent best frame | R: reset | Q: quit'
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            now = monotonic()
            if now - last_analysis >= 1.0/a.analysis_fps - 1e-9:
                last_analysis = now
                latest = gate.analyze_preview(raw, a.guide, timestamp=now)
                selector.push(raw, latest, timestamp=now)
                if log:
                    log.write(json.dumps({'timestamp_s':now, **compact_result(latest)}, allow_nan=False)+'\n')
                    log.flush()

            display = raw.copy()
            ready = bool(latest and latest.get('capture_allowed'))
            color = (40,220,40) if ready else (0,190,255)
            poly = np.asarray(latest['guide_polygon'], np.int32) if latest else guide_poly(raw, a.guide)
            cv2.polylines(display, [poly], True, color, 2)
            if latest:
                loc = latest.get('localization') or {}
                if loc.get('polygon') is not None:
                    cv2.polylines(display, [np.asarray(loc['polygon'],np.int32)], True, (255,180,50), 1)
                code = latest.get('guidance_code') or 'HOLD_STEADY'
                headline = MESSAGES.get(code, code)
                line2 = f"{latest.get('workflow_state')} | {latest.get('capture_quality_state')} | {latest.get('timing_ms',{}).get('total',0):.0f} ms"
                line3 = f"best-buffer {selector.buffered_frames} frames / {selector.buffered_megabytes:.1f} MiB"
            else:
                headline, line2, line3 = 'Dang khoi dong', '', ''
            scale = min(1.0, 960/display.shape[1])
            display = cv2.resize(display, None, fx=scale, fy=scale)
            panel = np.full((130, max(720, display.shape[1]), 3), 20, np.uint8)
            for i, text in enumerate((headline,line2,line3,final_text)):
                cv2.putText(panel, text, (10,25+i*27), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            color if i==0 else (235,235,235), 1, cv2.LINE_AA)
            canvas = np.zeros((display.shape[0], panel.shape[1], 3), np.uint8)
            canvas[:,:display.shape[1]] = display
            cv2.imshow('Passport Quality Gate SDK Reference Demo', np.vstack([panel,canvas]))
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key == ord('r'):
                gate.reset(); selector.clear(); latest = None; last_analysis = -1e9
            if key == ord('c'):
                trigger = monotonic()
                selected = selector.select_recent(trigger)
                if selected is None:
                    chosen = raw.copy()
                    selection_meta = {'source':'current_frame_fallback','selected_frame_age_ms':0.0}
                else:
                    chosen = selected.frame
                    selection_meta = {'source':'recent_best_ready', **selected.metadata()}

                # Final check is authoritative and runs on the exact selected pixels.
                final = gate.analyze_final(chosen, a.guide, timestamp=trigger)
                final_allowed = bool(final.get('capture_allowed')) or final.get('state') == 'ACCEPT'

                passport_crop = None
                crop_error = None
                if final_allowed:
                    try:
                        passport_crop = extract_passport_page(chosen, final)
                    except (ValueError, cv2.error) as exc:
                        crop_error = f'{type(exc).__name__}: {exc}'

                crop_status = 'crop-ready' if passport_crop is not None else ('crop-failed' if final_allowed else 'retake')
                final_text = (
                    f"FINAL {final.get('state')} | "
                    f"{final.get('guidance_code') or final.get('primary_issue')} | "
                    f"{selection_meta['source']} | {crop_status}"
                )
                print(final_text, flush=True)

                if a.save_dir:
                    capture_index += 1
                    a.save_dir.mkdir(parents=True, exist_ok=True)
                    stem = strftime('%Y%m%d_%H%M%S') + f'_{capture_index:03}'

                    # OCR handoff image: accepted passport-page crop only.
                    passport_name = None
                    if passport_crop is not None:
                        passport_name = stem + '_passport.jpg'
                        cv2.imwrite(str(a.save_dir/passport_name), passport_crop)

                    # Development/debug only; never the preferred OCR handoff image.
                    debug_name = stem + '_full_debug.jpg'
                    cv2.imwrite(str(a.save_dir/debug_name), chosen)

                    payload = {
                        'selection': selection_meta,
                        'final': final,
                        'ocr_handoff_image': passport_name,
                        'debug_full_frame': debug_name,
                        'crop_error': crop_error,
                    }
                    (a.save_dir/(stem+'.json')).write_text(
                        json.dumps(payload, indent=2, allow_nan=False),
                        encoding='utf-8',
                    )

                gate.reset(); selector.clear(); latest = None; last_analysis = -1e9
    finally:
        cap.release()
        if log:
            log.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
