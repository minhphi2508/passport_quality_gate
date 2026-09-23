"""Opt-in validation harness for recent-best-frame selection.

Nothing is written unless the user presses C. On C, the harness saves:
- click_frame.jpg: full frame visible at button press (debug/reference)
- selected_frame.jpg: selected recent full frame (debug/reference)
- click_passport_crop.jpg: passport crop from click frame when final ACCEPT
- selected_passport_crop.jpg: passport crop from selected frame when final ACCEPT
- comparison.json: selection metadata and final decisions

The passport crops are the OCR-relevant outputs. Full frames are retained only
for manual comparison/debugging in this validation harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, strftime

import cv2
import numpy as np

from passport_quality_gate.api import PassportQualityGate, to_public_result
from passport_quality_gate.capture_output import extract_passport_page
from passport_quality_gate.frame_selector import BestFrameConfig, BestFrameSelector


def guide_poly(frame, box):
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    return np.array([
        [round(x*w), round(y*h)],
        [round((x+bw)*w), round(y*h)],
        [round((x+bw)*w), round((y+bh)*h)],
        [round(x*w), round((y+bh)*h)],
    ], np.int32)


def _accepted(result: dict) -> bool:
    # Works with the Astra integration fix while remaining safe on the frozen
    # raw-state convention used by older SDK snapshots.
    return bool(result.get('capture_allowed')) or result.get('state') == 'ACCEPT'


def _try_crop(frame, result):
    if not _accepted(result):
        return None, None
    try:
        return extract_passport_page(frame, result), None
    except (ValueError, cv2.error) as exc:
        return None, f'{type(exc).__name__}: {exc}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0')
    p.add_argument('--capture-backend', choices=['auto','dshow','msmf'], default='auto')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--analysis-fps', type=float, default=6.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--window-ms', type=float, default=750.0)
    p.add_argument('--output', type=Path, default=Path('outputs/best_frame_compare'))
    p.add_argument('--guide', default='0.16,0.18,0.68,0.64')
    a = p.parse_args()
    guide = tuple(float(x) for x in a.guide.split(','))
    if len(guide) != 4:
        p.error('--guide must be x,y,w,h')

    gate = PassportQualityGate(device=a.device)
    selector = BestFrameSelector(BestFrameConfig(window_ms=a.window_ms))
    source = int(a.source) if a.source.isdigit() else a.source
    backend = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}[a.capture_backend]
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        p.error('Cannot open camera source')
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

    latest = None
    last_analysis = -1e9
    index = 0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            now = monotonic()
            if now - last_analysis >= 1.0 / max(a.analysis_fps, 1e-6):
                last_analysis = now
                latest = gate.analyze_preview(raw, guide, timestamp=now)
                selector.push(raw, latest, timestamp=now)

            display = raw.copy()
            color = (40,220,40) if latest and latest.get('capture_allowed') else (0,190,255)
            poly = np.asarray(latest['guide_polygon'], np.int32) if latest else guide_poly(raw, guide)
            cv2.polylines(display, [poly], True, color, 2)
            text = f"C compare | buffer={selector.buffered_frames} | Q quit"
            cv2.putText(display, text, (12,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('Best-frame validation', display)
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if key == ord('c'):
                trigger = monotonic()
                click_frame = raw.copy()
                selected = selector.select_recent(trigger)

                gate.reset()
                click_final = gate.analyze_final(click_frame, guide, timestamp=trigger)

                if selected is None:
                    selected_frame = click_frame.copy()
                    selection = {"source":"click_frame_fallback", "selected_frame_age_ms":0.0}
                else:
                    selected_frame = selected.frame
                    selection = {"source":"recent_best_ready", **selected.metadata()}

                gate.reset()
                selected_final = gate.analyze_final(selected_frame, guide, timestamp=trigger)

                click_crop, click_crop_error = _try_crop(click_frame, click_final)
                selected_crop, selected_crop_error = _try_crop(selected_frame, selected_final)

                index += 1
                out = a.output / (strftime('%Y%m%d_%H%M%S') + f'_{index:03}')
                out.mkdir(parents=True, exist_ok=True)

                # Full frames are debug/reference only.
                cv2.imwrite(str(out/'click_frame.jpg'), click_frame)
                cv2.imwrite(str(out/'selected_frame.jpg'), selected_frame)

                click_crop_name = None
                selected_crop_name = None
                if click_crop is not None:
                    click_crop_name = 'click_passport_crop.jpg'
                    cv2.imwrite(str(out/click_crop_name), click_crop)
                if selected_crop is not None:
                    selected_crop_name = 'selected_passport_crop.jpg'
                    cv2.imwrite(str(out/selected_crop_name), selected_crop)

                payload = {
                    "selection": selection,
                    "click_final": to_public_result(click_final),
                    "selected_final": to_public_result(selected_final),
                    "click_passport_crop": click_crop_name,
                    "selected_passport_crop": selected_crop_name,
                    "click_crop_error": click_crop_error,
                    "selected_crop_error": selected_crop_error,
                    "ocr_handoff_preferred": selected_crop_name,
                }
                (out/'comparison.json').write_text(
                    json.dumps(payload, indent=2, allow_nan=False),
                    encoding='utf-8',
                )
                print(json.dumps(payload, indent=2, allow_nan=False), flush=True)

                gate.reset(); selector.clear(); latest = None; last_analysis = -1e9
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
