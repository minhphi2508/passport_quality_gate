"""Minimal public-API example. No debug files are created unless --json-out is set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from passport_quality_gate.api import PassportQualityGate


def parse_guide(text: str):
    vals = tuple(float(x) for x in text.split(','))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError('guide must be x,y,w,h')
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image', type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--mode', choices=['final', 'document-crop'], default='final')
    p.add_argument('--guide', type=parse_guide, default=(0.16, 0.18, 0.68, 0.64))
    p.add_argument('--json-out', type=Path)
    a = p.parse_args()
    frame = cv2.imread(str(a.image))
    if frame is None:
        p.error(f'Cannot decode image: {a.image}')
    gate = PassportQualityGate(device=a.device)
    result = gate.analyze_document_crop(frame) if a.mode == 'document-crop' else gate.analyze_final(frame, a.guide)
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
