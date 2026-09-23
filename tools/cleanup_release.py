"""Archive legacy research/demo files without deleting source history.

Dry-run by default. Use --apply after reviewing the plan.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVES = {
    'archive/research_scripts': [
        'scripts/benchmark_latency.py','scripts/benchmark_quality.py','scripts/benchmark_quality_v3.py',
        'scripts/benchmark_v2.py','scripts/benchmark_v3.py','scripts/compare_v1_v2.py',
        'scripts/experiment_docshrnet.py','scripts/generate_degradations.py','scripts/simulate_preview.py',
        'scripts/summarize_v4_2_log.py',
    ],
    'archive/legacy_runners': [
        'scripts/run_demo.py','scripts/run_folder.py','scripts/run_image.py',
        'scripts/run_image_v4.py','scripts/run_webcam.py','scripts/run_webcam_gui.py','scripts/run_webcam_v2.py',
        'scripts/run_webcam_v4.py',
    ],
    'archive/legacy_configs': ['configs/thresholds_v3.yaml'],
    'archive/docs_history': ['README_V4_3_FP2.md','API_CONTRACT_DRAFT.md','STEP2_README.md'],
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true')
    a=p.parse_args()
    moved=0
    for dest_rel, items in MOVES.items():
        dest=ROOT/dest_rel
        for rel in items:
            src=ROOT/rel
            if not src.exists():
                continue
            print(('MOVE ' if a.apply else 'WOULD MOVE ')+f'{rel} -> {dest_rel}/')
            if a.apply:
                dest.mkdir(parents=True, exist_ok=True)
                target=dest/src.name
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite {target}')
                shutil.move(str(src), str(target))
            moved+=1
    caches=[]
    for pth in ROOT.rglob('__pycache__'):
        if pth.is_dir(): caches.append(pth)
    for pth in [ROOT/'.pytest_cache', ROOT/'src/passport_quality_gate.egg-info']:
        if pth.exists(): caches.append(pth)
    for pth in caches:
        print(('REMOVE ' if a.apply else 'WOULD REMOVE ')+str(pth.relative_to(ROOT)))
        if a.apply:
            shutil.rmtree(pth, ignore_errors=True)
    for pyc in ROOT.rglob('*.pyc'):
        print(('REMOVE ' if a.apply else 'WOULD REMOVE ')+str(pyc.relative_to(ROOT)))
        if a.apply:
            pyc.unlink(missing_ok=True)
    print(f"{'Applied' if a.apply else 'Dry-run'}: {moved} move candidates")


if __name__=='__main__':
    main()
