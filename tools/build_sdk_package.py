"""Build source ZIP + installable wheel after verifying frozen Golden core/assets."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
VERSION = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
SOURCE_ZIP = DIST / f'passport_quality_gate_sdk_v{VERSION}.zip'

INCLUDE_DIRS = ['src/passport_quality_gate', 'configs', 'models', 'examples', 'docs', 'tests']
INCLUDE_FILES = [
    'README.md', 'VERSION', 'CHANGELOG.md', 'pyproject.toml', 'GOLDEN_CORE_SHA256.json',
    'requirements.txt', 'requirements-detector.txt', 'requirements-tested.txt', 'requirements-advanced.txt',
    'tools/verify_golden_core.py', 'tools/build_sdk_package.py', 'tools/acceptance_check.py',
]
EXCLUDE_PARTS = {'__pycache__', '.pytest_cache', 'passport_quality_gate.egg-info', 'archive', 'outputs', '.venv', 'dist'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_core() -> None:
    data = json.loads((ROOT / 'GOLDEN_CORE_SHA256.json').read_text(encoding='utf-8'))
    bad = []
    for rel, expected in data['files'].items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(f'MISSING {rel}')
            continue
        if sha(p) != expected:
            bad.append(f'CHANGED {rel}')
    if bad:
        raise SystemExit('Golden core verification failed:\n' + '\n'.join(bad))


def verify_assets() -> None:
    pairs = [
        ('configs/thresholds_v4.yaml', 'src/passport_quality_gate/assets/thresholds_v4.yaml'),
        ('models/passport_detector_ver3_best.pt', 'src/passport_quality_gate/assets/passport_detector_ver3_best.pt'),
    ]
    bad = [f'{a} != {b}' for a, b in pairs if sha(ROOT/a) != sha(ROOT/b)]
    if bad:
        raise SystemExit('Packaged asset parity failed:\n' + '\n'.join(bad))


def allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return path.is_file() and not any(p in EXCLUDE_PARTS for p in rel.parts) and path.suffix not in EXCLUDE_SUFFIXES


def build_source_zip() -> tuple[Path, str]:
    files = []
    for rel in INCLUDE_DIRS:
        base = ROOT / rel
        if base.exists():
            files.extend(p for p in base.rglob('*') if allowed(p))
    for rel in INCLUDE_FILES:
        p = ROOT / rel
        if p.is_file():
            files.append(p)
    unique = {p.relative_to(ROOT).as_posix(): p for p in files}
    with zipfile.ZipFile(SOURCE_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for arc in sorted(unique):
            z.write(unique[arc], arc)
    return SOURCE_ZIP, sha(SOURCE_ZIP)


def build_wheel() -> Path:
    wheelhouse = DIST / 'wheelhouse'
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    wheelhouse.mkdir(parents=True)
    cmd = [sys.executable, '-m', 'pip', 'wheel', '.', '--no-deps', '--no-build-isolation', '-w', str(wheelhouse)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    wheels = sorted(wheelhouse.glob('*.whl'))
    if len(wheels) != 1:
        raise SystemExit(f'Expected one wheel, got {len(wheels)}')
    target = DIST / wheels[0].name
    shutil.copy2(wheels[0], target)
    shutil.rmtree(wheelhouse)
    return target


def main() -> None:
    verify_core()
    verify_assets()
    DIST.mkdir(exist_ok=True)
    source, source_sha = build_source_zip()
    wheel = build_wheel()
    wheel_sha = sha(wheel)
    sums = DIST / 'SHA256SUMS.txt'
    sums.write_text(
        f'{source_sha}  {source.name}\n{wheel_sha}  {wheel.name}\n',
        encoding='utf-8',
    )
    print(f'Built {source}')
    print(f'Built {wheel}')
    print(f'Checksums {sums}')


if __name__ == '__main__':
    main()
