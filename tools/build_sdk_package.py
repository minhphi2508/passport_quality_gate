"""Build the v0.1.4 source SDK and wheel after release-core verification."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
SOURCE_ZIP = DIST / f"passport_quality_gate_sdk_v{VERSION}.zip"

INCLUDE_DIRS = ["src/passport_quality_gate", "configs", "models", "examples", "docs", "tests"]
INCLUDE_FILES = [
    "README.md", "VERSION", "CHANGELOG.md", "pyproject.toml", "requirements.txt",
    "RELEASE_CORE_SHA256.json",
    "tools/verify_release_core.py",
    "tools/build_sdk_package.py", "tools/acceptance_check.py",
]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", "passport_quality_gate.egg-info", "outputs", ".venv", "dist"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release_core() -> None:
    data = json.loads((ROOT / "RELEASE_CORE_SHA256.json").read_text(encoding="utf-8"))
    bad = []
    for rel, expected in data["files"].items():
        path = ROOT / rel
        if not path.is_file():
            bad.append(f"MISSING {rel}")
        elif sha(path) != expected:
            bad.append(f"CHANGED {rel}")
    if bad:
        raise SystemExit("Release core verification failed:\n" + "\n".join(bad))


def verify_version() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    api_text = (ROOT / "src/passport_quality_gate/api.py").read_text(encoding="utf-8")
    if pyproject["project"]["version"] != VERSION or f'SDK_CANDIDATE_VERSION = "{VERSION}"' not in api_text:
        raise SystemExit("VERSION, pyproject.toml and api.py disagree")


def verify_assets() -> None:
    pairs = [
        ("configs/capture_viewport.yaml", "src/passport_quality_gate/assets/capture_viewport.yaml"),
    ]
    bad = [f"{a} != {b}" for a, b in pairs if sha(ROOT / a) != sha(ROOT / b)]
    if bad:
        raise SystemExit("Packaged asset parity failed:\n" + "\n".join(bad))


def allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return path.is_file() and not any(part in EXCLUDE_PARTS for part in rel.parts) and path.suffix not in EXCLUDE_SUFFIXES


def build_source_zip() -> Path:
    files = []
    for rel in INCLUDE_DIRS:
        base = ROOT / rel
        if base.exists():
            files.extend(path for path in base.rglob("*") if allowed(path))
    for rel in INCLUDE_FILES:
        path = ROOT / rel
        if path.is_file():
            files.append(path)
    unique = {path.relative_to(ROOT).as_posix(): path for path in files}
    with zipfile.ZipFile(SOURCE_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname in sorted(unique):
            archive.write(unique[arcname], arcname)
    return SOURCE_ZIP


def build_wheel() -> Path:
    wheelhouse = DIST / "wheelhouse"
    shutil.rmtree(wheelhouse, ignore_errors=True)
    wheelhouse.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheelhouse)],
        cwd=ROOT,
        check=True,
    )
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one wheel, got {len(wheels)}")
    target = DIST / wheels[0].name
    shutil.copy2(wheels[0], target)
    shutil.rmtree(wheelhouse)
    return target


def main() -> None:
    verify_version()
    verify_release_core()
    verify_assets()
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir()
    source = build_source_zip()
    wheel = build_wheel()
    sums = DIST / "SHA256SUMS.txt"
    sums.write_text(
        f"{sha(source)}  {source.name}\n{sha(wheel)}  {wheel.name}\n",
        encoding="utf-8",
    )
    print(f"Built {source}")
    print(f"Built {wheel}")
    print(f"Checksums {sums}")


if __name__ == "__main__":
    main()
