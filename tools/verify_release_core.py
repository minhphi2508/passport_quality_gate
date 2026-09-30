"""Verify the accepted v0.1.4 capture-viewport runtime fingerprint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_CORE_SHA256.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    for rel, expected in data["files"].items():
        path = ROOT / rel
        if not path.is_file():
            bad.append(f"MISSING {rel}")
        elif digest(path) != expected:
            bad.append(f"CHANGED {rel}")
    if bad:
        raise SystemExit("Release core verification failed:\n" + "\n".join(bad))
    print(f"Release core OK: {len(data['files'])} files match {data['policy']} / v{data['sdk_version']}")


if __name__ == "__main__":
    main()
