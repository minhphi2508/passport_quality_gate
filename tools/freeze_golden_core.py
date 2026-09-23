from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "GOLDEN_CORE_SHA256.json"

CORE_PATHS = [
    "src/passport_quality_gate/__init__.py",
    "src/passport_quality_gate/analyzer.py",
    "src/passport_quality_gate/config.py",
    "src/passport_quality_gate/corners.py",
    "src/passport_quality_gate/decision.py",
    "src/passport_quality_gate/defaults.yaml",
    "src/passport_quality_gate/geometry.py",
    "src/passport_quality_gate/localization.py",
    "src/passport_quality_gate/motion.py",
    "src/passport_quality_gate/quality.py",
    "src/passport_quality_gate/readability.py",
    "configs/thresholds_v4.yaml",
    "models/passport_detector_ver3_best.pt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the validated FP2 Golden runtime core.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing manifest (normally do not use).")
    args = ap.parse_args()

    if MANIFEST.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing {MANIFEST.name}; use verify_golden_core.py instead.")

    missing = [p for p in CORE_PATHS if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit("Missing Golden core files:\n  " + "\n  ".join(missing))

    payload = {
        "schema_version": 1,
        "quality_policy": "FP2-GOLDEN-ACTUAL",
        "note": "Immutable validated quality-policy/runtime core. Refactors outside this list may proceed without redefining the quality policy.",
        "files": {p: sha256(ROOT / p) for p in CORE_PATHS},
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen Golden core: {len(CORE_PATHS)} files -> {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
