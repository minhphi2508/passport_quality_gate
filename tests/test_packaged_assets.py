from hashlib import sha256
from pathlib import Path


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_packaged_runtime_assets_exist_and_profile_matches_source():
    root = Path(__file__).resolve().parents[1]
    assets = root / "src" / "passport_quality_gate" / "assets"
    assert (assets / "passport_detector_ver3_best.pt").is_file()
    assert (root / "src" / "passport_quality_gate" / "defaults.yaml").is_file()
    assert digest(root / "configs" / "capture_viewport.yaml") == digest(assets / "capture_viewport.yaml")
