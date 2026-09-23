from hashlib import sha256
from pathlib import Path


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_packaged_config_and_weights_are_exact_golden_copies():
    root = Path(__file__).resolve().parents[1]
    assert digest(root / "configs" / "thresholds_v4.yaml") == digest(
        root / "src" / "passport_quality_gate" / "assets" / "thresholds_v4.yaml"
    )
    assert digest(root / "models" / "passport_detector_ver3_best.pt") == digest(
        root / "src" / "passport_quality_gate" / "assets" / "passport_detector_ver3_best.pt"
    )
