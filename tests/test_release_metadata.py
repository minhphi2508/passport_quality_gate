from pathlib import Path
import tomllib

from passport_quality_gate.api import PassportQualityGate, SDK_CANDIDATE_VERSION
from passport_quality_gate.localization import Detection


class NoPassportLocalizer:
    def locate(self, frame, mode="final"):
        return Detection(source="test")


def test_release_version_and_detector_dependency_are_declared():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == "0.1.2"
    assert SDK_CANDIDATE_VERSION == "0.1.2"
    deps = [str(x).lower() for x in data["project"]["dependencies"]]
    assert any(x.startswith("ultralytics>=8.3") for x in deps)


def test_runtime_info_matches_metadata():
    gate = PassportQualityGate(device="cpu", localizer=NoPassportLocalizer())
    assert gate.runtime_info() == gate.metadata
