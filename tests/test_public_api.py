import numpy as np
import pytest

from passport_quality_gate.api import PassportQualityGate, QUALITY_POLICY
from passport_quality_gate.localization import Detection
from passport_quality_gate.types import GuideBox, coerce_guide_box


class NoPassportLocalizer:
    def locate(self, frame, mode="final"):
        return Detection(source="test")


def test_guide_box_accepts_supported_forms():
    expected = {"x": 0.16, "y": 0.18, "w": 0.68, "h": 0.64}
    assert coerce_guide_box(GuideBox(**expected)) == expected
    assert coerce_guide_box(expected) == expected
    assert coerce_guide_box((0.16, 0.18, 0.68, 0.64)) == expected


def test_guide_box_rejects_out_of_frame():
    with pytest.raises(ValueError):
        GuideBox(0.5, 0.2, 0.6, 0.6)


def test_public_api_delegates_to_frozen_engine_without_io():
    gate = PassportQualityGate(device="cpu", localizer=NoPassportLocalizer())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    preview = gate.analyze_preview(frame, (0.16, 0.18, 0.68, 0.64), timestamp=1.0)
    assert preview["passport_found"] is False
    assert preview["capture_allowed"] is False
    assert preview["guidance_code"] == "PLACE_PASSPORT_IN_FRAME"

    gate.reset()
    final = gate.analyze_final(frame, (0.16, 0.18, 0.68, 0.64), timestamp=2.0)
    assert final["passport_found"] is False
    assert final["state"] == "RETAKE"
    assert gate.metadata["quality_policy"] == QUALITY_POLICY


def test_document_crop_does_not_require_guide():
    gate = PassportQualityGate(device="cpu", localizer=NoPassportLocalizer())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = gate.analyze_document_crop(frame, timestamp=3.0)
    assert result["capture_context"] == "document_crop"

from passport_quality_gate.api import to_public_result
from passport_quality_gate.diagnostics import runtime_diagnostics


def test_public_contract_is_small_and_json_ready():
    raw = {
        "capture_allowed": True,
        "capture_quality_state": "READY",
        "workflow_state": "READY",
        "guidance_code": "MOVE_CLOSER",
        "recommended_adjustment": "MOVE_CLOSER",
        "blocking_issues": [],
        "advisories": ["MOVE_CLOSER"],
        "timing_ms": {"total": 12.3, "internal": 9.0},
        "raw_metrics": {"large": "internal"},
    }
    public = to_public_result(raw)
    assert public == {
        "capture_allowed": True,
        "capture_quality_state": "READY",
        "workflow_state": "READY",
        "guidance_code": "MOVE_CLOSER",
        "recommended_adjustment": "MOVE_CLOSER",
        "blocking_issues": [],
        "advisories": ["MOVE_CLOSER"],
        "timing_ms": {"total": 12.3},
    }


def test_public_convenience_method_uses_same_golden_decision():
    gate = PassportQualityGate(device="cpu", localizer=NoPassportLocalizer())
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    compact = gate.analyze_preview_public(frame, (0.1, 0.1, 0.8, 0.8), timestamp=1.0)
    assert compact["capture_allowed"] is False
    assert compact["guidance_code"] == "PLACE_PASSPORT_IN_FRAME"
    assert "raw_metrics" not in compact


def test_runtime_diagnostics_reports_packaged_assets():
    info = runtime_diagnostics()
    assert info["quality_policy"] == QUALITY_POLICY
    assert info["default_config_exists"] is True
    assert info["default_weights_exists"] is True
