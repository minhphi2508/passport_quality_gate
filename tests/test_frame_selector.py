import numpy as np

from passport_quality_gate.frame_selector import BestFrameConfig, BestFrameSelector


def result(*, ready=True, blur=0.1, lowres=0.1, bbox=(10, 10, 110, 70), state="READY"):
    return {
        "capture_allowed": ready,
        "capture_quality_state": "READY" if ready else "NOT_READY",
        "state": state,
        "quality": {
            "blur_score": blur,
            "low_resolution_score": lowres,
            "too_dark_score": 0.0,
            "too_bright_score": 0.0,
            "low_contrast_score": 0.0,
            "glare_score": 0.0,
            "noise_score": 0.0,
        },
        "raw_metrics": {"motion": {"score": 0.0}},
        "confidence": 0.9,
        "localization": {"bbox": list(bbox)},
    }


def test_selects_sharper_recent_ready_frame_even_if_older():
    sel = BestFrameSelector(BestFrameConfig(window_ms=750, recency_weight=0.04))
    older = np.full((20, 30, 3), 11, np.uint8)
    newer = np.full((20, 30, 3), 22, np.uint8)
    sel.push(older, result(blur=0.02), timestamp=1.00)
    sel.push(newer, result(blur=0.55), timestamp=1.20)
    chosen = sel.select_recent(trigger_timestamp=1.25)
    assert chosen is not None
    assert int(chosen.frame[0, 0, 0]) == 11
    assert 240 <= chosen.age_ms <= 260


def test_non_ready_frames_are_not_kept_but_can_define_current_geometry():
    sel = BestFrameSelector(BestFrameConfig(window_ms=750))
    good = np.full((20, 30, 3), 9, np.uint8)
    shaken = np.full((20, 30, 3), 4, np.uint8)
    sel.push(good, result(ready=True), timestamp=2.00)
    sel.push(shaken, result(ready=False, blur=0.9, state="ADJUST"), timestamp=2.15)
    chosen = sel.select_recent(trigger_timestamp=2.20)
    assert chosen is not None
    assert int(chosen.frame[0, 0, 0]) == 9
    assert sel.buffered_frames == 1


def test_stale_ready_frame_is_not_returned():
    sel = BestFrameSelector(BestFrameConfig(window_ms=500, max_age_ms=600))
    sel.push(np.zeros((10, 10, 3), np.uint8), result(), timestamp=1.0)
    assert sel.select_recent(trigger_timestamp=1.8) is None


def test_geometry_change_rejects_old_document_position():
    sel = BestFrameSelector(BestFrameConfig(window_ms=1000, min_bbox_iou=0.5))
    sel.push(np.full((10, 10, 3), 1, np.uint8), result(bbox=(0, 0, 100, 60)), timestamp=1.0)
    sel.push(np.full((10, 10, 3), 2, np.uint8), result(ready=False, bbox=(300, 200, 400, 260)), timestamp=1.1)
    assert sel.select_recent(trigger_timestamp=1.2) is None


def test_memory_budget_bounds_buffer():
    frame = np.zeros((100, 100, 3), np.uint8)  # ~0.029 MiB
    sel = BestFrameSelector(BestFrameConfig(max_frames=20, max_memory_mb=0.07, window_ms=5000, max_age_ms=5000))
    for i in range(8):
        sel.push(frame, result(), timestamp=float(i) / 10)
    assert sel.buffered_megabytes <= 0.071
    assert sel.buffered_frames <= 3
