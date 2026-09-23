from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any, Deque, Optional

import numpy as np


@dataclass(frozen=True)
class BestFrameConfig:
    """Bounded in-memory recent-frame selection.

    The selector is deliberately independent of camera resolution and hardware.
    The caller decides which frames are analyzed and pushed. Frames are never
    written to disk by this utility.
    """

    window_ms: float = 750.0
    max_age_ms: float = 900.0
    max_frames: int = 12
    max_memory_mb: float = 96.0
    min_bbox_iou: float = 0.40
    recency_weight: float = 0.04

    def __post_init__(self) -> None:
        if self.window_ms <= 0 or self.max_age_ms <= 0:
            raise ValueError("window_ms and max_age_ms must be positive")
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if not 0 <= self.min_bbox_iou <= 1:
            raise ValueError("min_bbox_iou must be in [0, 1]")
        if self.recency_weight < 0:
            raise ValueError("recency_weight must be non-negative")


@dataclass
class SelectedFrame:
    frame: np.ndarray
    result: dict[str, Any]
    timestamp: float
    age_ms: float
    quality_score: float
    selection_score: float

    def metadata(self) -> dict[str, Any]:
        return {
            "selected_frame_age_ms": round(float(self.age_ms), 3),
            "quality_score": round(float(self.quality_score), 6),
            "selection_score": round(float(self.selection_score), 6),
            "capture_quality_state": self.result.get("capture_quality_state"),
            "capture_allowed": bool(self.result.get("capture_allowed")),
        }


@dataclass
class _Candidate:
    frame: np.ndarray
    result: dict[str, Any]
    timestamp: float
    quality_score: float
    nbytes: int


class BestFrameSelector:
    """Keep only a small rolling RAM buffer and return the best recent READY frame.

    This solves the common pre-shutter problem where the frame at the exact
    button press is slightly worse than a frame shortly before it. The selector
    does not open/manage a camera and does not save images.
    """

    def __init__(self, config: Optional[BestFrameConfig] = None) -> None:
        self.config = config or BestFrameConfig()
        self._items: Deque[_Candidate] = deque()
        self._bytes = 0
        self._latest_result: Optional[dict[str, Any]] = None
        self._latest_timestamp: Optional[float] = None

    @staticmethod
    def _score01(value: Any, unknown: float = 0.35) -> float:
        if value is None:
            return unknown
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return unknown

    @classmethod
    def quality_score(cls, result: dict[str, Any]) -> float:
        """Rank READY frames using existing Golden FP2 metrics only.

        Lower defect scores are better. Glare intentionally has a small weight
        because FP2 glare is known to be imperfect and should not dominate
        best-frame ranking.
        """
        q = result.get("quality") or {}
        raw = result.get("raw_metrics") or {}
        motion = (raw.get("motion") or {}).get("score")
        exposure = max(
            cls._score01(q.get("too_dark_score"), 0.25),
            cls._score01(q.get("too_bright_score"), 0.25),
        )
        penalties = (
            0.42 * cls._score01(q.get("blur_score"))
            + 0.30 * cls._score01(q.get("low_resolution_score"))
            + 0.10 * exposure
            + 0.08 * cls._score01(motion, 0.25)
            + 0.05 * cls._score01(q.get("low_contrast_score"), 0.25)
            + 0.03 * cls._score01(q.get("glare_score"), 0.20)
            + 0.02 * cls._score01(q.get("noise_score"), 0.20)
        )
        score = 1.0 - penalties
        try:
            score += 0.03 * max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            pass
        if result.get("capture_quality_state") == "OPTIMAL":
            score += 0.02
        return max(0.0, min(1.05, score))

    @staticmethod
    def _bbox(result: Optional[dict[str, Any]]) -> Optional[tuple[float, float, float, float]]:
        if not result:
            return None
        bbox = (result.get("localization") or {}).get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0
        self._latest_result = None
        self._latest_timestamp = None

    def _prune(self, now: float) -> None:
        max_age_s = self.config.max_age_ms / 1000.0
        while self._items and now - self._items[0].timestamp > max_age_s:
            old = self._items.popleft()
            self._bytes -= old.nbytes
        budget = int(self.config.max_memory_mb * 1024 * 1024)
        while self._items and (len(self._items) > self.config.max_frames or self._bytes > budget):
            old = self._items.popleft()
            self._bytes -= old.nbytes

    def push(self, frame: np.ndarray, result: dict[str, Any], timestamp: Optional[float] = None) -> None:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be uint8 HxWx3")
        t = monotonic() if timestamp is None else float(timestamp)
        self._latest_result = deepcopy(result)
        self._latest_timestamp = t

        # Only READY/capture-allowed frames are worth retaining for OCR handoff.
        if bool(result.get("capture_allowed")):
            stored = frame.copy()
            candidate = _Candidate(
                frame=stored,
                result=deepcopy(result),
                timestamp=t,
                quality_score=self.quality_score(result),
                nbytes=int(stored.nbytes),
            )
            self._items.append(candidate)
            self._bytes += candidate.nbytes
        self._prune(t)

    def select_recent(self, trigger_timestamp: Optional[float] = None) -> Optional[SelectedFrame]:
        now = monotonic() if trigger_timestamp is None else float(trigger_timestamp)
        self._prune(now)
        window_s = self.config.window_ms / 1000.0
        latest_bbox = self._bbox(self._latest_result)
        candidates: list[_Candidate] = []
        for item in self._items:
            age = now - item.timestamp
            if age < -1e-6 or age > window_s:
                continue
            if latest_bbox is not None:
                item_bbox = self._bbox(item.result)
                if item_bbox is not None and self._iou(latest_bbox, item_bbox) < self.config.min_bbox_iou:
                    continue
            candidates.append(item)
        if not candidates:
            return None

        def rank(item: _Candidate) -> float:
            age = max(0.0, now - item.timestamp)
            freshness = max(0.0, 1.0 - age / max(window_s, 1e-6))
            return item.quality_score + self.config.recency_weight * freshness

        best = max(candidates, key=rank)
        age_ms = max(0.0, (now - best.timestamp) * 1000.0)
        return SelectedFrame(
            frame=best.frame.copy(),
            result=deepcopy(best.result),
            timestamp=best.timestamp,
            age_ms=age_ms,
            quality_score=best.quality_score,
            selection_score=rank(best),
        )

    @property
    def buffered_frames(self) -> int:
        return len(self._items)

    @property
    def buffered_megabytes(self) -> float:
        return self._bytes / (1024 * 1024)
