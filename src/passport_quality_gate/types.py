from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Union


@dataclass(frozen=True)
class GuideBox:
    """Normalized UI guide rectangle.

    Coordinates are fractions of the input frame: x, y, width and height are
    expected in [0, 1]. The quality engine itself remains resolution agnostic;
    callers are responsible for passing the guide used by their own UI.
    """

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        vals = (float(self.x), float(self.y), float(self.w), float(self.h))
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            raise ValueError("guide_box values must be finite")
        x, y, w, h = vals
        if w <= 0 or h <= 0:
            raise ValueError("guide_box width and height must be positive")
        if x < 0 or y < 0 or x + w > 1 or y + h > 1:
            raise ValueError("guide_box must fit inside normalized frame coordinates [0, 1]")

    def as_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "w": float(self.w), "h": float(self.h)}


GuideBoxLike = Union[GuideBox, Mapping[str, float], Sequence[float]]


def coerce_guide_box(value: GuideBoxLike) -> dict[str, float]:
    if isinstance(value, GuideBox):
        return value.as_dict()
    if isinstance(value, Mapping):
        try:
            box = GuideBox(value["x"], value["y"], value["w"], value["h"])
        except KeyError as exc:
            raise ValueError("guide_box mapping must contain x, y, w, h") from exc
        return box.as_dict()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return GuideBox(*value).as_dict()
    raise TypeError("guide_box must be GuideBox, mapping {x,y,w,h}, or a 4-value sequence")
