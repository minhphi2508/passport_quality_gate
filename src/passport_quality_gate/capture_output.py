from __future__ import annotations

import cv2
import numpy as np

from .geometry import order_quad


def extract_passport_page(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Extract the detected passport page from the exact frame that was
    final-checked.

    Uses the final localization polygon. If MRZ localization exists,
    orient the crop so the MRZ-adjacent edge becomes the bottom edge.
    """
    loc = result.get("localization") or {}
    polygon = loc.get("polygon")

    if polygon is None:
        raise ValueError("No passport polygon available")

    p = order_quad(np.asarray(polygon, dtype=np.float32))

    mrz_polygon = loc.get("mrz_polygon")
    if mrz_polygon is not None:
        mrz = np.asarray(mrz_polygon, dtype=np.float32).reshape(-1, 2)
        mrz_center = mrz.mean(axis=0)

        mids = (p + np.roll(p, -1, axis=0)) / 2
        mrz_edge = int(np.argmin(np.linalg.norm(mids - mrz_center, axis=1)))

        # Same orientation convention as the Golden rectification path:
        # MRZ-adjacent page edge becomes bottom.
        p = np.roll(p, 2 - mrz_edge, axis=0)

    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)

    width = max(32, int(round((lengths[0] + lengths[2]) / 2)))
    height = max(32, int(round((lengths[1] + lengths[3]) / 2)))

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(p, dst)

    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )