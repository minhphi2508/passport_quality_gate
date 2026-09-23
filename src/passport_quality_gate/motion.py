"""Robust time-normalized page motion for live preview.

The detector box naturally jitters a few pixels between frames.  V4.1 filters
that jitter before turning motion into HOLD_STEADY severity.
"""
import numpy as np
from .readability import ramp


class MotionAnalyzer:
    def __init__(self, config):
        self.config = config
        self.previous = None
        self.filtered_speed = 0.0

    def reset(self):
        self.previous = None
        self.filtered_speed = 0.0

    def update(self, det, shape, timestamp):
        p = np.asarray(det.polygon, np.float32) / np.array(
            [shape[1], shape[0]], np.float32
        )
        a = p[1] - p[0]
        b = p[3] - p[0]
        center = p.mean(0)
        scale = float(np.sqrt(abs(a[0] * b[1] - a[1] * b[0])))
        state = (timestamp, center, scale)

        result = dict(
            raw_speed=0.0,
            speed=0.0,
            score=0.0,
            observed=False,
            center_delta=0.0,
            scale_delta=0.0,
        )

        if self.previous is not None:
            dt = timestamp - self.previous[0]
            if dt <= 0:
                raise ValueError('Preview timestamps must strictly increase')
            if dt < 2.0:
                center_delta = float(np.linalg.norm(center - self.previous[1]))
                scale_delta = abs(scale - self.previous[2])

                # Ignore ordinary hand / detector micro-jitter before dividing
                # by dt.  The values are fractions of the full camera frame.
                dc = max(0.0, center_delta - float(self.config.get('center_deadzone', 0.010)))
                ds = max(0.0, scale_delta - float(self.config.get('scale_deadzone', 0.015)))
                raw_speed = max(dc, ds) / dt

                alpha = float(self.config.get('ema_alpha', 0.45))
                self.filtered_speed = (
                    alpha * raw_speed + (1.0 - alpha) * self.filtered_speed
                )
                score = ramp(
                    self.filtered_speed,
                    float(self.config['speed_good']),
                    float(self.config['speed_bad']),
                )
                result = dict(
                    raw_speed=raw_speed,
                    speed=self.filtered_speed,
                    score=score,
                    observed=True,
                    center_delta=center_delta,
                    scale_delta=scale_delta,
                )

        self.previous = state
        return result
