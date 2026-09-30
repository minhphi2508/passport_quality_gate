"""Explicit product capture boundary, independent of advisory guide_box.

Rectangles use normalized (x,y,w,h) in the oriented, unmirrored camera buffer.
Crop uses inward rounding: pixels outside the visible rectangle never enter ROI.
"""
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np


@dataclass(frozen=True)
class CaptureViewport:
    x: float
    y: float
    w: float
    h: float

    def pixel_rect(self, shape):
        values=(self.x,self.y,self.w,self.h)
        if not all(math.isfinite(v) for v in values) or self.w<=0 or self.h<=0:
            raise ValueError('Viewport must be finite with positive size')
        height,width=shape[:2]
        x0=max(0,min(width,math.ceil(self.x*width-1e-9)))
        y0=max(0,min(height,math.ceil(self.y*height-1e-9)))
        x1=max(0,min(width,math.floor((self.x+self.w)*width+1e-9)))
        y1=max(0,min(height,math.floor((self.y+self.h)*height+1e-9)))
        if x1-x0<8 or y1-y0<8:
            raise ValueError('Resolved viewport must be at least 8x8 pixels')
        return x0,y0,x1,y1

    def extract(self, frame, *, transform=None):
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError('Camera frame must be uint8 HxWx3 BGR')
        x0,y0,x1,y1=self.pixel_rect(frame.shape)
        roi=frame[y0:y1,x0:x1].copy()
        metadata=dict(viewport_normalized=[self.x,self.y,self.w,self.h],
            viewport_pixel_rect=[x0,y0,x1,y1],analysis_frame_size={'width':x1-x0,'height':y1-y0},
            raw_frame_size={'width':frame.shape[1],'height':frame.shape[0]},
            preview_transform=deepcopy(transform or {'rotation_clockwise':0,'mirrored':False,'mode':'buffer'}))
        return roi,metadata


def orient_camera(frame, rotation_clockwise=0):
    if rotation_clockwise not in (0,90,180,270):
        raise ValueError('Rotation must be 0, 90, 180 or 270')
    return np.ascontiguousarray(np.rot90(frame,-rotation_clockwise//90))


def map_preview_viewport(rect, preview_size, sensor_size, *, rotation_clockwise=0, mirrored=False, mode='fit'):
    """Map screen pixel (x,y,w,h) through FIT/FILL and preview-only mirror.

    Returns a viewport in orient_camera(sensor, rotation) coordinates and metadata.
    FIT letterbox intersections are clamped to valid camera pixels at extraction.
    Host must supply the actual preview transform, not inferred screen bounds.
    """
    if rotation_clockwise not in (0,90,180,270) or mode not in ('fit','fill'):
        raise ValueError('Invalid preview transform')
    pw,ph=preview_size; sw,sh=sensor_size
    if min(pw,ph,sw,sh)<=0 or not all(math.isfinite(v) for v in (*preview_size,*sensor_size,*rect)):
        raise ValueError('Invalid preview geometry')
    bw,bh=(sh,sw) if rotation_clockwise in (90,270) else (sw,sh)
    scale=(min if mode=='fit' else max)(pw/bw,ph/bh)
    ox,oy=(pw-bw*scale)/2,(ph-bh*scale)/2
    x,y,w,h=rect
    if w<=0 or h<=0: raise ValueError('Viewport size must be positive')
    x0=(x-ox)/scale; y0=(y-oy)/scale
    if mirrored: x0=bw-(x+w-ox)/scale
    viewport=CaptureViewport(x0/bw,y0/bh,w/scale/bw,h/scale/bh)
    meta=dict(rotation_clockwise=rotation_clockwise,mirrored=bool(mirrored),mode=mode,
        preview_size=list(preview_size),sensor_size=list(sensor_size),scale=scale,offset=[ox,oy],screen_rect=list(rect))
    return viewport,meta
