from dataclasses import dataclass, field
from .corners import estimate_corners
from pathlib import Path
import cv2
import numpy as np
from .geometry import box_quad, order_quad, overlap

@dataclass
class Detection:
    polygon: object = None
    mrz_polygon: object = None
    confidence: float = 0.0
    source: str = 'yolo'
    corners_reliable: bool = False
    mrz_confidence: float = 0.0
    corner_evidence: dict = field(default_factory=dict)
    @property
    def found(self):
        return self.polygon is not None
    def as_dict(self):
        p=None if self.polygon is None else np.asarray(self.polygon)
        return dict(source=self.source,polygon=None if p is None else p.tolist(),
          bbox=None if p is None else [*p.min(0).tolist(),*p.max(0).tolist()],
          mrz_polygon=None if self.mrz_polygon is None else np.asarray(self.mrz_polygon).tolist(),
          confidence=self.confidence,mrz_confidence=self.mrz_confidence,corners_reliable=self.corners_reliable,corner_evidence=self.corner_evidence)


def refine_corners(frame, bbox, min_ratio, mrz=None, mrz_inside=.95):
    q,_=estimate_corners(frame,bbox,mrz,{'min_area_ratio':min_ratio})
    return q

class YoloLocalizer:
    def __init__(self, weights, config, device='cpu', model=None):
        if model is None:
            if not Path(weights).is_file(): raise FileNotFoundError(f'Detector weights missing: {weights}')
            from ultralytics import YOLO
            model=YOLO(str(weights))
        self.model=model; self.config=config; self.device=device
        names=set(model.names.values())
        if not {'passport_page','mrz'}<=names: raise ValueError(f'Unexpected detector classes: {model.names}')
    def locate(self,frame,mode='final'):
        r=self.model.predict(source=frame,device=self.device,imgsz=self.config['imgsz'],
             conf=self.config['confidence'],verbose=False,save=False)[0]
        pages=[]; mrzs=[]
        if r.boxes is not None:
            for b in r.boxes:
                name=r.names[int(b.cls.item())]
                item=(float(b.conf.item()),np.asarray(b.xyxy[0].cpu().tolist(),np.float32))
                if name=='passport_page': pages.append(item)
                elif name=='mrz': mrzs.append(item)
        if not pages: return Detection()
        conf,bbox=max(pages,key=lambda b:b[0]); p=box_quad(bbox)
        # Associate MRZ with selected page; do not take an unrelated global maximum.
        valid=[(c,box_quad(b)) for c,b in mrzs if overlap(p,box_quad(b))/max(1,(b[2]-b[0])*(b[3]-b[1]))>=self.config['min_mrz_inside']]
        mc,m=max(valid,key=lambda b:b[0]) if valid else (0.,None)
        q,ce=estimate_corners(frame,bbox,m,self.config.get('corners'))
        return Detection(q if q is not None else p,m,conf,'yolo',q is not None,mc,ce)

class ProxyLocalizer:
    """Synthetic smoke fallback ONLY. A rectangle is not proof of passport identity."""
    def locate(self,frame,mode='final'):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in sorted(contours,key=cv2.contourArea,reverse=True):
            if cv2.contourArea(c)<frame.shape[0]*frame.shape[1]*.03: continue
            q=cv2.approxPolyDP(c,.02*cv2.arcLength(c,True),True)
            if len(q)==4 and cv2.isContourConvex(q):
                return Detection(order_quad(q),None,.5,'synthetic_proxy',True)
        return Detection(source='synthetic_proxy')
