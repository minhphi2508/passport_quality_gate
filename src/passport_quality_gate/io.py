from pathlib import Path
import json
import cv2
import numpy as np
from .localization import Detection

def load_image(path):
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Cannot decode image: {path}')
    return im

def load_detection(meta):
    d=meta['detection']
    return Detection(None if d['polygon'] is None else np.array(d['polygon'],np.float32),
        None if d['mrz_polygon'] is None else np.array(d['mrz_polygon'],np.float32),
        d['confidence'],d['source'],d['corners_reliable'],d.get('mrz_confidence',0.))

def save_result(path,frame,result,debug):
    base=Path(path); base.parent.mkdir(parents=True,exist_ok=True)
    base.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    overlay=frame.copy()
    cv2.polylines(overlay,[np.array(result['guide_polygon'],np.int32)],True,(255,180,0),2)
    for key,color in [('polygon',(0,220,0)),('mrz_polygon',(240,0,240))]:
        p=result['localization'][key]
        if p is not None: cv2.polylines(overlay,[np.array(p,np.int32)],True,color,2)
    lines=[f"{result['state']} | {result.get('guidance_code') or result['primary_issue']} | {result['localization']['source']}"]
    g=result['geometry']
    if g:
        lines.append(f"fillW {g.get('fill_w',0):.2f} fillH {g.get('fill_h',0):.2f} dx {g['center_offset_x']:.2f} dy {g['center_offset_y']:.2f} angle {g['rotation_deg']:.1f}")
    for i,line in enumerate(lines):
        cv2.rectangle(overlay,(0,i*30),(overlay.shape[1],(i+1)*30),(25,25,25),-1)
        cv2.putText(overlay,line,(10,22+i*30),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
    cv2.imwrite(str(base.parent/(base.name+'_overlay.png')),overlay)
    if debug:
        cv2.imwrite(str(base.parent/(base.name+'_crop.png')),debug['crop'])
        cv2.imwrite(str(base.parent/(base.name+'_glare_mask.png')),debug['glare_mask'])
        gl=debug['crop'].copy(); m=debug['glare_mask']>0
        gl[m]=(gl[m]*.4+np.array([0,0,255])*.6).astype(np.uint8)
        cv2.imwrite(str(base.parent/(base.name+'_glare_overlay.png')),gl)
        heat=debug['blur_heatmap']
        if heat is not None:
            heat=np.log1p(heat); heat=(heat/max(1,float(heat.max()))*255).astype(np.uint8)
            colored=cv2.applyColorMap(heat,cv2.COLORMAP_VIRIDIS)
            # V4 name reflects what this debug image actually is: local sharpness/edge energy.
            cv2.imwrite(str(base.parent/(base.name+'_sharpness_heatmap.png')),colored)
            cv2.imwrite(str(base.parent/(base.name+'_blur_heatmap.png')),colored)  # legacy alias
