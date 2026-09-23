"""V2 deterministic text-detail evidence. No OCR, no image enhancement.
Scores are provisional severity, never probabilities of OCR success.
The focus descriptor measures how much edge energy survives extra smoothing:
blurred edges change less than already sharp edges. Noise is estimated separately.
"""
from time import perf_counter
import cv2
import numpy as np
from .geometry import clip, rectify


def ramp(value, good, bad):
    return clip((float(value)-good)/(bad-good))


def robust_patch(gray, cfg):
    # Work in native sampled pixels (or downsampled), never upscale missing detail.
    g=gray.astype(np.float32)
    p02,p90=np.percentile(g,[2,90]); contrast=float(p90-p02)
    smooth=cv2.GaussianBlur(g,(0,0),cfg['pre_sigma'])
    extra=cv2.GaussianBlur(smooth,(0,0),cfg['probe_sigma'])
    def energy(a):
        x=cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)/8
        y=cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)/8
        return x*x+y*y,x,y
    e,gx,gy=energy(smooth); e2,_,_=energy(extra)
    _,gx2,gy2=energy(extra)
    xenergy=float((gx[3:-3,3:-3]**2).sum()); yenergy=float((gy[3:-3,3:-3]**2).sum())
    rx=float((gx2[3:-3,3:-3]**2).sum()/max(1e-6,xenergy))
    ry=float((gy2[3:-3,3:-3]**2).sum()/max(1e-6,yenergy))
    e=e[3:-3,3:-3]; e2=e2[3:-3,3:-3]
    ratio=float(e2.sum()/max(1e-6,e.sum()))
    coarse=cv2.GaussianBlur(g,(0,0),1.2)
    _,cx,cy=energy(coarse); cm=np.sqrt(cx*cx+cy*cy)
    flat=cm<np.percentile(cm,40)
    residual=g-cv2.GaussianBlur(g,(0,0),.65)
    vals=residual[flat]
    noise=float(np.median(np.abs(vals-np.median(vals)))/.6745) if vals.size else 0.
    local=cv2.boxFilter(g,-1,(15,15)); ink=(local-g)>cfg['ink_delta']
    density=float(ink.mean())
    n,_,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    components=sum(2<=s[cv2.CC_STAT_AREA]<=g.size*.18 and s[cv2.CC_STAT_HEIGHT]>=2 for s in stats[1:])
    # A handful of sharp borders/portrait edges must not stand in for readable text.
    text_like=(cfg['min_ink_density']<=density<=cfg['max_ink_density'] and components>=cfg['min_components'])
    xx=float((gx*gx).mean()); yy=float((gy*gy).mean()); xy=float((gx*gy).mean())
    coherence=float(np.sqrt((xx-yy)**2+4*xy*xy)/max(1e-9,xx+yy))
    return dict(contrast=contrast,edge_retention=ratio,edge_retention_x=rx,edge_retention_y=ry,axis_energy_fraction_x=xenergy/max(1e-6,xenergy+yenergy),noise_sigma=noise,ink_density=density,
                component_count=int(components),text_like=bool(text_like),directional_coherence=coherence,
                normalized_gradient=float(np.sqrt(e.mean())/max(contrast,1.)))


def region_features(gray, mask, cfg, grid):
    if np.count_nonzero(mask)<64: return dict(available=False,reason='empty_region',patches=[])
    x,y,w,h=cv2.boundingRect(mask)
    patches=[]
    for iy in range(grid[0]):
        for ix in range(grid[1]):
            x1=x+round(ix*w/grid[1]);x2=x+round((ix+1)*w/grid[1])
            y1=y+round(iy*h/grid[0]);y2=y+round((iy+1)*h/grid[0])
            if min(x2-x1,y2-y1)<12: continue
            m=mask[y1:y2,x1:x2]
            if np.mean(m>0)<.95: continue
            p=robust_patch(gray[y1:y2,x1:x2],cfg)
            p['bbox']=[x1,y1,x2,y2];patches.append(p)
    informative=[p for p in patches if p['text_like'] and p['contrast']>=cfg['evidence_contrast_min']]
    if len(informative)<cfg['min_informative_patches']:
        return dict(available=False,reason='insufficient_text_evidence',patches=patches,
                    informative_patches=len(informative),patch_count=len(patches))
    ratios=np.array([p['edge_retention'] for p in informative])
    noise=np.array([p['noise_sigma'] for p in informative])
    contrast=np.array([p['contrast'] for p in informative])
    axis_ratios=[max(p['edge_retention_x'],p['edge_retention_y']) if .15<p['axis_energy_fraction_x']<.85 else p['edge_retention'] for p in informative]
    effective=np.maximum(ratios,np.array(axis_ratios))
    patch_scores=np.array([ramp(v,cfg['retention_good'],cfg['retention_bad']) for v in effective])
    # Tail catches localized loss; single unusual patch cannot condemn the whole page.
    blur_score=max(float(np.median(patch_scores)),float(np.percentile(patch_scores,75)))
    low_contrast=ramp(float(np.percentile(contrast,25)),cfg['contrast_good'],cfg['contrast_bad'])
    noise_score=ramp(float(np.percentile(noise,75)),cfg['noise_good'],cfg['noise_bad'])
    return dict(available=True,patches=patches,informative_patches=len(informative),patch_count=len(patches),
        edge_retention_p50=float(np.median(ratios)),edge_retention_p75=float(np.percentile(ratios,75)),
        contrast_p25=float(np.percentile(contrast,25)),noise_p75=float(np.percentile(noise,75)),
        blur_score=blur_score,low_contrast_score=low_contrast,noise_score=noise_score,
        blurred_patch_fraction=float(np.mean(patch_scores>=.6)))


def analyze_readability(frame,det,cfg):
    start=perf_counter()
    p=np.asarray(det.polygon,np.float32)
    lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    if det.mrz_polygon is not None:
        mids=(p+np.roll(p,-1,axis=0))/2
        bottom=int(np.argmin(np.linalg.norm(mids-np.asarray(det.mrz_polygon).mean(0),axis=1)))
        native_width=float((lengths[bottom]+lengths[(bottom+2)%4])/2)
    else: native_width=float(max(lengths))
    width=max(32,min(cfg['max_width'],int(native_width)))
    crop,valid,mrz,H=rectify(frame,det,width)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    # Conservative TD3 field-zone prior. It is not a trained text detector.
    text=np.zeros_like(valid)
    text[int(.15*h):int(.76*h),int(.32*w):int(.96*w)]=255
    text=cv2.bitwise_and(text,valid)
    body=region_features(gray,text,cfg,tuple(cfg['body_grid']))
    mr=region_features(gray,mrz,cfg,tuple(cfg['mrz_grid']))
    # Probe both halves of MRZ; an entirely erased side cannot disappear from aggregation.
    if np.count_nonzero(mrz):
        x,y,mw,mh=cv2.boundingRect(mrz); halves=[]
        for lo,hi in [(x,x+mw//2),(x+mw//2,x+mw)]:
            m=np.zeros_like(mrz);m[:,lo:hi]=mrz[:,lo:hi]
            halves.append(region_features(gray,m,cfg,(2,2)))
        mr['half_evidence']=[s['available'] for s in halves]
        mr['coverage_complete']=all(mr['half_evidence'])
    else: mr['coverage_complete']=False
    regions=[r for r in [body,mr] if r['available']]
    score=max([r['blur_score'] for r in regions],default=None)
    evidence=body['available'] and mr['available'] and mr['coverage_complete']
    pixel_pitch=None
    if det.mrz_polygon is not None:
        q=np.asarray(det.mrz_polygon,np.float32)
        qlength=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        pixel_pitch=float(max(qlength)/44.) # TD3 prior; not measured character segmentation.
    resolution_score=max(ramp(native_width,cfg['page_width_good'],cfg['page_width_bad']),
        ramp(pixel_pitch,cfg['mrz_pitch_good'],cfg['mrz_pitch_bad']) if pixel_pitch is not None else 0.)
    return dict(method='native_text_patch_edge_retention_v2',status='provisional_not_ocr_calibrated',
        evidence_sufficient=evidence,passport_native_width=native_width,analysis_width=width,
        mrz_pixels_per_character_prior=pixel_pitch,body=body,mrz=mr,blur_score=score,
        low_resolution_score=resolution_score,low_contrast_score=max([r['low_contrast_score'] for r in regions],default=None),
        noise_score=max([r['noise_score'] for r in regions],default=None),timing_ms=(perf_counter()-start)*1000)
