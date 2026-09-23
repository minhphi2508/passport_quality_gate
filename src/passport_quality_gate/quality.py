"""Direct image signals only: no recognition, enhancement, or learned probabilities."""
from time import perf_counter
import cv2
import numpy as np
from .geometry import clip


def exposure(crop, valid, cfg):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    v=gray[valid>0]
    if not len(v): raise ValueError('Empty valid passport ROI')
    p=np.percentile(v,[1,5,50,95,99])
    bright=float(np.mean(v>=cfg['bright_clip'])); dark=float(np.mean(v<=cfg['dark_clip']))
    return dict(mean=float(v.mean()),median=float(p[2]),p1=float(p[0]),p5=float(p[1]),p50=float(p[2]),p95=float(p[3]),p99=float(p[4]),
        dark_clipping_ratio=dark,bright_clipping_ratio=bright,dynamic_range=float(p[3]-p[1]),
        too_dark_score=clip((cfg['dark_median_ok']-p[2])/(cfg['dark_median_ok']-cfg['dark_median_fail'])),
        too_bright_score=clip((bright-cfg['bright_ratio_ok'])/(cfg['bright_ratio_fail']-cfg['bright_ratio_ok'])))


def sharpness(image, valid, cfg, advanced):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask=cv2.erode(valid,np.ones((5,5),np.uint8))>0
    if mask.sum()<16: return None
    times={}
    t=perf_counter(); lap=cv2.Laplacian(gray,cv2.CV_32F)
    lv=float(np.var(lap[mask])); times['laplacian']=(perf_counter()-t)*1000
    t=perf_counter(); gx=cv2.Sobel(gray,cv2.CV_32F,1,0); gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    ten=float(np.mean((gx*gx+gy*gy)[mask])); times['tenengrad']=(perf_counter()-t)*1000
    t=perf_counter(); diff=gray[:,2:]-gray[:,:-2]; bm=mask[:,2:] & mask[:,:-2]
    br=float(np.mean(diff[bm]**2)) if bm.any() else 0.; times['brenner']=(perf_counter()-t)*1000
    t=perf_counter(); size=cfg['patch_size']; vals=[]; heat=np.zeros(gray.shape,np.float32)
    for y in range(0,gray.shape[0],size):
        for x in range(0,gray.shape[1],size):
            m=mask[y:y+size,x:x+size]; a=lap[y:y+size,x:x+size]
            if m.sum()<16: continue
            value=float(np.var(a[m])); vals.append(value); heat[y:y+size,x:x+size]=value
    times['patches']=(perf_counter()-t)*1000
    fft=None
    if advanced and cfg['enable_fft_final']:
        t=perf_counter()
        # Hann window reduces crop-boundary energy; invalid padding filled with valid mean.
        a=np.where(mask,gray-float(gray[mask].mean()),0)
        a=a*np.hanning(a.shape[0])[:,None]*np.hanning(a.shape[1])[None,:]
        power=np.abs(np.fft.rfft2(a))**2
        fy=np.fft.fftfreq(a.shape[0])[:,None]; fx=np.fft.rfftfreq(a.shape[1])[None,:]
        fft=float(power[np.sqrt(fy*fy+fx*fx)>cfg['fft_cutoff']].sum()/max(1e-9,power.sum()))
        times['fft']=(perf_counter()-t)*1000
    return dict(laplacian=lv,tenengrad=ten,brenner=br,fft_high_frequency_ratio=fft,
        patch_p10=float(np.percentile(vals,10)),patch_p50=float(np.median(vals)),patch_p90=float(np.percentile(vals,90)),
        normalized_width=image.shape[1],timing_ms=times),heat


def blur(crop,valid,mrz,cfg,norm,mode):
    out=sharpness(crop,valid,cfg,mode=='final')
    if out is None: return dict(passport=None,mrz=None,blur_score=None,status='insufficient_pixels'),None
    page,heat=out; mr=None
    if np.count_nonzero(mrz)>16:
        x,y,w,h=cv2.boundingRect(mrz)
        width=norm['mrz_width'] if mode=='final' else min(norm['mrz_width'],norm['preview_width'])
        height=max(4,round(h*width/w))
        im=cv2.resize(crop[y:y+h,x:x+w],(width,height),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mrz[y:y+h,x:x+w],(width,height),interpolation=cv2.INTER_NEAREST)
        mout=sharpness(im,mask,cfg,mode=='final'); mr=mout[0] if mout is not None else None
    ref=cfg.get(f'{mode}_laplacian_reference'); mref=cfg.get(f'{mode}_mrz_laplacian_reference')
    score=None
    if cfg['calibrated']:
        if not ref or (mr is not None and not mref): raise ValueError('Calibrated blur requires positive page/MRZ references for this mode')
        score=clip((1-page['laplacian']/ref)/(1-cfg['reference_low_fraction']))
        if mr is not None: score=max(score,clip((1-mr['laplacian']/mref)/(1-cfg['reference_low_fraction'])))
    return dict(passport=page,mrz=mr,blur_score=score,status='calibrated' if score is not None else 'metrics_only_uncalibrated'),heat


def glare(crop,valid,mrz,cfg,mode):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma=np.linalg.norm(lab[:,:,1:]-128,axis=2)
    mean=cv2.boxFilter(gray,-1,(9,9)); std=np.sqrt(np.maximum(0,cv2.boxFilter(gray**2,-1,(9,9))-mean**2))
    k=max(9,int(crop.shape[1]*cfg['context_window_fraction'])|1)
    context=cv2.boxFilter(gray,-1,(k,k))
    clipped=crop.max(2)>=cfg['channel_clip']
    white=(gray>=cfg['luminance_min']) & (hsv[:,:,1]<=cfg['saturation_max_white'])
    colored=(hsv[:,:,2]>=cfg['value_min']) & (chroma>=cfg['chroma_min_colored']) & clipped
    candidates=(white|colored) & (std<cfg['texture_std_max']) & ((gray-context>=cfg['local_contrast_min']) | colored)
    candidates &= valid>0
    mask=(candidates.astype(np.uint8)*255)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    areas=stats[1:,cv2.CC_STAT_AREA]; keep=np.zeros_like(mask); regions=[]; count=max(1,np.count_nonzero(valid))
    mrz_count=max(1,np.count_nonzero(mrz))
    if np.count_nonzero(mrz):
        mx,my,mw,mh=cv2.boundingRect(mrz)
    else:
        mx=my=0; mw=mh=1
    delta=gray-context
    for i,a in enumerate(areas,1):
        if a/count>=cfg['min_component_ratio']:
            cmask=(labels==i) & (valid>0)
            keep[cmask]=255
            inter=cmask & (mrz>0)
            mrz_overlap=float(np.count_nonzero(inter)/mrz_count) if np.count_nonzero(mrz) else 0.0
            span_w=span_h=0.0
            if inter.any():
                ix,iy,iw,ih=cv2.boundingRect(inter.astype(np.uint8))
                span_w=float(iw/max(1,mw)); span_h=float(ih/max(1,mh))
            regions.append(dict(
                bbox=stats[i,:4].tolist(),area_ratio=float(a/count),
                mrz_overlap_ratio=mrz_overlap,mrz_span_w=span_w,mrz_span_h=span_h,
                clipping_ratio=float(clipped[cmask].mean()) if cmask.any() else 0.0,
                mean_intensity=float(gray[cmask].mean()) if cmask.any() else 0.0,
                mean_context_delta=float(delta[cmask].mean()) if cmask.any() else 0.0,
            ))
    keep=cv2.bitwise_and(keep,valid); area=float(np.count_nonzero(keep)/count)
    # A global bright wash is exposure evidence, not a localized reflection mask.
    global_highlight=area>cfg['max_local_area_ratio']
    if global_highlight: keep[:]=0; regions=[]; area=0.
    m=keep>0; mcount=np.count_nonzero(mrz)
    mo=float(np.count_nonzero(m & (mrz>0))/mcount) if mcount else None
    # This is explicitly a structural prior, NOT detected text.
    critical=np.zeros_like(keep); h,w=keep.shape
    critical[int(.18*h):int(.76*h),int(.32*w):int(.96*w)]=255
    critical=cv2.bitwise_and(critical,valid)
    co=float(np.count_nonzero(m & (critical>0))/max(1,np.count_nonzero(critical)))
    score=max(clip(area/cfg['area_fail']),clip((mo or 0)/cfg['mrz_overlap_fail']),clip(co/cfg['critical_overlap_fail']))
    return dict(glare_score=score,area_ratio=area,largest_component_ratio=max([r['area_ratio'] for r in regions],default=0.),
        mrz_overlap=mo,
        max_mrz_component_overlap=max([r['mrz_overlap_ratio'] for r in regions],default=0.),
        max_mrz_component_span_w=max([r['mrz_span_w'] for r in regions],default=0.),
        max_mrz_component_span_h=max([r['mrz_span_h'] for r in regions],default=0.),
        critical_region_overlap=co,critical_region_source='TD3_layout_prior_not_text_detection',
        clipping_inside_glare=float(clipped[m].mean()) if m.any() else 0.,
        mean_intensity=float(gray[m].mean()) if m.any() else 0.,mean_chroma=float(chroma[m].mean()) if m.any() else 0.,
        mean_texture_std=float(std[m].mean()) if m.any() else 0.,regions=regions,
        global_highlight_suppressed=global_highlight,method='deterministic_local_highlight_texture'),keep
