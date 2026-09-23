"""Image-supported quadrilaterals. Never promote a detector box or old corners.
Contours and interrupted line segments propose candidates; all four sides must
be supported in the current image. Thresholds are provisional research values.
"""
import itertools
import cv2
import numpy as np
from .geometry import order_quad, box_quad, overlap

DEFAULTS = dict(max_width=800, padding=.20, min_area_ratio=.72, max_area_ratio=1.2,
                min_bbox_iou=.70, edge_tolerance=2.5, min_side_support=.60,
                max_side_gap=.30, ambiguity_margin=.025, distinct_iou=.90,
                max_lines_per_side=3, min_line_fraction=.22)


def estimate_corners(frame, bbox, mrz=None, config=None):
    c=DEFAULTS | (config or {})
    h,w=frame.shape[:2]; b=np.asarray(bbox,np.float32)
    pad=c['padding']*max(b[2]-b[0],b[3]-b[1])
    x0,y0=np.maximum(0,np.floor(b[:2]-pad)).astype(int)
    x1,y1=np.minimum([w,h],np.ceil(b[2:]+pad)).astype(int)
    evidence=dict(method='contours_and_lines_current_frame',verified=False,
                  candidate_count=0,valid_count=0,reason='no_supported_quad')
    if x1-x0<16 or y1-y0<16: return None,evidence
    scale=min(1.,c['max_width']/max(x1-x0,y1-y0))
    roi=cv2.resize(frame[y0:y1,x0:x1],None,fx=scale,fy=scale)
    offset=np.array([x0,y0]); rb=(b-np.tile(offset,2))*scale
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    smooth=cv2.GaussianBlur(gray,(3,3),.8)
    # Two contrast levels propose contours, while unclosed edges validate them.
    edges=cv2.Canny(smooth,25,75)
    distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)

    # Even when a full quadrilateral cannot be verified, measure how much of
    # each detector-box side is actually supported by a current-frame image
    # edge.  These diagnostics are deliberately weaker than verified corners,
    # but are useful for detecting a page fragment whose visible portion is
    # still sharp enough to fool the quality gate.
    bb=box_quad(rb)
    bbox_side_support=[]; bbox_side_gap=[]; endpoint_support=[]
    for a,z in zip(bb,np.roll(bb,-1,axis=0)):
        pts=a+(z-a)*np.linspace(.03,.97,96)[:,None]
        ix=np.round(pts).astype(int)
        ix[:,0]=np.clip(ix[:,0],0,distance.shape[1]-1)
        ix[:,1]=np.clip(ix[:,1],0,distance.shape[0]-1)
        ok=distance[ix[:,1],ix[:,0]]<=c['edge_tolerance']
        bbox_side_support.append(float(ok.mean()))
        seg=max(12,len(ok)//4)
        endpoint_support.append((float(ok[:seg].mean()),float(ok[-seg:].mean())))
        longest=run=0
        for hit in ok:
            run=0 if hit else run+1
            longest=max(longest,run)
        bbox_side_gap.append(float(longest/len(ok)))
    top,right,bottom,left=endpoint_support
    bbox_corner_support=[
        min(top[0],left[1]),   # TL
        min(top[1],right[0]),  # TR
        min(right[1],bottom[0]), # BR
        min(bottom[1],left[0]),  # BL
    ]
    evidence.update(
        bbox_side_support=bbox_side_support,
        bbox_side_max_gap=bbox_side_gap,
        bbox_corner_support=bbox_corner_support,
    )
    proposals=[]
    for thresholds in [(25,75),(50,150)]:
        ed=cv2.Canny(smooth,*thresholds)
        for k in [3,7]:
            mask=cv2.morphologyEx(ed,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
            contours=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
            for cnt in sorted(contours,key=cv2.contourArea,reverse=True)[:25]:
                if cv2.contourArea(cnt)<.5*(rb[2]-rb[0])*(rb[3]-rb[1]): continue
                hull=cv2.convexHull(cnt)
                for shape in [cnt,hull]:
                    perimeter=cv2.arcLength(shape,True)
                    for eps in [.012,.025,.045]:
                        q=cv2.approxPolyDP(shape,eps*perimeter,True)
                        if len(q)==4 and cv2.isContourConvex(q): proposals.append((q.reshape(4,2),'contour'))
    # Connect long segments without requiring a single closed contour.
    bw,bh=rb[2:]-rb[:2]; center=(rb[:2]+rb[2:])/2
    lines=cv2.HoughLinesP(edges,1,np.pi/360,threshold=max(16,int(min(bw,bh)*.12)),
                         minLineLength=max(15,int(min(bw,bh)*c['min_line_fraction'])),maxLineGap=12)
    groups=[[],[],[],[]] # top,right,bottom,left
    if lines is not None:
        for v in lines.reshape(-1,4):
            a,z=v[:2].astype(float),v[2:].astype(float); d=z-a; mid=(a+z)/2
            if abs(d[1])<=.65*abs(d[0]):
                side=0 if mid[1]<center[1] else 2
                if abs(mid[1]-rb[1 if side==0 else 3])>.23*bh: continue
            elif abs(d[0])<=.65*abs(d[1]):
                side=3 if mid[0]<center[0] else 1
                if abs(mid[0]-rb[0 if side==3 else 2])>.23*bw: continue
            else: continue
            line=np.cross([*a,1],[*z,1]); line=line/max(1e-9,np.linalg.norm(line[:2]))
            groups[side].append((float(np.linalg.norm(d)),line))
        chosen=[]
        for group in groups:
            keep=[]
            for _,line in sorted(group,key=lambda x:-x[0]):
                if any(min(np.linalg.norm(line-k),np.linalg.norm(line+k))<4 for k in keep): continue
                keep.append(line)
                if len(keep)>=c['max_lines_per_side']:break
            chosen.append(keep)
        for four in itertools.product(*chosen):
            pts=[]
            for i in range(4):
                point=np.cross(four[i-1],four[i])
                if abs(point[2])<1e-6: break
                pts.append(point[:2]/point[2])
            if len(pts)==4:proposals.append((np.asarray(pts),'lines'))
    valid=[]; seen=set(); ba=abs(cv2.contourArea(bb))
    for candidate,method in proposals:
        try:q=order_quad(candidate)
        except ValueError:continue
        key=tuple(np.round(q/2).astype(int).ravel())
        if key in seen:continue
        seen.add(key)
        area=abs(cv2.contourArea(q)); ratio=area/max(1,ba)
        if not c['min_area_ratio']<=ratio<=c['max_area_ratio']:continue
        inter=overlap(q,bb); iou=inter/max(1,area+ba-inter)
        if iou<c['min_bbox_iou']:continue
        if (q<1).any() or (q[:,0]>=roi.shape[1]-1).any() or (q[:,1]>=roi.shape[0]-1).any():continue
        world=q/scale+offset
        if mrz is not None and overlap(world,mrz)/max(1,abs(cv2.contourArea(np.asarray(mrz,np.float32))))<.95:continue
        supports=[]; gaps=[]
        for a,z in zip(q,np.roll(q,-1,axis=0)):
            pts=a+(z-a)*np.linspace(.03,.97,96)[:,None]
            ix=np.round(pts).astype(int)
            ok=distance[ix[:,1],ix[:,0]]<=c['edge_tolerance']
            supports.append(float(ok.mean())); longest=run=0
            for hit in ok:
                run=0 if hit else run+1;longest=max(longest,run)
            gaps.append(longest/len(ok))
        if min(supports)<c['min_side_support'] or max(gaps)>c['max_side_gap']:continue
        score=.6*min(supports)+.25*np.mean(supports)+.15*iou
        valid.append((float(score),world.astype(np.float32),method,supports,gaps))
    evidence.update(candidate_count=len(seen),valid_count=len(valid))
    if not valid:return None,evidence
    valid.sort(key=lambda x:-x[0]); best=valid[0]
    for other in valid[1:]:
        inter=overlap(best[1],other[1]); union=abs(cv2.contourArea(best[1]))+abs(cv2.contourArea(other[1]))-inter
        if inter/max(1,union)<c['distinct_iou'] and best[0]-other[0]<c['ambiguity_margin']:
            evidence.update(reason='ambiguous_page_boundaries',score=best[0]);return None,evidence
    evidence.update(verified=True,reason='four_sides_supported',score=best[0],proposal=best[2],
                    side_support=best[3],side_max_gap=best[4])
    return best[1],evidence
