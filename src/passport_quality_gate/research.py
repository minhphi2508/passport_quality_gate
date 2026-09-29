"""Opt-in VNext evidence. No OCR, model, persistence, or detector invocation.

Scores are provisional severities, not calibrated probabilities. Side identities
are in camera coordinates. MRZ priors are corroboration only, never a sole veto.
"""
from collections import deque
from copy import deepcopy
from time import perf_counter

import cv2
import numpy as np

from .geometry import box_quad, order_quad, rectify, overlap
from .quality import glare

SIDES = ('top', 'right', 'bottom', 'left')
ACTIONS = dict(left='MOVE_RIGHT', right='MOVE_LEFT', top='MOVE_DOWN', bottom='MOVE_UP')


def ramp(x, lo, hi):
    return float(np.clip((x-lo)/max(1e-9, hi-lo), 0, 1))


def two_cue(cues):
    values = sorted(cues.values(), reverse=True)
    return float(np.sqrt(values[0]*values[1])) if len(values) > 1 else 0.


def boundary_evidence(frame, det, cfg):
    """Sample normals across each camera-side edge, retaining semantic labels."""
    quad = order_quad(det.polygon)
    height, width = frame.shape[:2]
    ce = det.corner_evidence or {}
    supports = ce.get('bbox_side_support', [0.]*4)
    gaps = ce.get('bbox_side_max_gap', [1.]*4)
    corner = ce.get('bbox_corner_support', [0.]*4)
    out = {}
    center = quad.mean(0)
    for i, (a, b) in enumerate(zip(quad, np.roll(quad, -1, axis=0))):
        edge = b-a
        normal = np.array([-edge[1], edge[0]]) / max(1., np.linalg.norm(edge))
        midpoint = (a+b)/2
        if np.dot(normal, center-midpoint) < 0:
            normal *= -1
        # Name by outward normal, not polygon index (rotated frames remain directional).
        if abs(normal[0]) > abs(normal[1]):
            side = 'left' if normal[0] > 0 else 'right'
        else:
            side = 'top' if normal[1] > 0 else 'bottom'
        points = a + np.linspace(.08, .92, cfg['boundary_samples'])[:, None]*edge
        inside = np.rint(points+normal*cfg['boundary_offset_px']).astype(int)
        outside = np.rint(points-normal*cfg['boundary_offset_px']).astype(int)
        valid = ((inside[:,0]>=0)&(inside[:,0]<width)&(inside[:,1]>=0)&(inside[:,1]<height)&
                 (outside[:,0]>=0)&(outside[:,0]<width)&(outside[:,1]>=0)&(outside[:,1]<height))
        color = gradient = 0.
        if valid.any():
            ip, op = inside[valid], outside[valid]
            pixels=np.concatenate([frame[ip[:,1],ip[:,0]],frame[op[:,1],op[:,0]]])[:,None,:]
            lab=cv2.cvtColor(pixels,cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
            gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY).ravel().astype(np.float32)
            n=len(ip)
            color = float(np.median(np.linalg.norm(lab[:n]-lab[n:], axis=1)))
            gradient = float(np.median(np.abs(gray[:n]-gray[n:])))
        # Existing bbox evidence is camera-side ordered top/right/bottom/left.
        j = SIDES.index(side)
        support = float(supports[j]) if len(supports)==4 else 0.
        strength = max(support, ramp(color, 0, cfg['color_transition']), ramp(gradient, 0, cfg['normal_gradient']))
        out[side] = dict(support=support, max_gap=float(gaps[j]) if len(gaps)==4 else 1.,
                         corner_support=float(corner[j]) if len(corner)==4 else 0.,
                         color_transition=color, normal_gradient=gradient,
                         sample_valid_fraction=float(valid.mean()), strength=strength,
                         state='PRESENT' if strength >= .6 else 'UNCERTAIN')
    return out


def fallback_mrz(frame, det, cfg):
    """Two aligned text rows in the lower page; no recognized text is produced."""
    crop, valid, _, matrix = rectify(frame, det, cfg['fallback_width'])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    y0 = int(gray.shape[0]*cfg['fallback_lower'])
    roi = gray[y0:]
    if min(roi.shape) < 8:
        return {'score': 0., 'polygon': None, 'rows': 0}
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)))
    gradient = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0))
    grad = np.uint8(np.clip(gradient/max(1., float(gradient.max()))*255, 0, 255))
    _, mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask[valid[y0:]==0] = 0
    # Remove page-border strokes before joining text, otherwise a vertical
    # border can connect both MRZ rows into one tall component.
    inset=max(2,int(roi.shape[1]*.025))
    mask[:,:inset]=0; mask[:,-inset:]=0; mask[-3:]=0
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    contours = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    rows = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w >= roi.shape[1]*cfg['fallback_min_width'] and 4 <= h <= roi.shape[0]*.5:
            ink = float((blackhat[y:y+h,x:x+w]>20).mean())
            if cfg['fallback_min_ink'] <= ink <= cfg['fallback_max_ink']:
                rows.append((x,y,w,h,ink))
    best = {'score': 0., 'polygon': None, 'rows': len(rows)}
    for a in rows:
        for b in rows:
            if b[1] <= a[1]+a[3]:
                continue
            x=min(a[0],b[0]); y=a[1]; w=max(a[0]+a[2], b[0]+b[2])-x; h=b[1]+b[3]-y
            aspect=w/max(1,h)
            alignment=1-abs(a[0]-b[0])/max(a[2],b[2])
            width_match=min(a[2],b[2])/max(a[2],b[2])
            row_match=min(a[3],b[3])/max(a[3],b[3])
            gap=(b[1]-a[1]-a[3])/max(a[3],b[3])
            if not cfg['fallback_min_aspect'] <= aspect <= cfg['fallback_max_aspect'] or not .15 <= gap <= 2.5:
                continue
            score=float(alignment*width_match*row_match)
            if score > best['score']:
                poly=cv2.perspectiveTransform(box_quad([x,y+y0,x+w,y+h+y0])[None], np.linalg.inv(matrix))[0]
                best=dict(score=score, polygon=poly.tolist(), rows=2, aspect=aspect, alignment=alignment, ink=[a[4],b[4]])
    return best


def prepare_detection(frame, det, cfg, min_conf):
    """Current YOLO alone is authoritative. Classical candidates are telemetry."""
    det = deepcopy(det)
    present = bool(det.found and det.mrz_polygon is not None and det.mrz_confidence >= min_conf)
    out = dict(state='STRONG' if present else 'ABSENT', credible=present,
               current_yolo_present=present, yolo_confidence=float(det.mrz_confidence),
               fallback=None, source='yolo' if present else 'none', held=False)
    if det.found and not present:
        if cfg.get('fallback_telemetry', True):
            out['fallback'] = fallback_mrz(frame, det, cfg)
        det.mrz_polygon = None
    return det, out


def content_at_cut(frame, det, cfg):
    """Text/detail components meeting the physical camera boundary.

    Restrict sampling to the localized page span. Uniform border lines and blank
    margins do not count: multiple compact ink components must meet the cut and
    extend inward, with contrast across the local strip. No guide ROI is used.
    """
    height, width = frame.shape[:2]
    p = order_quad(det.polygon)
    lo, hi = p.min(0), p.max(0)
    gaps = dict(left=lo[0], right=width-1-hi[0], top=lo[1], bottom=height-1-hi[1])
    depth = max(8, int(min(np.ptp(p, axis=0))*cfg['cut_strip_fraction']))
    result = {}
    for side in SIDES:
        contact = 1-ramp(float(gaps[side]), 0., cfg['cut_contact_px'])
        result[side] = dict(score=0., contact=contact, components=0, contrast=0.)
        if contact <= 0:
            continue
        if side in ('left', 'right'):
            y0=max(0,int(lo[1])); y1=min(height,int(hi[1])+1)
            strip=frame[y0:y1,:depth] if side=='left' else frame[y0:y1,-depth:][:,::-1]
        else:
            x0=max(0,int(lo[0])); x1=min(width,int(hi[0])+1)
            strip=frame[:depth,x0:x1] if side=='top' else frame[-depth:,x0:x1][::-1]
            strip=strip.transpose(1,0,2)
        if min(strip.shape[:2])<3:
            continue
        gray=cv2.cvtColor(np.ascontiguousarray(strip),cv2.COLOR_BGR2GRAY)
        contrast=float(np.percentile(gray,90)-np.percentile(gray,10))
        background=float(np.percentile(gray,80))
        ink=(gray < background-cfg['cut_ink_delta']).astype(np.uint8)
        n, labels, stats, _=cv2.connectedComponentsWithStats(ink)
        count=0
        for label_id, (x,y,w,h,area) in enumerate(stats[1:],1):
            if x<=cfg['cut_touch_px'] and w>=cfg['cut_min_depth_px'] and area>=cfg['cut_min_component_area'] and 2<=h<=gray.shape[0]*cfg['cut_max_component_span']:
                count+=1
                # Sparse cut text can occupy far below 10% of the whole strip.
                # Measure its own contrast, not a page-wide percentile that
                # would wash it out among harmless blank margins.
                pixels=gray[y:y+h,x:x+w][labels[y:y+h,x:x+w]==label_id]
                contrast=max(contrast,background-float(np.median(pixels)))
        detail=ramp(count,0,cfg['cut_components_hard'])*ramp(contrast,cfg['cut_contrast_soft'],cfg['cut_contrast_hard'])
        result[side]=dict(score=contact*detail, contact=contact, components=count, contrast=contrast)
    return result


def side_completeness(g, boundaries, mrz, top_content, expected_outside, cfg):
    result={}
    aspect=float(g.get('page_aspect_ratio') or cfg['page_aspect_prior'])
    for side in SIDES:
        boundary=boundaries.get(side,{})
        gap=float(g.get('frame_edge_gaps',{}).get(side,1.))
        contact=1-ramp(gap,0,cfg['edge_contact'])
        proximity=1-ramp(gap,0,cfg['edge_near'])
        other=[v.get('strength',0.) for s,v in boundaries.items() if s!=side]
        strength=boundary.get('strength',0.)
        asym=ramp((float(np.median(other)) if other else 0.)-strength,.28,.65)
        if boundary.get('sample_valid_fraction',1.) < .5:
            # Unobservable outside pixels are not proof of a missing boundary.
            asym=0.
        shape=(1-ramp(aspect,cfg['aspect_low_bad'],cfg['aspect_low_good'])) if side in ('left','right') else ramp(aspect,cfg['aspect_high_good'],cfg['aspect_high_bad'])
        # Low MRZ aspect is a layout cue, correlated with expected bounds: one group.
        mrz_shape=1-ramp(float(g.get('mrz_aspect_ratio') or 9.),5.3,7.) if side in ('left','right') and mrz.get('current_yolo_present',False) else 0.
        layout=max(float(expected_outside.get(side,0.)), mrz_shape*proximity)
        cues=dict(frame_contact=contact, boundary_asymmetry=asym, layout=layout, page_shape=shape)
        score=two_cue(cues)
        if side=='top':
            substantial=ramp(float(expected_outside.get('top_fraction',0.)),.06,cfg['top_substantial'])
            score *= max(top_content, substantial, shape)
        result[side]=dict(cut_score=score, state='MISSING' if score>=cfg['side_block'] else
                          ('PRESENT' if strength>=.6 else 'UNCERTAIN'), cues=cues,
                          border_proximity=proximity, boundary=boundary)
    return result


def expected_bounds(det, shape, cfg, credible):
    out={s:0. for s in SIDES}; out['top_fraction']=0.
    if det.mrz_polygon is None or not credible:
        return out
    m=order_quad(det.mrz_polygon)
    # Work in the MRZ's local axes then project to camera coordinates.
    u=m[1]-m[0]; width=float(np.linalg.norm(u))/cfg['mrz_width_prior']; u=u/max(1.,np.linalg.norm(u))
    v=np.array([-u[1],u[0]])
    height=width/cfg['page_aspect_prior']
    center=m.mean(0)+(0.5-cfg['mrz_center_y_prior'])*height*v
    expected=np.array([center-width/2*u-height/2*v,center+width/2*u-height/2*v,
                       center+width/2*u+height/2*v,center-width/2*u+height/2*v])
    lo,hi=expected.min(0),expected.max(0); h,w=shape[:2]
    fractions=dict(left=max(0.,-lo[0])/width,right=max(0.,hi[0]-w)/width,
                   top=max(0.,-lo[1])/height,bottom=max(0.,hi[1]-h)/height)
    out.update({s:ramp(float(f),0,cfg['expected_outside_fraction']) for s,f in fractions.items()})
    out['top_fraction']=float(fractions['top'])
    return out


class ResearchSession:
    def __init__(self, cfg):
        self.cfg=cfg
        self.reset()

    def reset(self):
        self.crops=deque(maxlen=self.cfg['temporal_crops'])
        self.last_time=None
        self.last_polygon=None
        self.guidance=None
        self.guidance_since=0.
        self.last_yolo_time=None
        self.last_yolo_polygon=None
        self.last_yolo_page=None
        self.last_text_evidence=False
        self.last_strong_motion=None

    def prepare(self, frame, det, min_conf, now, mode, *, page_reliable=True):
        det, evidence=prepare_detection(frame,det,self.cfg,min_conf)
        if not page_reliable:
            self.last_yolo_time=None
            self.last_text_evidence=False
            det.mrz_polygon=None
            evidence.update(state='ABSENT',credible=False,current_yolo_present=False,held=False,source='unreliable_page')
            return det,evidence
        if mode!='preview':
            return det,evidence
        if not det.found:
            self.last_yolo_time=None
            return det,evidence
        if evidence['current_yolo_present']:
            self.last_yolo_time=now
            self.last_yolo_polygon=np.asarray(det.mrz_polygon,np.float32).copy()
            self.last_yolo_page=order_quad(det.polygon).copy()
        elif self.last_yolo_time is not None:
            age=now-self.last_yolo_time
            page=order_quad(det.polygon)
            jump=float(np.max(np.abs(page-self.last_yolo_page))/max(1.,np.linalg.norm(np.ptp(page,axis=0))))
            if 0 <= age <= self.cfg['mrz_miss_debounce_s'] and jump <= self.cfg['mrz_hold_geometry_jump']:
                # Hold presence only; a stale rectangle cannot authorize new
                # downstream MRZ diagnoses (V3 stage-aware addendum).
                evidence.update(state='HELD',credible=True,held=True,source='recent_yolo',held_age_s=age)
        return det,evidence

    def motion_evidence(self, score, now, mode):
        if mode!='preview':
            return False
        if score >= self.cfg['guidance_strong_motion']:
            self.last_strong_motion=now
        return self.last_strong_motion is not None and 0 <= now-self.last_strong_motion <= self.cfg['guidance_motion_memory_s']

    def observe_geometry(self, frame, det, g, mrz):
        boundaries=boundary_evidence(frame,det,self.cfg)
        # Only the effective upper crop strip: text meeting the cut, not normal header text far below it.
        p=order_quad(det.polygon); lo=np.maximum(p.min(0).astype(int),0); hi=np.minimum(p.max(0).astype(int),[frame.shape[1],frame.shape[0]])
        strip=frame[lo[1]:min(hi[1],lo[1]+max(3,int((hi[1]-lo[1])*.025))),lo[0]:hi[0]]
        ink=0.
        if strip.size:
            gray=cv2.cvtColor(strip,cv2.COLOR_BGR2GRAY)
            ink=float((gray < float(np.median(gray))-30).mean())
        top_content=ramp(ink,0,self.cfg['top_ink_fraction'])
        outside=expected_bounds(det,frame.shape,self.cfg,mrz.get('current_yolo_present',False))
        sides=side_completeness(g,boundaries,mrz,top_content,outside,self.cfg)
        content=content_at_cut(frame,det,self.cfg)
        for side, info in sides.items():
            info['geometry_cut_score']=info['cut_score']
            info['content_at_cut']=content[side]
            info['cut_score']=max(info['cut_score'],content[side]['score'])
            if info['cut_score']>=self.cfg['side_block']:
                info['state']='MISSING'
        g['legacy_crop_risk']=g['crop_risk']
        g['border_proximity']={s:v['border_proximity'] for s,v in sides.items()}
        g['side_completeness']=sides
        g['crop_risk']=max(v['cut_score'] for v in sides.values())
        return dict(mrz=mrz, side_completeness=sides, cut_scores={s:v['cut_score'] for s,v in sides.items()},
                    top_content_risk=top_content, expected_outside=outside)

    def mrz_glare(self, frame, det, cfg_glare, now, mode):
        """Warp only original-frame MRZ. Candidate mask gates every local veto."""
        if det.mrz_polygon is None:
            self.crops.clear(); self.last_polygon=None
            return dict(score=0., observed=False)
        p=order_quad(det.mrz_polygon)
        lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
        width=self.cfg['mrz_width']; height=max(16,min(width//3,int(width*(lengths[1]+lengths[3])/max(1.,lengths[0]+lengths[2]))))
        H=cv2.getPerspectiveTransform(p,box_quad([0,0,width-1,height-1]))
        roi=cv2.warpPerspective(frame,H,(width,height))
        valid=cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),H,(width,height),flags=cv2.INTER_NEAREST)
        metrics,mask=glare(roi,valid,valid,cfg_glare,mode)
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        small=cv2.resize(gray,(220,40)).astype(np.float32)
        temporal=0.
        if self.last_time is not None and (now-self.last_time>self.cfg['temporal_gap_s'] or
                (self.last_polygon is not None and np.max(np.abs(p-self.last_polygon))>np.linalg.norm(np.ptp(p,axis=0))*.15)):
            self.crops.clear()
        if mode=='preview' and self.crops:
            previous=self.crops[-1]
            shift,response=cv2.phaseCorrelate(previous,small)
            if response>=self.cfg['temporal_alignment_response'] and max(abs(x) for x in shift)<self.cfg['temporal_shift_px']:
                aligned=cv2.warpAffine(previous,np.float32([[1,0,shift[0]],[0,1,shift[1]]]),(220,40),borderMode=cv2.BORDER_REFLECT)
                cm=cv2.resize(mask,(220,40),interpolation=cv2.INTER_NEAREST)>0
                if cm.any():
                    temporal=ramp(float(np.maximum(small-aligned,0)[cm].mean()),8,35)
        if mode=='preview':
            self.crops.append(small); self.last_time=now; self.last_polygon=p.copy()
        edges=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0))
        cells=[]
        for row in range(2):
            for col in range(self.cfg['mrz_cells']):
                y1,y2=round(row*height/2),round((row+1)*height/2)
                x1,x2=round(col*width/self.cfg['mrz_cells']),round((col+1)*width/self.cfg['mrz_cells'])
                cell=gray[y1:y2,x1:x2]; candidate=mask[y1:y2,x1:x2]>0
                vm=valid[y1:y2,x1:x2]>0
                if not vm.any():
                    continue
                coverage=float(candidate[vm].mean()); clipping=float((cell[vm]>=250).mean())
                contrast=float(np.std(cell[vm])); edge_density=float((edges[y1:y2,x1:x2][vm]>35).mean())
                # Bright clean paper has no local candidate or no optical detail loss.
                # Require joint contrast/stroke loss, not either weak metric alone.
                # Compare with neighboring cells to avoid treating blank gutters
                # between intact characters as erased text.
                nx1=max(0,x1-(x2-x1)); nx2=min(width,x2+(x2-x1))
                neighbor=np.concatenate([gray[y1:y2,nx1:x1].ravel(),gray[y1:y2,x2:nx2].ravel()])
                neighbor_ink=float((neighbor < self.cfg['glare_ink_level']).mean()) if neighbor.size else 0.
                texture_loss=min(1-ramp(contrast,self.cfg['glare_contrast_low'],self.cfg['glare_contrast_good']),
                                 1-ramp(edge_density,self.cfg['glare_edges_low'],self.cfg['glare_edges_good']))
                text_context=ramp(neighbor_ink,self.cfg['glare_neighbor_ink_soft'],self.cfg['glare_neighbor_ink_hard'])
                damage=texture_loss*max(text_context,temporal)
                risk=ramp(coverage,self.cfg['cell_candidate_soft'],self.cfg['cell_candidate_hard'])*ramp(clipping,self.cfg['cell_clip_soft'],self.cfg['cell_clip_hard'])*damage
                # Temporal evidence strengthens a spatial/optical candidate, never creates one.
                risk=min(1.,risk*(1+.15*temporal))
                cells.append(dict(row=row,col=col,score=risk,candidate_coverage=coverage,clipping=clipping,
                                  contrast=contrast,edge_density=edge_density,damage=damage,neighbor_ink=neighbor_ink))
        worst=max(cells,key=lambda x:x['score'],default={'score':0.})
        return dict(score=float(worst['score']), observed=True, worst_cell=worst, cells=cells,
                    temporal_reflection=temporal, candidate_metrics=metrics, roi_size=[width,height])

    def guidance_for(self, default, active, scores, research, now, mode):
        if research.get('page_reliable') is False:
            self.guidance='PLACE_PASSPORT_IN_FRAME'
            self.guidance_since=now
            return dict(default,code='PLACE_PASSPORT_IN_FRAME',source_issue='PASSPORT_NOT_FOUND',severity=1.)
        if research.get('mrz',{}).get('current_yolo_present') is False:
            active=[c for c in active if c!='MRZ_GLARE']
            scores=dict(scores,MRZ_GLARE=0.)
            if default.get('source_issue')=='MRZ_GLARE' or default.get('code') in ('MRZ_GLARE','REDUCE_REFLECTION_ON_MRZ'):
                default=dict(default,code='HOLD_STEADY' if research['mrz'].get('held') else 'SHOW_BOTTOM_TEXT')
        candidates={}
        cuts=research.get('cut_scores',{})
        missing=[s for s,v in cuts.items() if v>=self.cfg['side_block']]
        if missing and any(c in active for c in ('CROPPED','DOCUMENT_INCOMPLETE')):
            code=ACTIONS[missing[0]] if len(missing)==1 else 'SHOW_ALL_EDGES'
            candidates[code]=(3.,'DOCUMENT_INCOMPLETE')
        mapping={'MRZ_GLARE':'TILT_TO_REMOVE_BOTTOM_REFLECTION','GLARE':'REDUCE_REFLECTION','TOO_DARK':'INCREASE_LIGHT',
                 'TOO_BRIGHT':'REDUCE_DIRECT_LIGHT','HOLD_STEADY':'HOLD_STEADY','BLUR':'WAIT_FOR_FOCUS',
                 'LOCALIZATION_UNCERTAIN':'CENTER_AND_HOLD','MRZ_NOT_FOUND':'SHOW_BOTTOM_TEXT',
                 'LOW_RESOLUTION':'MOVE_CLOSER','PASSPORT_TOO_FAR':'MOVE_CLOSER',
                 'QUALITY_UNCERTAIN':'HOLD_STEADY','CHECKING_STABILITY':'HOLD_STEADY'}
        independent=any(v>=.85 for v in cuts.values()) or any(
            side.get('content_at_cut',{}).get('score',0.)>=self.cfg['side_block']
            for side in research.get('side_completeness',{}).values()) or any(float(scores.get(c) or 0.) >= threshold for c,threshold in
                         [('MRZ_GLARE',.6),('GLARE',.6),('TOO_DARK',.85),('TOO_BRIGHT',.85)])
        causal_motion=research.get('recent_strong_motion',False) and not independent and any(c in active for c in
                      ('HOLD_STEADY','BLUR','MRZ_NOT_FOUND','LOCALIZATION_UNCERTAIN','QUALITY_UNCERTAIN','PASSPORT_NOT_FOUND'))
        if causal_motion:
            candidates['HOLD_STEADY']=(6.,'HOLD_STEADY')
        for issue in active:
            if issue=='MRZ_GLARE' and research.get('mrz',{}).get('current_yolo_present') is False:
                continue
            if issue=='QUALITY_UNCERTAIN' and any(c!='QUALITY_UNCERTAIN' for c in active):
                continue
            if issue in ('CROPPED','DOCUMENT_INCOMPLETE') and missing:
                continue
            code=mapping.get(issue, issue)
            if issue=='BLUR' and research.get('motion_score',0.) >= .3:
                code='HOLD_STEADY'
            base=3. if issue=='MRZ_GLARE' else (2.2 if issue in ('TOO_DARK','TOO_BRIGHT') else
                   (2. if issue in ('GLARE','HOLD_STEADY','MRZ_NOT_FOUND') else 1.))
            rank=base+float(scores.get(issue) or 0.)
            if issue=='PASSPORT_NOT_FOUND': rank=5.
            if code not in candidates or rank>candidates[code][0]:
                candidates[code]=(rank,issue)
        if not candidates:
            self.guidance=None
            code=mapping.get(default.get('code'),default.get('code'))
            if mode=='final' and default.get('source_issue') is None:
                code='READY'
            return dict(default,code=code)
        chosen=max(candidates,key=lambda code:candidates[code][0])
        # Never preserve recovered causes, or hide a new critical instruction.
        if mode=='preview' and self.guidance in candidates and now-self.guidance_since<self.cfg['guidance_min_s']:
            if candidates[chosen][0]<candidates[self.guidance][0]+self.cfg['guidance_switch_margin']:
                chosen=self.guidance
        if chosen!=self.guidance:
            self.guidance=chosen; self.guidance_since=now
        source=candidates[chosen][1]
        return dict(default,code=chosen,source_issue=source,severity=scores.get(source))


GUIDANCE_TEXT = {
    'READY': 'Ready to capture',
    'MOVE_CLOSER': 'Move the passport closer',
    'MOVE_RIGHT': 'Move the passport to the right',
    'MOVE_LEFT': 'Move the passport to the left',
    'MOVE_DOWN': 'Move the passport down',
    'MOVE_UP': 'Move the passport up',
    'SHOW_BOTTOM_TEXT': 'Make sure the two lines of text at the bottom are visible',
    'TILT_TO_REMOVE_BOTTOM_REFLECTION': 'Tilt the passport to remove reflection from the bottom text',
    'HOLD_STEADY': 'Hold the passport and camera steady',
    'CENTER_AND_HOLD': 'Center the passport and hold it steady',
    'SHOW_ALL_EDGES': 'Bring the whole passport page into view',
    'REDUCE_REFLECTION': 'Tilt the passport away from direct light',
    'INCREASE_LIGHT': 'Move to a brighter place',
    'REDUCE_DIRECT_LIGHT': 'Move away from direct light',
    'WAIT_FOR_FOCUS': 'Hold steady while the camera focuses',
    'PASSPORT_TOO_CLOSE': 'Move the passport farther from the camera',
    'ROTATED': 'Rotate the passport until its text is horizontal',
    'PERSPECTIVE_TOO_HIGH': 'Hold the camera parallel to the passport page',
    'LOW_CONTRAST': 'Use even lighting so the text is clear',
    'NOISE': 'Use brighter, even lighting and hold steady',
    'PASSPORT_NOT_FOUND': 'Bring the passport page into view',
    'PLACE_PASSPORT_IN_FRAME': 'Place the passport page in the camera frame',
}


def guidance_text(code):
    return GUIDANCE_TEXT.get(code, 'Center the passport and hold it steady')
