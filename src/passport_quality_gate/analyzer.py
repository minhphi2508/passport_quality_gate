from time import perf_counter, monotonic
import numpy as np
from .config import load_config
from .geometry import guide_polygon, analyze_geometry, rectify, order_quad
from .quality import exposure, blur, glare
from .readability import analyze_readability
from .decision import DecisionEngine, TemporalStabilizer
from .motion import MotionAnalyzer


_GEOMETRY_ISSUES = {
    'CROPPED','DOCUMENT_INCOMPLETE','PASSPORT_TOO_FAR','PASSPORT_TOO_CLOSE','MOVE_LEFT','MOVE_RIGHT',
    'MOVE_UP','MOVE_DOWN','ROTATED','PERSPECTIVE_TOO_HIGH','LOCALIZATION_UNCERTAIN','MRZ_NOT_FOUND'
}
_QUALITY_ISSUES = {'TOO_DARK','TOO_BRIGHT','LOW_RESOLUTION','BLUR','GLARE','LOW_CONTRAST','NOISE','QUALITY_UNCERTAIN'}


def _ramp01(value, low, high):
    if value is None:
        return 0.0
    value=float(value)
    low=float(low); high=float(high)
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value-low)/(high-low)))


def _confirmed_glare_score(glare_metrics, readability, cfg):
    """Convert broad highlight candidates into OCR-impact glare evidence.

    The legacy glare mask intentionally over-detects bright/low-texture regions.
    It remains useful as a candidate/debug mask, but must not veto capture unless
    there is evidence that the highlight is actually destroying OCR-critical
    detail (MRZ or text readability).
    """
    raw=glare_metrics.get('glare_score')
    if raw is None:
        return None
    icfg=cfg.get('impact_confirmation', {})
    if not icfg.get('enabled', True):
        return float(raw)

    mrz=float(glare_metrics.get('mrz_overlap') or 0.0)
    critical=float(glare_metrics.get('critical_region_overlap') or 0.0)
    area=float(glare_metrics.get('area_ratio') or 0.0)
    clipped=float(glare_metrics.get('clipping_inside_glare') or 0.0)
    blur=float((readability or {}).get('blur_score') or 0.0)
    low_contrast=float((readability or {}).get('low_contrast_score') or 0.0)
    damage=max(blur, low_contrast)

    mrz_risk=(
        _ramp01(mrz, icfg.get('mrz_overlap_soft',0.08), icfg.get('mrz_overlap_hard',0.35))
        * _ramp01(clipped, icfg.get('clip_soft',0.25), icfg.get('clip_hard',0.75))
    )

    # FP2.1 targeted MRZ rule: the broad aggregate mask remains permissive,
    # but one *connected* glare-like component that materially crosses the MRZ
    # is treated much more strictly.  This avoids FP3's failure mode where the
    # aggregate MRZ overlap of ordinary bright paper drove glare to ~1.0, while
    # still catching a localized reflection strip/blob over OCR-critical MRZ.
    component_risk=0.0
    ccfg=icfg.get('mrz_component', {})
    if ccfg.get('enabled', True):
        for region in glare_metrics.get('regions') or []:
            ov=float(region.get('mrz_overlap_ratio') or 0.0)
            sw=float(region.get('mrz_span_w') or 0.0)
            sh=float(region.get('mrz_span_h') or 0.0)
            delta=float(region.get('mean_context_delta') or 0.0)
            rclip=float(region.get('clipping_ratio') or 0.0)
            overlap_strength=_ramp01(ov,ccfg.get('overlap_soft',0.012),ccfg.get('overlap_hard',0.045))
            span_strength=(
                _ramp01(sw,ccfg.get('span_w_soft',0.05),ccfg.get('span_w_hard',0.16))
                * _ramp01(sh,ccfg.get('span_h_soft',0.20),ccfg.get('span_h_hard',0.55))
            )
            spatial=max(overlap_strength,span_strength)
            optical=max(
                _ramp01(delta,ccfg.get('delta_soft',9.0),ccfg.get('delta_hard',24.0)),
                _ramp01(rclip,ccfg.get('clip_soft',0.15),ccfg.get('clip_hard',0.55)),
            )
            floor=float(ccfg.get('reliability_floor',0.72))
            component_risk=max(component_risk,spatial*(floor+(1.0-floor)*optical))
    body_risk=(
        _ramp01(critical, icfg.get('critical_overlap_soft',0.20), icfg.get('critical_overlap_hard',0.60))
        * damage
    )
    area_risk=(
        _ramp01(area, icfg.get('area_soft',0.20), icfg.get('area_hard',0.45))
        * damage
    )
    extreme_risk=(
        _ramp01(critical, icfg.get('extreme_critical_soft',0.45), icfg.get('extreme_critical_hard',0.75))
        * _ramp01(clipped, icfg.get('extreme_clip_soft',0.55), icfg.get('extreme_clip_hard',0.85))
    )
    confirmed=max(mrz_risk, component_risk, body_risk, area_risk, extreme_risk)
    return min(float(raw), float(confirmed))


class Analyzer:
    """One instance per stream. BGR uint8 inputs; backwards-compatible V3 keys remain."""
    def __init__(self, localizer, config=None):
        self.config = load_config(config)
        self.localizer = localizer
        self.engine = DecisionEngine(self.config)
        self.temporal = TemporalStabilizer(self.engine)
        self.motion = MotionAnalyzer(self.config['motion'])
        self.debug = {}

    def reset(self):
        self.temporal.reset(); self.motion.reset(); self.debug = {}

    def _context(self, mode, capture_context):
        if capture_context is None:
            return 'live_preview' if mode == 'preview' else 'full_frame_final'
        if capture_context not in self.config['policy']:
            raise ValueError('capture_context must be live_preview, full_frame_final or document_crop')
        if mode == 'preview' and capture_context != 'live_preview':
            raise ValueError('preview mode requires capture_context=live_preview')
        return capture_context

    def _advisories(self, evidence, g, scores, context):
        out=[]
        if evidence and evidence.get('page_confident') and not evidence.get('corners_verified'):
            out.append('PERSPECTIVE_UNVERIFIED')
        policy=self.config['policy'][context]
        blocking=set(policy.get('blocking', []))
        # V4.2: an advisory can be either an active non-blocking issue (for
        # example the page is smaller than the ideal guide but still readable)
        # or a near-threshold quality warning.  This keeps UI composition advice
        # separate from actual capture eligibility.
        for code in policy.get('advisory_candidates', []):
            value=scores.get(code)
            if value is None:
                continue
            active=self.engine.active(code,value)
            if active and code not in blocking:
                out.append(code)
                continue
            if active:
                continue
            floor=float(self.config['decision'].get('advisory_fraction',.65))*self.engine.enter_threshold(code)
            if value>=floor:
                out.append(code)
        return list(dict.fromkeys(out))

    def _physical_recovery_guidance(self, g):
        fg=g.get('frame_edge_gaps') or {}
        margin=float(self.config['geometry'].get('recovery_border_margin',.045))
        mapping={'left':'MOVE_RIGHT','right':'MOVE_LEFT','top':'MOVE_DOWN','bottom':'MOVE_UP'}
        candidates=[]
        for edge,action in mapping.items():
            gap=fg.get(edge)
            if gap is not None and float(gap)<margin:
                sev=max(0.,min(1.,(margin-float(gap))/max(.005,margin)))
                candidates.append((sev,action,edge))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        best=candidates[0]
        dominant=[x for x in candidates if x[0]>=best[0]-.15]
        return best[1] if len(dominant)==1 else None

    def _guidance(self, issue, g, context):
        mapping=self.config['guidance']['map']
        code=mapping.get(issue, issue)
        if issue=='LOW_RESOLUTION' and g.get('fill_w',0)>=.85:
            code='CAMERA_RESOLUTION_LOW'
        if issue=='LOCALIZATION_UNCERTAIN':
            code='SHOW_ALL_EDGES'
        if issue=='DOCUMENT_INCOMPLETE':
            # Prefer an actionable direction only when a physical camera edge
            # proves which side is missing.  Do NOT infer the direction from
            # the centre of a visible fragment: a left-side fragment of a page
            # can have its centre to the left while the correct recovery action
            # is to move the full document left, i.e. the opposite of ordinary
            # alignment.  If the missing side is not provable, ask for all edges.
            code=self._physical_recovery_guidance(g) or 'SHOW_ALL_EDGES'
        # A physically cropped page is a real blocker, but the partner-facing
        # action should be directional whenever one missing edge is clear.
        if issue=='CROPPED' and context=='live_preview':
            code=self._physical_recovery_guidance(g) or 'SHOW_ALL_EDGES'
        severity=None if issue in (None,'READY') else None
        return dict(code='READY' if issue=='READY' else code, source_issue=issue, capture_context=context, severity=severity)

    def _advisory_guidance(self, advisories, g, context, scores):
        items=[]
        for issue in advisories:
            if issue=='PERSPECTIVE_UNVERIFIED':
                items.append(dict(code='PERSPECTIVE_UNVERIFIED',source_issue=issue,capture_context=context,severity=None))
                continue
            item=self._guidance(issue,g,context)
            item['severity']=scores.get(issue)
            items.append(item)
        return items

    def _workflow_state(self, mode, found, issue):
        if mode=='final': return 'ACCEPT' if issue is None else 'RETAKE'
        if not found or issue=='PASSPORT_NOT_FOUND': return 'SEARCHING'
        if issue in (None,'READY'): return 'READY'
        if issue=='HOLD_STEADY': return 'HOLD_STEADY'
        if issue=='CHECKING_STABILITY': return 'QUALITY_CHECK'
        if issue in _GEOMETRY_ISSUES: return 'ALIGNING'
        return 'QUALITY_CHECK'

    def analyze_frame(self, frame, guide_box=None, mode='preview', *, timestamp=None, detection=None, capture_context=None):
        if mode not in ('preview','final'): raise ValueError('mode must be preview or final')
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<8:
            raise ValueError('frame must be nonempty uint8 HxWx3 BGR')
        context=self._context(mode,capture_context)
        if guide_box is None:
            if context!='document_crop': raise ValueError('guide_box is required except for document_crop')
            guide_box=dict(x=0.,y=0.,w=1.,h=1.)
        now=monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(now): raise ValueError('timestamp must be finite')
        if mode=='preview' and self.temporal.last is not None and now<=self.temporal.last:
            raise ValueError('Preview timestamps must strictly increase')
        start=perf_counter(); times={}; self.debug={}
        guide=guide_polygon(guide_box,frame.shape)
        t=perf_counter(); det=detection if detection is not None else self.localizer.locate(frame,mode); times['localization']=(perf_counter()-t)*1000
        g={}; q={}; raw={}; limitations=[]; evidence={}
        if det.found:
            det.polygon=order_quad(det.polygon)
            t=perf_counter(); g=analyze_geometry(det,guide,frame.shape,self.config['geometry']); times['geometry']=(perf_counter()-t)*1000
            t=perf_counter(); crop,valid,mrz,matrix=rectify(frame,det,self.config['normalization'][mode+'_width']); times['normalization']=(perf_counter()-t)*1000
            t=perf_counter(); e=exposure(crop,valid,self.config['exposure']); times['exposure']=(perf_counter()-t)*1000
            t=perf_counter(); b,heat=blur(crop,valid,mrz,self.config['blur'],self.config['normalization'],mode); times['blur']=(perf_counter()-t)*1000
            t=perf_counter()
            if mode=='final' or self.config['glare']['enable_preview']:
                gl,mask=glare(crop,valid,mrz,self.config['glare'],mode)
            else:
                gl={'glare_score':None}; mask=np.zeros(crop.shape[:2],np.uint8)
            times['glare']=(perf_counter()-t)*1000
            t=perf_counter(); rd=analyze_readability(frame,det,self.config['readability']) if self.config['readability']['enabled'] else None; times['readability']=(perf_counter()-t)*1000
            score=b['blur_score']
            if rd is not None and rd['blur_score'] is not None:
                score=max(score,rd['blur_score']) if score is not None else rd['blur_score']
            confirmed_glare=_confirmed_glare_score(gl,rd,self.config['glare'])
            gl['candidate_score']=gl.get('glare_score')
            gl['decision_glare_score']=confirmed_glare
            q={k:e[k] for k in ('too_dark_score','too_bright_score')}
            q.update(blur_score=score,glare_score=confirmed_glare,
                low_resolution_score=rd['low_resolution_score'] if rd else None,
                low_contrast_score=rd['low_contrast_score'] if rd else None,
                noise_score=rd['noise_score'] if rd else None)
            m=self.motion.update(det,frame.shape,now) if mode=='preview' else None
            q['motion_score']=m['score'] if m else 0.
            raw=dict(exposure=e,blur=b,glare=gl,readability=rd,motion=m)
            policy=self.config['policy'][context]
            page_confident=det.confidence>=policy['min_page_confidence']
            mrz_present=det.mrz_polygon is not None and det.mrz_confidence>=policy['min_mrz_confidence']
            if page_confident and mrz_present and det.corners_reliable: grade='A'
            elif page_confident and mrz_present: grade='B'
            elif page_confident: grade='C'
            else: grade='D'
            evidence=dict(
                mrz_present=mrz_present,corners_verified=bool(det.corners_reliable),page_confident=page_confident,
                text_detail_observed=bool(rd and rd['evidence_sufficient']),blur_observed=score is not None,
                glare_observed=gl['glare_score'] is not None,localization_grade=grade,
                perspective_verified=bool(det.corners_reliable),corner_evidence=det.corner_evidence or {},
            )
            evidence['document_completeness_score']=self.engine.document_incomplete_score(g,evidence,context)
            evidence['document_completeness_ok']=not self.engine.active('DOCUMENT_INCOMPLETE',evidence['document_completeness_score'])
            if not self.config['blur']['calibrated']: limitations.append('QUALITY_THRESHOLDS_NOT_OCR_CALIBRATED')
            if not mrz_present: limitations.append('MRZ_NOT_RELIABLY_DETECTED')
            if not det.corners_reliable: limitations.append('PERSPECTIVE_UNVERIFIED')
            if not evidence['text_detail_observed']: limitations.append('INSUFFICIENT_TEXT_DETAIL_EVIDENCE')
            if det.source!='yolo': limitations.append('SYNTHETIC_OR_SUPPLIED_LOCALIZATION_NOT_IDENTITY_VERIFICATION')
            self.debug=dict(crop=crop,valid=valid,mrz=mrz,glare_mask=mask,blur_heatmap=heat,homography=matrix)
        else:
            self.motion.reset(); e={}

        t=perf_counter(); scores=self.engine.scores(det.found,g,q,evidence,context)
        blockers_policy=self.engine.blocking_codes(context)
        raw_issue=self.engine.primary(scores,allowed=blockers_policy)
        active_blockers=[c for c in blockers_policy if self.engine.active(c,scores.get(c))]
        raw_state=('RETAKE' if raw_issue else 'ACCEPT') if mode=='final' else ('ADJUST' if raw_issue else 'READY')
        raw_decision=dict(state=raw_state,primary_issue=raw_issue or ('READY' if mode=='preview' else None))
        state=raw_state; issue=raw_issue; stable=None
        if mode=='preview':
            stable=self.temporal.update(scores,now,g,blocking_codes=blockers_policy)

            # Ordinary HOLD_STEADY must persist before it revokes capture.  Only
            # extremely strong instantaneous motion is treated as an immediate
            # blocker.  Other defects still block immediately as before.
            non_motion_codes=[c for c in blockers_policy if c!='HOLD_STEADY']
            raw_non_motion=self.engine.primary(scores,allowed=non_motion_codes)
            hold_value=float(scores.get('HOLD_STEADY') or 0.0)
            immediate_hold=hold_value>=float(self.config['motion'].get('immediate_block_score',0.92))
            clean_long_enough=stable.get('clean_duration_s',0)>=self.config['temporal']['ready_s']-1e-9
            ready=(stable['state']=='READY' and raw_non_motion is None and not immediate_hold and clean_long_enough)
            state='READY' if ready else 'ADJUST'
            if ready:
                issue='READY'
            elif immediate_hold:
                issue='HOLD_STEADY'
            elif stable['primary_issue']=='HOLD_STEADY' and self.engine.active('HOLD_STEADY',scores.get('HOLD_STEADY')):
                issue='HOLD_STEADY'
            elif stable['primary_issue'] not in (None,'READY') and self.engine.active(stable['primary_issue'],scores.get(stable['primary_issue'])):
                issue=stable['primary_issue']
            elif raw_non_motion is not None:
                issue=raw_non_motion
            else:
                # Waiting for debounce is not the same as accusing the user of
                # shaking the document.  Show a neutral stability message.
                issue='CHECKING_STABILITY'
        times['decision_temporal']=(perf_counter()-t)*1000; times['total']=(perf_counter()-start)*1000

        advisories=self._advisories(evidence,g,scores,context)
        advisory_guidance=self._advisory_guidance(advisories,g,context,scores)
        guidance=self._guidance(issue,g,context)
        if guidance['severity'] is None and issue not in (None,'READY'):
            guidance['severity']=scores.get(issue)
        guide_codes=set(self.config['geometry'].get('guide_advisory_codes',[]))
        active_guide_advisories=[c for c in advisories if c in guide_codes and self.engine.active(c,scores.get(c))]
        capture_quality_state=('NOT_READY' if state!='READY' else
            ('READY' if advisories else 'OPTIMAL'))
        recommended_adjustment=None
        for item in advisory_guidance:
            if item.get('source_issue') in guide_codes:
                recommended_adjustment=item
                break
        workflow=self._workflow_state(mode,det.found,issue if issue!='READY' else None)
        all_issues=[]
        blocking_set=set(blockers_policy)
        for c,v in scores.items():
            all_issues.append(dict(code=c,score=v,active=self.engine.active(c,v),blocking=c in blocking_set,
                                   enter_threshold=self.engine.enter_threshold(c)))
        return dict(
            schema_version='0.4.3-fp2.1',mode=mode,capture_context=context,state=state,workflow_state=workflow,
            primary_issue=issue,passport_found=det.found,frame_size=dict(width=frame.shape[1],height=frame.shape[0]),timestamp_s=now,
            guidance_code=guidance['code'],guidance=guidance,requires_final_check=mode=='preview',confidence=det.confidence,
            localization=det.as_dict(),guide_polygon=guide.tolist(),geometry=g,quality=q,raw_metrics=raw,all_issues=all_issues,
            raw_decision=raw_decision,temporal=stable,timing_ms=times,limitations=limitations,
            ready_for_capture=state=='READY',capture_allowed=state=='READY',blocking_issues=active_blockers,
            capture_quality_state=capture_quality_state,guide_alignment_ideal=not bool(active_guide_advisories),
            advisories=advisories,advisory_guidance=advisory_guidance,recommended_adjustment=recommended_adjustment,
            quality_evidence=evidence,thresholds_status='provisional',production_validated=False,
            acceptance_scope='v4_3_fp2_1_targeted_policy_not_authenticity_or_ocr_guarantee'
        )
