"""Regressions for real manual failures described by corrective handoff V3.

Deterministic generated proxies; no personal passport images or OCR fixtures.
"""
from copy import deepcopy
import cv2
import numpy as np
import pytest

from passport_quality_gate.analyzer import Analyzer, _confirmed_glare_score
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.config import load_config
from passport_quality_gate.geometry import box_quad
from passport_quality_gate.localization import Detection, ProxyLocalizer
from passport_quality_gate.research import ResearchSession, content_at_cut, guidance_text
from passport_quality_gate.synthetic import sample, GUIDE


def cfg(): return load_config('configs/research_vnext.yaml')


def ready_analyzer():
    image, det, _=sample()
    analyzer=Analyzer(ProxyLocalizer(),cfg())
    for i in range(10):
        r=analyzer.analyze_frame(image,GUIDE,detection=deepcopy(det),timestamp=i*.2)
    assert r['capture_allowed']
    return analyzer,image,det


def test_isolated_mrz_miss_holds_recent_yolo_but_sustained_absence_blocks():
    analyzer,image,det=ready_analyzer()
    absent=deepcopy(det); absent.mrz_polygon=None; absent.mrz_confidence=0.
    r=analyzer.analyze_frame(image,GUIDE,detection=absent,timestamp=2.)
    assert r['capture_allowed'], r['blocking_issues']
    assert r['research']['mrz']['held']
    assert all(v==0 for v in r['research']['expected_outside'].values())
    r=analyzer.analyze_frame(image,GUIDE,detection=absent,timestamp=2.3)
    assert not r['capture_allowed']
    assert 'MRZ_NOT_FOUND' in r['blocking_issues']
    assert r['guidance_code']=='SHOW_BOTTOM_TEXT'
    # Strong classical fallback is still visible, but has no acceptance power.
    assert r['research']['mrz']['fallback']['score']>=.75
    for t in [2.5,3.,4.]:
        r=analyzer.analyze_frame(image,GUIDE,detection=absent,timestamp=t)
        assert not r['capture_allowed']


def test_final_never_uses_preview_mrz_cache_or_fallback():
    analyzer,image,det=ready_analyzer()
    absent=deepcopy(det); absent.mrz_polygon=None; absent.mrz_confidence=0.
    r=analyzer.analyze_frame(image,GUIDE,'final',detection=absent,timestamp=1.9)
    assert r['state']=='RETAKE'
    assert 'MRZ_NOT_FOUND' in r['blocking_issues']
    assert not r['research']['mrz']['credible']
    assert r['localization']['mrz_polygon'] is None


def test_low_confidence_cannot_borrow_fallback_and_final_threshold_is_separate():
    image,det,_=sample(); config=cfg()
    config['policy']['live_preview']['min_mrz_confidence']=.4
    config['policy']['full_frame_final']['min_mrz_confidence']=.45
    a=Analyzer(ProxyLocalizer(),config)
    det.mrz_confidence=.42
    for i in range(10): r=a.analyze_frame(image,GUIDE,detection=det,timestamp=i*.2)
    assert r['capture_allowed']
    r=a.analyze_frame(image,GUIDE,'final',detection=det,timestamp=2.)
    assert r['state']=='RETAKE'
    det.mrz_confidence=.3
    a.reset()
    r=a.analyze_frame(image,GUIDE,detection=det,timestamp=3.)
    assert not r['research']['mrz']['credible']


def test_mrz_hold_cannot_cross_session_or_large_geometry_jump():
    a,image,det=ready_analyzer(); absent=deepcopy(det)
    absent.mrz_polygon=None; absent.mrz_confidence=0.; absent.polygon[:,0]+=200
    r=a.analyze_frame(image,GUIDE,detection=absent,timestamp=1.9)
    assert not r['research']['mrz']['held']
    a.reset(); absent.polygon=det.polygon.copy()
    r=a.analyze_frame(image,GUIDE,detection=absent,timestamp=2.)
    assert not r['research']['mrz']['credible']


@pytest.mark.parametrize('loss', ['MRZ_NOT_FOUND','LOCALIZATION_UNCERTAIN','QUALITY_UNCERTAIN','PASSPORT_NOT_FOUND','BLUR'])
def test_recent_strong_motion_owns_causal_guidance(loss):
    s=ResearchSession(cfg()['research'])
    assert s.motion_evidence(1.,0.,'preview')
    recent=s.motion_evidence(0.,.2,'preview')
    r=s.guidance_for({'code':loss},[loss],{loss:1.},{'recent_strong_motion':recent},.2,'preview')
    assert r['code']=='HOLD_STEADY'
    assert not s.motion_evidence(0.,1.,'preview')


def test_independent_cut_or_glare_preempts_motion_attribution():
    s=ResearchSession(cfg()['research'])
    r=s.guidance_for({},['DOCUMENT_INCOMPLETE','QUALITY_UNCERTAIN'],{'DOCUMENT_INCOMPLETE':1.,'QUALITY_UNCERTAIN':1.},
                     {'recent_strong_motion':True,'cut_scores':{'left':.9}},0.,'preview')
    assert r['code']=='MOVE_RIGHT'
    r=s.guidance_for({},['MRZ_GLARE','BLUR'],{'MRZ_GLARE':.9,'BLUR':1.},{'recent_strong_motion':True},.1,'preview')
    assert r['code']=='TILT_TO_REMOVE_BOTTOM_REFLECTION'


def test_addendum_page_localization_is_prerequisite_even_with_recent_motion():
    a,image,det=ready_analyzer()
    a.motion.update=lambda *args: {'score':1.}
    a.analyze_frame(image,GUIDE,detection=deepcopy(det),timestamp=2.)
    r=a.analyze_frame(image,GUIDE,detection=Detection(),timestamp=2.1)
    assert not r['capture_allowed']
    assert r['guidance_code']=='PLACE_PASSPORT_IN_FRAME'


@pytest.mark.parametrize('cause,action', [('LOW_RESOLUTION','MOVE_CLOSER'),('QUALITY_UNCERTAIN','HOLD_STEADY'),
    ('MRZ_NOT_FOUND','SHOW_BOTTOM_TEXT'),('CHECKING_STABILITY','HOLD_STEADY')])
def test_internal_codes_never_leak_as_primary_guidance(cause,action):
    s=ResearchSession(cfg()['research'])
    r=s.guidance_for({'code':cause},[] if cause=='CHECKING_STABILITY' else [cause],{cause:1.},{},0.,'preview')
    assert r['code']==action
    assert 'MRZ' not in guidance_text(r['code'])


def test_far_frame_end_to_end_move_closer():
    image,det,_=sample('scale',.35)
    r=Analyzer(ProxyLocalizer(),cfg()).analyze_frame(image,GUIDE,detection=det,timestamp=0.)
    assert not r['capture_allowed']
    assert r['guidance_code']=='MOVE_CLOSER', r['guidance_code']


@pytest.mark.parametrize('side', ['left','right','top','bottom'])
def test_small_content_cut_blocks_without_global_shape_change(side):
    image=np.full((450,700,3),235,np.uint8)
    p=box_quad([0,0,699,449]); det=Detection(p,None,1.,'test',False,0.)
    # Several compact strokes cross one actual camera cut boundary. The same
    # page shape is shared by positive and blank controls, not a cropped-box prior.
    for offset in (110,160,210):
        if side=='left': cv2.rectangle(image,(0,offset),(12,offset+8),(30,30,30),-1)
        elif side=='right': cv2.rectangle(image,(687,offset),(699,offset+8),(30,30,30),-1)
        elif side=='top': cv2.rectangle(image,(offset,0),(offset+8,12),(30,30,30),-1)
        else: cv2.rectangle(image,(offset,437),(offset+8,449),(30,30,30),-1)
    scores=content_at_cut(image,det,cfg()['research'])
    assert scores[side]['score']>=.55, scores
    assert all(v['score']==0. for k,v in scores.items() if k!=side)
    g={'crop_risk':0.,'page_aspect_ratio':1.55,'frame_edge_gaps':dict(left=0.,right=0.,top=0.,bottom=0.)}
    evidence=ResearchSession(cfg()['research']).observe_geometry(image,det,g,{'credible':False,'current_yolo_present':False})
    assert evidence['cut_scores'][side]>=.55


@pytest.mark.parametrize('margin', [0,1,3])
def test_blank_margin_or_uniform_page_line_not_content_cut(margin):
    image=np.full((450,700,3),235,np.uint8)
    cv2.line(image,(margin,0),(margin,449),(40,40,40),1)
    det=Detection(box_quad([margin,0,699,449]),None,1.)
    scores=content_at_cut(image,det,cfg()['research'])
    assert scores['left']['score']==0.


def test_spatial_mrz_candidate_requires_independent_damage():
    candidate={'glare_score':.9,'mrz_overlap':.4,'clipping_inside_glare':.9,'regions':[{
        'mrz_overlap_ratio':.2,'mrz_span_w':.4,'mrz_span_h':.8,'mean_context_delta':25.,'clipping_ratio':1.}]}
    assert _confirmed_glare_score(candidate,{},cfg()['glare'],localized=True)==0.
    candidate['mrz_damage_score']=.1
    assert _confirmed_glare_score(candidate,{},cfg()['glare'],localized=True)<.6
    candidate['mrz_damage_score']=.9
    assert _confirmed_glare_score(candidate,{},cfg()['glare'],localized=True)>=.6


@pytest.mark.parametrize('color', [(255,255,255),(215,255,255),(245,245,245)])
def test_monitor_bright_security_shine_with_preserved_strokes_is_negative(color):
    image,det,_=sample(); x1,y1=det.mrz_polygon.min(0).astype(int); x2,y2=det.mrz_polygon.max(0).astype(int)
    roi=image[y1:y2,x1:x2]; ink=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)<150
    roi[~ink]=color
    s=ResearchSession(cfg()['research']); result=s.mrz_glare(image,det,cfg()['glare'],0.,'final')
    assert result['score']<.6, result['worst_cell']
    final=Analyzer(ProxyLocalizer(),cfg()).analyze_frame(image,GUIDE,'final',detection=det,timestamp=0.)
    assert 'MRZ_GLARE' not in final['blocking_issues'], final['quality']


def test_destructive_local_patch_still_blocks_final_with_plain_instruction():
    image,det,_=sample(); x,y=det.mrz_polygon.min(0).astype(int); ymax=int(det.mrz_polygon[:,1].max())
    cv2.rectangle(image,(x+220,y+8),(x+255,ymax-4),(255,255,255),-1)
    r=Analyzer(ProxyLocalizer(),cfg()).analyze_frame(image,GUIDE,'final',detection=det,timestamp=0.)
    assert r['state']=='RETAKE'
    assert 'MRZ_GLARE' in r['blocking_issues']
    assert 'MRZ' not in r['guidance_text']


def test_physical_bottom_loss_cannot_be_rescued_by_fake_classical_rows(monkeypatch):
    import passport_quality_gate.research as research
    image,det,_=sample()
    image=image[:490].copy()  # Original bottom lines start below this boundary.
    det.polygon[:,1]=np.minimum(det.polygon[:,1],489)
    det.mrz_polygon=None; det.mrz_confidence=0.; det.corners_reliable=False
    fake={'score':1.,'polygon':box_quad([210,420,810,480]).tolist(),'rows':2}
    monkeypatch.setattr(research,'fallback_mrz',lambda *args:deepcopy(fake))
    a=Analyzer(ProxyLocalizer(),cfg())
    for t in [0.,.2,.4,.6,1.]:
        r=a.analyze_frame(image,GUIDE,detection=det,timestamp=t)
        assert not r['capture_allowed']
        assert 'MRZ_NOT_FOUND' in r['blocking_issues']
        assert r['localization']['mrz_polygon'] is None
    assert a.analyze_frame(image,GUIDE,'final',detection=det)['state']=='RETAKE'


def test_guide_is_not_a_crop_boundary_or_only_detection_roi():
    image,det,_=sample(); a=Analyzer(ProxyLocalizer(),cfg())
    narrow_guide=dict(x=.4,y=.4,w=.2,h=.2)
    for i in range(10): r=a.analyze_frame(image,narrow_guide,detection=deepcopy(det),timestamp=i*.2)
    assert r['capture_allowed'],r['blocking_issues']
    assert r['research']['mrz']['current_yolo_present']


def test_public_research_result_exposes_plain_text_without_changing_default_contract():
    from passport_quality_gate.api import to_public_result
    a,image,det=ready_analyzer()
    result=a.analyze_frame(image,GUIDE,detection=det,timestamp=2.)
    assert to_public_result(result)['guidance_text']=='Ready to capture'
    baseline=Analyzer(ProxyLocalizer()).analyze_frame(image,GUIDE,detection=det,timestamp=0.)
    assert 'guidance_text' not in to_public_result(baseline)


@pytest.mark.parametrize('confidence', [0., .05])
@pytest.mark.parametrize('mode', ['preview','final'])
def test_stage_one_hides_all_quality_and_advisory_guidance(confidence,mode):
    image,det,_=sample('scale',.35)
    det.confidence=confidence
    a=Analyzer(ProxyLocalizer(),cfg())
    r=a.analyze_frame(image,GUIDE,mode,detection=det,timestamp=0.)
    assert r['guidance_code']=='PLACE_PASSPORT_IN_FRAME'
    assert not r['advisory_guidance']
    assert r['recommended_adjustment'] is None
    assert not r['capture_allowed']


def test_held_presence_never_authorizes_new_mrz_quality_diagnoses():
    a,image,det=ready_analyzer()
    det.mrz_polygon=None; det.mrz_confidence=0.
    r=a.analyze_frame(image,GUIDE,detection=det,timestamp=2.)
    assert r['research']['mrz']['held']
    assert r['localization']['mrz_polygon'] is None
    assert not r['research']['mrz_glare']['observed']
    assert r['quality_evidence']['text_detail_held']
    assert not r['quality_evidence']['text_detail_current_observed']
    s=ResearchSession(cfg()['research'])
    out=s.guidance_for({'code':'MRZ_GLARE'},['MRZ_GLARE'],{'MRZ_GLARE':1.},
        {'page_reliable':True,'mrz':{'current_yolo_present':False,'held':True}},0.,'preview')
    assert out['code']=='HOLD_STEADY'


def test_page_present_motion_plus_mrz_loss_gives_hold_steady():
    a,image,det=ready_analyzer()
    a.motion.update=lambda *args: {'score':1.}
    det.mrz_polygon=None; det.mrz_confidence=0.
    r=a.analyze_frame(image,GUIDE,detection=det,timestamp=2.5)
    assert not r['capture_allowed']
    assert r['guidance_code']=='HOLD_STEADY'


def test_unreliable_page_cannot_seed_mrz_debounce():
    image,det,_=sample(); a=Analyzer(ProxyLocalizer(),cfg())
    low=deepcopy(det); low.confidence=.01
    a.analyze_frame(image,GUIDE,detection=low,timestamp=0.)
    det.mrz_polygon=None; det.mrz_confidence=0.
    r=a.analyze_frame(image,GUIDE,detection=det,timestamp=.1)
    assert not r['research']['mrz']['held']
    assert r['guidance_code']=='SHOW_BOTTOM_TEXT'
