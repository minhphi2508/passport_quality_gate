from copy import deepcopy

import cv2
import numpy as np
import pytest

from passport_quality_gate.analyzer import Analyzer, _confirmed_glare_score
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.config import load_config
from passport_quality_gate.geometry import box_quad
from passport_quality_gate.localization import Detection, ProxyLocalizer
from passport_quality_gate.research import ResearchSession, side_completeness, fallback_mrz, prepare_detection
from passport_quality_gate.synthetic import GUIDE, sample


def config():
    return load_config('configs/research_vnext.yaml')


def settle(image, detection, cfg=None):
    a=Analyzer(ProxyLocalizer(), cfg or config())
    for i in range(9):
        result=a.analyze_frame(image,GUIDE,timestamp=i*.2,detection=detection)
    return result


def boundaries(strength=.2):
    return {s:dict(strength=strength) for s in ('left','right','top','bottom')}


def geometry(**kwargs):
    return dict(page_aspect_ratio=1.42, frame_edge_gaps=dict(left=.1,right=.1,top=.1,bottom=.1), **kwargs)


def test_single_cue_never_blocks_and_uniform_weak_edges_remain_uncertain():
    g=geometry(); g['frame_edge_gaps']['left']=0.
    out=side_completeness(g,boundaries(),{},1.,{},config()['research'])
    assert max(x['cut_score'] for x in out.values())==0
    assert out['left']['state']=='UNCERTAIN'


@pytest.mark.parametrize('side', ['left','right','top','bottom'])
def test_corroborated_cut_keeps_side_and_direction(side):
    cfg=config()['research']; session=ResearchSession(cfg)
    g=geometry(); g['frame_edge_gaps'][side]=0.
    b=boundaries(.9); b[side]['strength']=0.
    cuts=side_completeness(g,b,{},1.,{},cfg)
    assert cuts[side]['state']=='MISSING'
    assert all(v['cut_score']==0 for s,v in cuts.items() if s!=side)
    expected=dict(left='MOVE_RIGHT',right='MOVE_LEFT',top='MOVE_DOWN',bottom='MOVE_UP')
    result=session.guidance_for({},['DOCUMENT_INCOMPLETE'],{'DOCUMENT_INCOMPLETE':1.},
                                {'cut_scores':{s:v['cut_score'] for s,v in cuts.items()}},0.,'preview')
    assert result['code']==expected[side]


def test_top_blank_trim_allowed_but_critical_or_substantial_cut_blocks():
    cfg=config()['research']; g=geometry(); g['frame_edge_gaps']['top']=0.
    b=boundaries(.9); b['top']['strength']=0.
    blank=side_completeness(g,b,{},0.,{},cfg)
    assert blank['top']['cut_score']==0
    assert side_completeness(g,b,{},1.,{},cfg)['top']['cut_score']>=.55
    assert side_completeness(g,b,{},0.,{'top_fraction':.2},cfg)['top']['cut_score']>=.55


def test_complete_weak_edges_and_close_margin_can_ready():
    image,det,_=sample()
    det.corners_reliable=False
    det.corner_evidence={'bbox_side_support':[.2]*4,'bbox_side_max_gap':[.7]*4}
    r=settle(image,det)
    assert r['capture_allowed'], r['blocking_issues']
    # Translate the complete page to a 2px positive camera margin.
    shift=int(det.polygon[:,0].min())-2
    image=cv2.warpAffine(image,np.float32([[1,0,-shift],[0,1,0]]),(image.shape[1],image.shape[0]))
    det.polygon[:,0]-=shift; det.mrz_polygon[:,0]-=shift
    r=settle(image,det)
    assert r['capture_allowed'], r['blocking_issues']
    assert r['guidance_code']!='SHOW_ALL_EDGES'
    assert r['geometry']['legacy_crop_risk']>r['geometry']['crop_risk']


def test_classical_fallback_two_rows_are_telemetry_only_v3():
    image,det,_=sample(); det.mrz_polygon=None; det.mrz_confidence=0.
    fallback=fallback_mrz(image,det,config()['research'])
    assert fallback['score']>=.75, fallback
    updated,state=prepare_detection(image,det,config()['research'],.4)
    assert det.mrz_polygon is None  # caller-owned input is not mutated
    # V3 explicitly revokes the V2 fallback acceptance authority.
    assert state['state']=='ABSENT' and not state['credible']
    assert updated.mrz_polygon is None
    r=settle(image,det)
    assert 'MRZ_NOT_FOUND' in r['blocking_issues']
    assert not r['capture_allowed']
    blank=np.full_like(image,220)
    assert fallback_mrz(blank,det,config()['research'])['score']==0.


def test_localized_component_is_not_clamped_by_global_raw_glare():
    metric={'glare_score':.08,'mrz_overlap':.01,'regions':[{
        'mrz_overlap_ratio':.06,'mrz_span_w':.18,'mrz_span_h':.65,
        'mean_context_delta':25.,'clipping_ratio':.8}]}
    cfg=config()['glare']
    assert _confirmed_glare_score(metric,{},cfg)<=.08
    # V3: spatial candidate alone is insufficient, even with a bright blob.
    assert _confirmed_glare_score(metric,{},cfg,localized=True)==0.
    metric['mrz_damage_score']=1.
    assert _confirmed_glare_score(metric,{},cfg,localized=True)>=.6
    metric['regions']=[]
    assert _confirmed_glare_score(metric,{},cfg,localized=True)<.1


def test_mrz_glare_clean_bright_and_local_damage():
    cfg=config(); s=ResearchSession(cfg['research'])
    image,det,_=sample()
    clean=s.mrz_glare(image,det,cfg['glare'],0.,'preview')
    assert clean['score']<.6
    damaged=image.copy()
    x1,y1=det.mrz_polygon.min(0).astype(int); x2,y2=det.mrz_polygon.max(0).astype(int)
    cv2.rectangle(damaged,(x1+220,y1+8),(x1+255,y2-4),(255,255,255),-1)
    result=s.mrz_glare(damaged,det,cfg['glare'],.1,'preview')
    assert result['score']>=.6, result['worst_cell']
    assert len(s.crops)<=cfg['research']['temporal_crops']
    s.reset(); assert not s.crops
    bright=np.full_like(image,250)
    assert s.mrz_glare(bright,det,cfg['glare'],0.,'final')['score']==0.


def test_guidance_recovery_critical_preemption_and_hysteresis():
    session=ResearchSession(config()['research'])
    active=['BLUR','LOW_CONTRAST']; scores=dict(BLUR=.8,LOW_CONTRAST=.7)
    first=session.guidance_for({},active,scores,{},0.,'preview')
    assert first['code']=='WAIT_FOR_FOCUS'
    scores['LOW_CONTRAST']=.85
    assert session.guidance_for({},active,scores,{},.1,'preview')['code']=='WAIT_FOR_FOCUS'
    active.append('MRZ_GLARE'); scores['MRZ_GLARE']=.8
    assert session.guidance_for({},active,scores,{},.2,'preview')['code']=='TILT_TO_REMOVE_BOTTOM_REFLECTION'
    assert session.guidance_for({'code':'READY'},[],{}, {},.3,'preview')['code']=='READY'


class CountingLocalizer:
    def __init__(self, det): self.det=det; self.calls=0
    def locate(self, frame, mode='preview'):
        self.calls+=1
        return deepcopy(self.det)


def test_one_detector_pass_final_exact_pixels_and_public_metadata():
    image,det,_=sample(); localizer=CountingLocalizer(det)
    gate=PassportQualityGate(config=config(),localizer=localizer,device='cpu')
    r=gate.analyze_preview(image,GUIDE,timestamp=0.)
    assert localizer.calls==1
    final=gate.analyze_final(image,GUIDE,timestamp=1.)
    assert localizer.calls==2
    assert final['capture_allowed']==(final['state']=='ACCEPT')
    assert gate.runtime_info()['quality_policy']=='VNEXT-CORRECTIVE-V3'
    assert r['research']['mrz']['state']=='STRONG'


@pytest.mark.parametrize('kind,level,allowed', [('clean',0,True),('scale',.65,True),('scale',.35,False)])
def test_existing_good_scenarios(kind,level,allowed):
    image,det,_=sample(kind,level)
    assert settle(image,det)['capture_allowed']==allowed


@pytest.mark.parametrize('side,shift,action', [
    ('left',(-280,0),'MOVE_RIGHT'), ('right',(300,0),'MOVE_LEFT'),
    ('top',(0,-220),'MOVE_DOWN'), ('bottom',(0,250),'MOVE_UP')])
def test_image_level_clipped_detector_boxes(side,shift,action):
    image,det,_=sample(); dx,dy=shift
    image=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(1024,768))
    for polygon in (det.polygon,det.mrz_polygon):
        polygon += [dx,dy]
        polygon[:,0]=np.clip(polygon[:,0],0,1023)
        polygon[:,1]=np.clip(polygon[:,1],0,767)
    det.corners_reliable=False
    result=settle(image,det)
    assert not result['capture_allowed']
    assert result['research']['cut_scores'][side]>=.55
    assert result['guidance_code']==action
    if side=='bottom':
        # A visible partial YOLO MRZ does not override independent bottom crop.
        assert result['research']['cut_scores']['bottom']>=.55


def test_multi_side_guidance_and_localization_only_neutral():
    s=ResearchSession(config()['research'])
    r=s.guidance_for({},['CROPPED'],{'CROPPED':1.}, {'cut_scores':{'left':1.,'bottom':1.}},0.,'preview')
    assert r['code']=='SHOW_ALL_EDGES'
    r=s.guidance_for({},['LOCALIZATION_UNCERTAIN'],{'LOCALIZATION_UNCERTAIN':1.},{},.1,'preview')
    assert r['code']=='CENTER_AND_HOLD'


def test_jitter_preserves_ready_and_strong_motion_is_not_graced():
    image,det,_=sample(); a=Analyzer(ProxyLocalizer(),config())
    for i in range(10):
        a.analyze_frame(image,GUIDE,detection=deepcopy(det),timestamp=i*.2)
    for i,dx in enumerate([1,-1,2,-2,1,0]):
        moved=deepcopy(det); moved.polygon[:,0]+=dx; moved.mrz_polygon[:,0]+=dx
        shifted=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,0]]),(1024,768))
        r=a.analyze_frame(shifted,GUIDE,detection=moved,timestamp=2.+i*.2)
        assert r['capture_allowed'],r['blocking_issues']
    # Inject measured severe motion to isolate policy from detector artifacts.
    a.motion.update=lambda *args: {'score':1., 'raw_speed':1., 'filtered_speed':1.}
    r=a.analyze_frame(image,GUIDE,detection=deepcopy(det),timestamp=3.3)
    assert not r['capture_allowed']
    assert r['primary_issue']=='HOLD_STEADY'


def test_recovery_and_no_cross_document_temporal_glare():
    cfg=config(); s=ResearchSession(cfg['research']); image,det,_=sample()
    s.mrz_glare(image,det,cfg['glare'],0.,'preview')
    assert len(s.crops)==1
    s.mrz_glare(image,det,cfg['glare'],2.,'preview')
    assert len(s.crops)==1
    a=Analyzer(ProxyLocalizer(),cfg)
    a.analyze_frame(image,GUIDE,detection=Detection(),timestamp=0.)
    for i in range(1,16):
        result=a.analyze_frame(image,GUIDE,detection=deepcopy(det),timestamp=i*.2)
    assert result['capture_allowed']
    assert result['guidance_code']=='READY'


def test_no_yolo_mrz_blocks_even_without_cut_evidence_v3():
    image,det,_=sample(); image[:]=220; det.mrz_polygon=None; det.mrz_confidence=0.
    result=settle(image,det)
    assert not result['capture_allowed']
    assert result['research']['mrz']['state']=='ABSENT'
    assert 'MRZ_NOT_FOUND' in result['blocking_issues']
    assert 'QUALITY_UNCERTAIN' in result['blocking_issues']


def test_lighting_guidance_precedes_secondary_blur_and_motion_blur_is_specific():
    s=ResearchSession(config()['research'])
    r=s.guidance_for({},['BLUR','TOO_DARK'],dict(BLUR=1.,TOO_DARK=.7),{},0.,'preview')
    assert r['code']=='INCREASE_LIGHT'
    r=s.guidance_for({},['BLUR'],dict(BLUR=.8),{'motion_score':.5},.1,'preview')
    assert r['code']=='HOLD_STEADY'


def test_async_harness_uses_selected_pixels_and_never_saves_images_by_default(monkeypatch,tmp_path):
    import importlib.util
    import sys
    from concurrent.futures import Future
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('research_harness','examples/webcam_research_vnext.py')
    harness=importlib.util.module_from_spec(spec); spec.loader.exec_module(harness)
    seen=[]
    class Gate:
        def __init__(self,**kwargs): pass
        def runtime_info(self): return {}
        def reset(self): pass
        def analyze_preview(self,frame,guide,timestamp):
            return {'capture_allowed':True,'guidance_code':'READY','state':'READY','guide_polygon':[[0,0],[5,0],[5,5],[0,5]],
                    'blocking_issues':[],'raw_metrics':{},'quality':{},'timing_ms':{'total':1.}}
        def analyze_final(self,frame,guide,timestamp):
            seen.append(int(frame[0,0,0])); return {'capture_allowed':True}
    class Camera:
        def __init__(self,*args): self.index=0
        def set(self,*args): pass
        def isOpened(self): return True
        def release(self): pass
        def read(self):
            self.index+=1
            return True,np.full((16,16,3),self.index,np.uint8)
    class Executor:
        def __init__(self,**kwargs): pass
        def submit(self,fn,*args,**kwargs):
            future=Future(); future.set_result(fn(*args,**kwargs)); return future
        def shutdown(self,**kwargs): pass
    class Selector:
        def __init__(self): self.frames=[]
        def push(self,frame,result,timestamp): self.frames.append(frame.copy())
        def clear(self): self.frames=[]
        def select_recent(self,trigger_timestamp):
            return SimpleNamespace(frame=self.frames[0],metadata=lambda:{'selected':True})
    clock=iter(np.arange(0,20,.2)); keys=iter([-1,-1,ord('c'),ord('q')])
    monkeypatch.setattr(harness,'monotonic',lambda:float(next(clock)))
    monkeypatch.setattr(harness,'ThreadPoolExecutor',Executor)
    monkeypatch.setattr(harness,'PassportQualityGate',Gate)
    monkeypatch.setattr(harness,'BestFrameSelector',Selector)
    monkeypatch.setattr(harness.cv2,'VideoCapture',Camera)
    monkeypatch.setattr(harness.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(harness.cv2,'waitKey',lambda *args:next(keys))
    monkeypatch.setattr(harness.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(harness.cv2,'imwrite',lambda *args:pytest.fail('Unexpected image persistence'))
    monkeypatch.setattr(harness,'extract_passport_page',lambda frame,result:frame)
    monkeypatch.setattr(sys,'argv',['webcam_research_vnext.py','--metrics',str(tmp_path/'metrics.jsonl')])
    harness.main()
    assert seen==[1]  # Button was pressed on frame 3; final uses selected frame 1.
    import json
    events=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert any(e['event']=='capture' and e['crop_ok'] for e in events)
