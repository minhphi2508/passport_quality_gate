"""Image-level viewport replays plus V4 contract regressions; no real identities."""
from copy import deepcopy
import cv2
import numpy as np
import pytest
from passport_quality_gate.api import PassportQualityGate
from passport_quality_gate.viewport import CaptureViewport, map_preview_viewport, orient_camera
from passport_quality_gate.localization import Detection
from passport_quality_gate.synthetic import sample

V4='configs/research_v4.yaml'

class RecordingLocalizer:
    def __init__(self, det=None): self.det=det or Detection(); self.frames=[]
    def locate(self,frame,mode='preview'):
        self.frames.append(frame.copy())
        return deepcopy(self.det)

def replay_frames():
    roi,det,_=sample()
    h,w=roi.shape[:2]; shape=(h+160,w+200,3)
    rng=np.random.default_rng(183)
    frames=[np.zeros(shape,np.uint8),np.full(shape,255,np.uint8),rng.integers(0,256,shape,dtype=np.uint8)]
    for f in frames: f[80:80+h,100:100+w]=roi
    return frames,CaptureViewport(100/shape[1],80/shape[0],w/shape[1],h/shape[0]),det

def semantic(result):
    return {k:result[k] for k in ('localization','quality','state','guidance_code','blocking_issues','research','quality_evidence')}

@pytest.mark.parametrize('mode',['preview','final'])
def test_image_replay_outside_pixels_never_enter_localizer_or_change_outputs(mode):
    frames,v,det=replay_frames(); outputs=[]; seen=[]
    for frame in frames:
        loc=RecordingLocalizer(det); gate=PassportQualityGate(config=V4,localizer=loc)
        fn=getattr(gate,'analyze_capture_'+mode)
        outputs.append(semantic(fn(frame,v,timestamp=0.))); seen.append(loc.frames[0])
    assert all(np.array_equal(seen[0],f) for f in seen)
    assert outputs[0]==outputs[1]==outputs[2]
    assert seen[0].shape==(768,1024,3)

@pytest.mark.parametrize('rotation',[0,90,180,270])
@pytest.mark.parametrize('mirror',[False,True])
@pytest.mark.parametrize('mode',['fit','fill'])
def test_screen_mapping_matches_oriented_buffer(rotation,mirror,mode):
    raw=np.arange(60*100*3,dtype=np.uint8).reshape(60,100,3)
    oriented=orient_camera(raw,rotation); h,w=oriented.shape[:2]
    pw,ph=210,170; scale=(min if mode=='fit' else max)(pw/w,ph/h)
    ox,oy=(pw-w*scale)/2,(ph-h*scale)/2
    # Known interior pixel rectangle projected into screen coordinates.
    x,y,rw,rh=20,15,30,25
    sx=(w-x-rw if mirror else x)*scale+ox
    rect=(sx,y*scale+oy,rw*scale,rh*scale)
    v,meta=map_preview_viewport(rect,(pw,ph),(100,60),rotation_clockwise=rotation,mirrored=mirror,mode=mode)
    roi,info=v.extract(oriented,transform=meta)
    assert np.array_equal(roi,oriented[y:y+rh,x:x+rw])
    assert info['viewport_pixel_rect']==[x,y,x+rw,y+rh]

def test_round_inward_clamp_and_reject_empty():
    assert CaptureViewport(.101,.201,.501,.501).pixel_rect((100,100,3))==(11,21,60,70)
    assert CaptureViewport(-.1,-.1,1.2,1.2).pixel_rect((100,100,3))==(0,0,100,100)
    with pytest.raises(ValueError): CaptureViewport(2,2,.2,.2).pixel_rect((100,100,3))

@pytest.mark.parametrize('config',[None,'configs/research_vnext.yaml'])
def test_old_profiles_do_not_silently_adopt_viewport(config):
    gate=PassportQualityGate(config=config,localizer=RecordingLocalizer())
    with pytest.raises(ValueError): gate.analyze_roi_preview(np.zeros((100,100,3),np.uint8))

from passport_quality_gate.config import load_config
from passport_quality_gate.geometry import box_quad
from passport_quality_gate.motion import MotionAnalyzer
from passport_quality_gate.research import ResearchSession, content_at_cut
from passport_quality_gate.frame_selector import BestFrameSelector
from passport_quality_gate.capture_output import extract_passport_page

@pytest.mark.parametrize('side,action',[('left','MOVE_RIGHT'),('right','MOVE_LEFT'),('top','MOVE_DOWN'),('bottom','MOVE_UP')])
def test_partial_mrz_is_blocked_directionally_without_quality_diagnosis(side,action):
    image,det,_=sample(); h,w=image.shape[:2]
    if side=='left': det.mrz_polygon[[0,3],0]=0
    if side=='right': det.mrz_polygon[[1,2],0]=w-1
    if side=='top': det.mrz_polygon[[0,1],1]=0
    if side=='bottom': det.mrz_polygon[[2,3],1]=h-1
    gate=PassportQualityGate(config=V4,localizer=RecordingLocalizer(det))
    r=gate.analyze_roi_final(image,timestamp=0.)
    assert r['state']=='RETAKE'
    assert r['guidance_code']==action
    assert r['research']['mrz']['state']=='INCOMPLETE'
    assert not r['research']['mrz_glare']['observed']
    assert r['localization']['mrz_polygon'] is None


def test_implausibly_short_detected_mrz_blocks_without_fallback():
    image,det,_=sample(); det.mrz_polygon[:,0]=det.mrz_polygon[:,0]*.3+350
    r=PassportQualityGate(config=V4,localizer=RecordingLocalizer(det)).analyze_roi_final(image)
    assert r['state']=='RETAKE'
    assert r['guidance_code']=='SHOW_BOTTOM_TEXT'
    assert r['research']['mrz']['fallback'] is None

@pytest.mark.parametrize('side,action',[('left','MOVE_RIGHT'),('right','MOVE_LEFT'),('top','MOVE_DOWN'),('bottom','MOVE_UP')])
def test_actual_roi_content_cut_beats_low_resolution(side,action):
    frame=np.full((550,800,3),235,np.uint8)
    # ROI boundaries are far from the RAW image boundaries.
    viewport=CaptureViewport(50/800,50/550,700/800,450/550)
    for t in (110,160,210):
        if side=='left': cv2.rectangle(frame,(40,t+50),(62,t+58),(25,25,25),-1)
        elif side=='right': cv2.rectangle(frame,(737,t+50),(758,t+58),(25,25,25),-1)
        elif side=='top': cv2.rectangle(frame,(t+50,40),(t+58,62),(25,25,25),-1)
        else: cv2.rectangle(frame,(t+50,487),(t+58,510),(25,25,25),-1)
    det=Detection(box_quad([0,0,699,449]),None,1.,'test',False,0.)
    gate=PassportQualityGate(config=V4,localizer=RecordingLocalizer(det))
    r=gate.analyze_capture_final(frame,viewport)
    assert r['state']=='RETAKE'
    assert r['research']['cut_scores'][side]>=.55
    assert r['guidance_code']==action
    session=ResearchSession(load_config(V4)['research'])
    r=session.guidance_for({},['DOCUMENT_INCOMPLETE','LOW_RESOLUTION'],
        {'DOCUMENT_INCOMPLETE':.6,'LOW_RESOLUTION':1.},
        {'page_reliable':True,'cut_scores':{side:.6}},0.,'preview')
    assert r['code']==action


def test_blank_margin_does_not_require_all_four_page_corners():
    image,det,_=sample()
    # Crop to exact page bounds, with an inward one-pixel blank trim.
    roi=image[185:584,193:832].copy()
    det.polygon-=np.array([193,185]); det.mrz_polygon-=np.array([193,185])
    det.corners_reliable=False
    gate=PassportQualityGate(config=V4,localizer=RecordingLocalizer(det))
    r=gate.analyze_roi_final(roi)
    assert r['state']=='ACCEPT',r['blocking_issues']

@pytest.mark.parametrize('resolution',[1,2])
def test_page_relative_motion_is_invariant_to_roi_size_and_resolution(resolution):
    config=load_config(V4)['motion']; outputs=[]
    for shape in [(600,900,3),(1200,1800,3)]:
        motion=MotionAnalyzer(config); seq=[]
        for i,dx in enumerate([0,2,-2,1,0,120,-120,120]):
            p=box_quad([100+dx,100,700+dx,500])*resolution
            det=Detection(p,None,1.)
            seq.append(motion.update(det,tuple(v*resolution for v in shape[:2])+(3,),i*.1))
        outputs.append(seq)
    assert outputs[0]==outputs[1]
    assert max(v['score'] for v in outputs[0][:5])==0.
    assert outputs[0][-1]['score']>=.9


def test_selected_roi_identity_final_crop_and_no_outside_leak():
    frames,v,det=replay_frames(); loc=RecordingLocalizer(det)
    gate=PassportQualityGate(config=V4,localizer=loc); selector=BestFrameSelector()
    for i in range(10):
        roi,meta=v.extract(frames[i%3])
        r=gate.analyze_roi_preview(roi,timestamp=i*.2,viewport_metadata=meta)
        selector.push(roi,r,timestamp=i*.2)
    selected=selector.select_recent(trigger_timestamp=1.85)
    assert selected is not None
    frozen=selected.frame.copy()
    frames[0][:]=255
    result=gate.analyze_roi_final(selected.frame,timestamp=1.85,viewport_metadata=selected.result['capture_viewport'])
    assert np.array_equal(loc.frames[-1],frozen)
    assert result['state']=='ACCEPT'
    crop=extract_passport_page(selected.frame,result)
    assert crop.size>0
    assert selected.frame.nbytes<frames[0].nbytes


def test_viewport_change_invalidates_debounce_even_at_same_roi_size():
    frames,v,det=replay_frames();loc=RecordingLocalizer(det);gate=PassportQualityGate(config=V4,localizer=loc)
    gate.analyze_capture_preview(frames[0],v,timestamp=0.)
    loc.det.mrz_polygon=None;loc.det.mrz_confidence=0.
    changed=CaptureViewport(v.x+.01,v.y,v.w,v.h)
    r=gate.analyze_capture_preview(frames[0],changed,timestamp=.1)
    assert not r['research']['mrz']['held']
    assert not r['capture_allowed']


def test_v4_missing_current_mrz_only_debounces_presence():
    image,det,_=sample();loc=RecordingLocalizer(det);gate=PassportQualityGate(config=V4,localizer=loc)
    for i in range(10): r=gate.analyze_roi_preview(image,timestamp=i*.2)
    assert r['capture_allowed']
    loc.det.mrz_polygon=None;loc.det.mrz_confidence=0.
    r=gate.analyze_roi_preview(image,timestamp=2.)
    assert r['capture_allowed']
    assert not r['research']['mrz_glare']['observed']
    r=gate.analyze_roi_preview(image,timestamp=2.3)
    assert not r['capture_allowed']
    assert r['guidance_code']=='SHOW_BOTTOM_TEXT'
    assert gate.analyze_roi_final(image)['state']=='RETAKE'

@pytest.fixture(scope='module')
def real_gate():
    import torch
    torch.set_num_threads(2); cv2.setNumThreads(1)
    return PassportQualityGate(config=V4,device='cpu')


def test_packaged_yolo_image_replay_outside_invariance(real_gate):
    from pathlib import Path
    image=cv2.imread(str(Path(__file__).parent/'fixtures/v4/non_document_proxy.png'))
    v=CaptureViewport(.16,.18,.68,.64); x0,y0,x1,y1=v.pixel_rect(image.shape)
    variants=[np.zeros_like(image),np.full_like(image,255),np.random.default_rng(73).integers(0,256,image.shape,dtype=np.uint8),np.roll(image,320,axis=0)]
    results=[]
    for variant in variants:
        variant[y0:y1,x0:x1]=image[y0:y1,x0:x1]
        results.append(semantic(real_gate.analyze_capture_final(variant,v,timestamp=0.)))
    assert results.count(results[0])==len(results)
    assert results[0]['localization']['mrz_confidence']>0.5
    assert 'CROPPED' not in results[0]['blocking_issues']


def test_packaged_yolo_raw_mrz_outside_viewport_never_rescues(real_gate):
    from pathlib import Path
    image=cv2.imread(str(Path(__file__).parent/'fixtures/v4/non_document_proxy.png'))
    full=real_gate.analyze_roi_final(image)
    assert full['localization']['mrz_confidence']>.5
    viewport=CaptureViewport(.16,.18,.68,.44)
    real_gate.reset()
    for t in [0.,.2,.4,.6,1.]:
        r=real_gate.analyze_capture_preview(image,viewport,timestamp=t)
        assert not r['capture_allowed']
        assert r['research']['mrz']['state']=='ABSENT'
    r=real_gate.analyze_capture_final(image,viewport)
    assert r['state']=='RETAKE'
    assert r['guidance_code']=='SHOW_BOTTOM_TEXT'
    r=real_gate.analyze_capture_final(image,CaptureViewport(0,0,.16,1))
    assert r['guidance_code']=='PLACE_PASSPORT_IN_FRAME'
    assert not r['passport_found']

@pytest.mark.parametrize('kind',['bright','damage'])
def test_v4_preserves_glare_damage_rule(kind):
    image,det,_=sample(); x,y=det.mrz_polygon.min(0).astype(int); xmax,ymax=det.mrz_polygon.max(0).astype(int)
    if kind=='bright':
        roi=image[y:ymax,x:xmax]; ink=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)<150;roi[~ink]=255
    else: cv2.rectangle(image,(x+220,y+8),(x+255,ymax-4),(255,255,255),-1)
    r=PassportQualityGate(config=V4,localizer=RecordingLocalizer(det)).analyze_roi_final(image)
    assert ('MRZ_GLARE' in r['blocking_issues'])==(kind=='damage')


def test_stage_prerequisites_and_causal_motion_v4():
    image,det,_=sample();loc=RecordingLocalizer(det);gate=PassportQualityGate(config=V4,localizer=loc)
    gate._analyzer.motion.update=lambda *args: {'score':1.}
    loc.det.mrz_polygon=None;loc.det.mrz_confidence=0.
    r=gate.analyze_roi_preview(image,timestamp=0.)
    assert r['guidance_code']=='HOLD_STEADY'
    loc.det=Detection()
    r=gate.analyze_roi_preview(image,timestamp=.1)
    assert r['guidance_code']=='PLACE_PASSPORT_IN_FRAME'


def test_relative_motion_same_at_double_camera_resolution():
    sequences=[]
    for factor in [1,2]:
        motion=MotionAnalyzer(load_config(V4)['motion']);seq=[]
        for i,dx in enumerate([0,2,-2,1,0,120,-120,120]):
            p=box_quad([100+dx,100,700+dx,500])*factor
            seq.append(motion.update(Detection(p,None,1.),(600*factor,900*factor,3),i*.1)['score'])
        sequences.append(seq)
    assert sequences[0]==pytest.approx(sequences[1])


def test_fallback_telemetry_toggle_has_zero_v4_authority():
    image,det,_=sample();det.mrz_polygon=None;det.mrz_confidence=0.
    results=[]
    for enabled in [False,True]:
        config=load_config(V4);config['research']['fallback_telemetry']=enabled
        gate=PassportQualityGate(config=config,localizer=RecordingLocalizer(det))
        r=gate.analyze_roi_final(image)
        results.append([r[k] for k in ['state','guidance_code','quality','blocking_issues']])
        assert not r['capture_allowed']
        assert all(v==0 for v in r['research']['expected_outside'].values())
    assert results[0]==results[1]
