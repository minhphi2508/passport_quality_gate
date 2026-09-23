import copy
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
import yaml
from passport_quality_gate import Analyzer,Detection,ProxyLocalizer,YoloLocalizer
from passport_quality_gate.geometry import box_quad,guide_polygon,order_quad,analyze_geometry
from passport_quality_gate.quality import exposure,blur,glare
from passport_quality_gate.decision import DecisionEngine,TemporalStabilizer
from passport_quality_gate.synthetic import sample,GUIDE

@pytest.fixture
def cfg(): return yaml.safe_load((Path(__file__).parents[1]/'configs/thresholds.yaml').read_text())

@pytest.fixture
def analyzer(cfg):
    cv2.setNumThreads(2)
    return Analyzer(ProxyLocalizer(),cfg)

@pytest.mark.parametrize('guide',[dict(x=-1,y=0,w=1,h=1),dict(x=0,y=0,w=0,h=1),dict(x=0,y=0,w=2,h=1)])
def test_invalid_guide(guide):
    with pytest.raises(ValueError): guide_polygon(guide,(768,1024,3))

def test_45_degree_quad_no_duplicate_corners():
    p=np.array([[0,1],[1,0],[2,1],[1,2]],np.float32)*100
    assert len(np.unique(order_quad(p),axis=0))==4

@pytest.mark.parametrize('kind,level,expected',[('scale',.35,'PASSPORT_TOO_FAR'),('scale',1.8,'CROPPED'),('position',.2,'MOVE_LEFT'),('position',-.2,'MOVE_RIGHT'),('position_y',.3,'MOVE_UP'),('position_y',-.3,'MOVE_DOWN'),('rotation',20,'ROTATED'),('rotation',180,'ROTATED'),('perspective',.8,'PERSPECTIVE_TOO_HIGH'),('exposure',.2,'TOO_DARK'),('exposure',2,'TOO_BRIGHT')])
def test_directional_issues(analyzer,kind,level,expected):
    im,d,_=sample(kind,level); r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    issue={x['code']:x for x in r['all_issues']}[expected]
    assert issue['active'], (expected,r['geometry'],r['quality'])
    # V4 final policy deliberately treats guide alignment as diagnostic only.
    if expected in {'PASSPORT_TOO_FAR','MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN'}:
        assert not issue['blocking']
    else:
        assert r['primary_issue']==expected

def test_no_passport_gates_quality(analyzer):
    im=np.zeros((200,300,3),np.uint8)
    r=analyzer.analyze_frame(im,GUIDE,'final',detection=Detection())
    assert r['primary_issue']=='PASSPORT_NOT_FOUND' and r['quality']=={} and analyzer.debug=={}

def test_clean_with_provisional_blur(analyzer):
    im,d,_=sample(); r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['state']=='ACCEPT'
    assert r['quality']['blur_score'] < .6 and r['production_validated'] is False
    assert r['raw_metrics']['blur']['blur_score'] is None
    assert r['raw_metrics']['blur']['mrz']['laplacian']>0

def test_bbox_does_not_claim_perspective(analyzer):
    im,d,_=sample(); d.corners_reliable=False; d.mrz_polygon=None
    r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['geometry']['perspective_score'] is None
    assert not r['geometry']['orientation_known']
    assert r['raw_metrics']['blur']['mrz'] is None

def test_background_does_not_change_metrics(analyzer):
    im,d,_=sample(); one=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    changed=im.copy(); changed[:140]=np.random.default_rng(1).integers(0,256,changed[:140].shape,dtype=np.uint8)
    two=analyzer.analyze_frame(changed,GUIDE,'final',detection=d)
    assert one['raw_metrics']['blur']['passport']['laplacian']==two['raw_metrics']['blur']['passport']['laplacian']
    assert one['raw_metrics']['exposure']==two['raw_metrics']['exposure']

def test_gaussian_monotonic_passport_and_mrz(analyzer):
    vals=[]
    for sigma in [0,1,2,3,5]:
        im,d,_=sample('gaussian',sigma); r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
        vals.append([r['raw_metrics']['blur'][region]['laplacian'] for region in ['passport','mrz']])
    assert (np.diff(vals,axis=0)<0).all()

@pytest.mark.parametrize('kind',['glare','colored_glare'])
def test_local_glare_mrz_mask(analyzer,kind):
    im,d,_=sample(kind,50); r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['raw_metrics']['glare']['mrz_overlap']>.01
    assert np.count_nonzero(analyzer.debug['glare_mask'])>0
    assert r['quality']['too_bright_score']<.6

def test_global_bright_separate_from_glare(analyzer):
    im,d,_=sample('exposure',2); r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['quality']['too_bright_score']>=.6

def test_configurable_priority(cfg):
    e=DecisionEngine(cfg); scores={'BLUR':1.,'GLARE':1.}
    assert e.primary(scores)=='BLUR'
    cfg['decision']['priority']=['GLARE','BLUR']
    assert e.primary(scores)=='GLARE'

def test_temporal_ready_persistence_glitch_and_loss(cfg):
    st=TemporalStabilizer(DecisionEngine(cfg)); clean={c:0. for c in cfg['decision']['priority']}
    assert st.update(clean,0,{})['state']=='ADJUST'
    for i in range(1,6): out=st.update(clean,i*.2,{})
    assert out['state']=='READY'
    bad=clean|{'BLUR':1.}
    assert st.update(bad,1.2,{})['state']=='READY'
    assert st.update(clean,1.4,{})['state']=='READY'
    for i in range(8,16): out=st.update(bad,i*.2,{})
    assert out['primary_issue']=='BLUR'
    assert st.update(clean|{'PASSPORT_NOT_FOUND':1.},3.2,{})['primary_issue']=='PASSPORT_NOT_FOUND'
    assert st.update(clean,3.4,{})['state']=='ADJUST'
    with pytest.raises(ValueError): st.update(clean,3.4,{})

def test_final_independent_from_preview(analyzer):
    im,d,_=sample(); first=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=0,detection=d)
    assert first['state']=='ADJUST'
    assert analyzer.analyze_frame(im,GUIDE,'final',detection=d)['state']=='ACCEPT'

def test_yolo_wrapper_class_mapping_and_mrz_association(cfg):
    class Scalar:
        def __init__(self,v): self.v=v
        def item(self): return self.v
    class Tensor:
        def __init__(self,v): self.v=v
        def cpu(self): return self
        def tolist(self): return self.v
    def box(cls,conf,b): return SimpleNamespace(cls=Scalar(cls),conf=Scalar(conf),xyxy=[Tensor(b)])
    model=SimpleNamespace(names={0:'mrz',1:'passport_page'})
    model.predict=lambda **kwargs: [SimpleNamespace(names=model.names,boxes=[box(1,.9,[20,20,280,180]),box(0,.95,[290,0,320,10]),box(0,.8,[30,140,270,170])])]
    localizer=YoloLocalizer(None,cfg['localization'],model=model)
    d=localizer.locate(np.full((200,330,3),100,np.uint8))
    assert d.found and d.mrz_confidence==.8
    assert d.confidence==.9
    model.predict=lambda **kwargs:[SimpleNamespace(names=model.names,boxes=None)]
    assert not localizer.locate(np.zeros((200,300,3),np.uint8)).found

@pytest.mark.parametrize('angle',[90,180,270])
def test_semantic_normalization_preserves_sharpness(analyzer,angle):
    im,d,_=sample('rotation',0); ref=analyzer.analyze_frame(im,GUIDE,'final',detection=d)['raw_metrics']['blur']
    im,d,_=sample('rotation',angle); actual=analyzer.analyze_frame(im,GUIDE,'final',detection=d)['raw_metrics']['blur']
    for region in ['passport','mrz']:
        assert actual[region]['laplacian']==pytest.approx(ref[region]['laplacian'],rel=.02)

def test_blur_calibration_is_explicit_and_mode_specific(cfg):
    a=Analyzer(ProxyLocalizer(),cfg); im,d,_=sample(); r=a.analyze_frame(im,GUIDE,'final',detection=d)
    cfg['blur'].update(calibrated=True,final_laplacian_reference=r['raw_metrics']['blur']['passport']['laplacian'],final_mrz_laplacian_reference=r['raw_metrics']['blur']['mrz']['laplacian'])
    a=Analyzer(ProxyLocalizer(),cfg)
    im,d,_=sample('gaussian',3); r=a.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['quality']['blur_score']>.9 and r['primary_issue']=='BLUR'
    with pytest.raises(ValueError): a.analyze_frame(im,GUIDE,'preview',detection=d)

def test_simultaneous_issues_preserved(analyzer):
    im,d,_=sample('exposure',.15); d.polygon=(d.polygon-d.polygon.mean(0))*.3+d.polygon.mean(0); d.mrz_polygon=None
    r=analyzer.analyze_frame(im,GUIDE,'final',detection=d)
    active=[i['code'] for i in r['all_issues'] if i['active']]
    assert 'PASSPORT_TOO_FAR' in active and 'TOO_DARK' in active
    assert r['primary_issue']=='TOO_DARK'
