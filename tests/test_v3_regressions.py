"""V3 geometry recovery and temporal safety regressions."""
import cv2
import numpy as np
import pytest
from passport_quality_gate import Analyzer,ProxyLocalizer
from passport_quality_gate.corners import estimate_corners
from passport_quality_gate.geometry import overlap
from passport_quality_gate.synthetic import sample,GUIDE
from passport_quality_gate.config import load_config
from passport_quality_gate.decision import TemporalStabilizer,DecisionEngine

@pytest.mark.parametrize('kind,level',[('clean',0),('gaussian',1),('rotation',8),('perspective',.3)])
def test_corners_have_geometric_accuracy(kind,level):
    im,d,_=sample(kind,level);b=[*d.polygon.min(0),*d.polygon.max(0)]
    q,e=estimate_corners(im,b,d.mrz_polygon)
    assert q is not None and e['verified']
    intersection=overlap(q,d.polygon)
    assert intersection/(abs(cv2.contourArea(q))+abs(cv2.contourArea(d.polygon))-intersection)>.95

@pytest.mark.parametrize('kind',['blank','missing_side','unrelated_inner_box','outside_mrz'])
def test_corners_cannot_be_invented(kind):
    im=np.full((480,640,3),100,np.uint8);bbox=[100,100,540,380];mrz=None
    if kind=='missing_side':
        for a,b in [((100,100),(540,100)),((540,100),(540,380)),((540,380),(100,380))]:cv2.line(im,a,b,(240,240,240),3)
    if kind=='unrelated_inner_box':cv2.rectangle(im,(180,160),(450,290),(250,250,250),3)
    if kind=='outside_mrz':
        cv2.rectangle(im,(100,100),(540,380),(250,250,250),3)
        mrz=np.array([[10,10],[80,10],[80,30],[10,30]],np.float32)
    q,e=estimate_corners(im,bbox,mrz)
    assert q is None and not e['verified']

def test_interrupted_edges_recover_without_closed_contour():
    im=np.full((480,640,3),100,np.uint8)
    for a,b in [((100,100),(540,100)),((540,100),(540,380)),((540,380),(100,380)),((100,380),(100,100))]:
        a,b=np.array(a),np.array(b)
        for lo,hi in [(0,.43),(.57,1)]:cv2.line(im,tuple((a+(b-a)*lo).astype(int)),tuple((a+(b-a)*hi).astype(int)),(240,240,240),2)
    q,e=estimate_corners(im,[100,100,540,380])
    assert q is not None and e['verified'] and e['proposal']=='lines'

def test_old_resolution_error_recovers_from_plateau():
    cfg=load_config();st=TemporalStabilizer(DecisionEngine(cfg))
    clean={c:0. for c in cfg['decision']['priority']}
    for i in range(6):out=st.update(clean|{'LOW_RESOLUTION':.8},i*.2,{})
    assert out['primary_issue']=='LOW_RESOLUTION'
    for i in range(6,25):out=st.update(clean|{'LOW_RESOLUTION':.55},i*.2,{})
    assert out['state']=='READY'

def test_unverified_corners_do_not_become_fake_current_blocker():
    a=Analyzer(ProxyLocalizer());im,d,_=sample()
    d.corners_reliable=False
    for i in range(7): r=a.analyze_frame(im,GUIDE,timestamp=i*.2,detection=d)
    assert r['ready_for_capture']
    assert 'LOCALIZATION_UNCERTAIN' not in r['blocking_issues']
    assert 'PERSPECTIVE_UNVERIFIED' in r['advisories']

def test_bad_frame_restarts_continuous_ready_evidence():
    a=Analyzer(ProxyLocalizer());im,d,_=sample()
    for i in range(8):r=a.analyze_frame(im,GUIDE,timestamp=i*.2,detection=d)
    assert r['ready_for_capture']
    bad=cv2.GaussianBlur(im,(0,0),4.)
    assert not a.analyze_frame(bad,GUIDE,timestamp=1.6,detection=d)['ready_for_capture']
    for t in [1.8,2.,2.2,2.4]:assert not a.analyze_frame(im,GUIDE,timestamp=t,detection=d)['ready_for_capture']
    assert a.analyze_frame(im,GUIDE,timestamp=2.6,detection=d)['ready_for_capture']

def test_raw_frame_size_and_final_contract():
    a=Analyzer(ProxyLocalizer());im,d,_=sample();d.corners_reliable=False
    r=a.analyze_frame(im,GUIDE,'final',detection=d)
    assert r['frame_size']==dict(width=im.shape[1],height=im.shape[0])
    assert r['state']=='ACCEPT' and not r['requires_final_check']
    assert 'PERSPECTIVE_UNVERIFIED' in r['advisories']

def test_detector_underestimates_page_width():
    im,d,_=sample()
    # Actual YOLO failure observed in the V3 benchmark: right bound clips page.
    q,e=estimate_corners(im,[198.6,187.8,747.5,581.9],d.mrz_polygon)
    assert q is not None and e['verified']
    intersection=overlap(q,d.polygon)
    assert intersection/(abs(cv2.contourArea(q))+abs(cv2.contourArea(d.polygon))-intersection)>.95

def test_nested_boundaries_do_not_silently_choose_one():
    im=np.full((480,640,3),100,np.uint8)
    cv2.rectangle(im,(100,100),(540,380),(240,240,240),2)
    cv2.rectangle(im,(113,112),(527,368),(240,240,240),2)
    q,e=estimate_corners(im,[100,100,540,380])
    assert q is None and e['reason']=='ambiguous_page_boundaries'

def test_webcam_capture_checks_raw_pixels_and_saves_only_on_opt_in(monkeypatch,tmp_path):
    import sys,json,importlib
    from pathlib import Path
    scripts_dir = Path(__file__).parents[1] / 'scripts'
    if not (scripts_dir / 'run_webcam_v3.py').is_file():
        pytest.skip('legacy webcam runner intentionally omitted from clean SDK source package')
    monkeypatch.syspath_prepend(str(scripts_dir))
    runner=importlib.import_module('run_webcam_v3')
    im,d,_=sample()
    class Localizer:
        def locate(self,frame,mode):
            assert np.array_equal(frame,im)
            return d
    analyzer=Analyzer(Localizer())
    class Capture:
        def isOpened(self):return True
        def set(self,*args):return True
        def get(self,*args):return 30.
        def read(self):return True,im.copy()
        def release(self):pass
    monkeypatch.setattr(runner,'build',lambda args:(analyzer,GUIDE))
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:Capture())
    monkeypatch.setattr(cv2,'imshow',lambda *args:None)
    keys=iter([ord('c'),ord('q')]);monkeypatch.setattr(cv2,'waitKey',lambda *args:next(keys))
    monkeypatch.setattr(cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(sys,'argv',['run_webcam_v3','--log',str(tmp_path/'log.jsonl'),'--save-captures',str(tmp_path/'captures')])
    runner.main()
    rows=[json.loads(x) for x in (tmp_path/'log.jsonl').read_text().splitlines()]
    final=[r for r in rows if r['mode']=='final']
    assert len(final)==1 and final[0]['state']=='ACCEPT'
    assert len(list((tmp_path/'captures').rglob('capture.png')))==1
    assert len(list((tmp_path/'captures').rglob('final.json')))==1
