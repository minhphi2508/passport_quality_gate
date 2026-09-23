"""Safety-of-decision regressions. No real passport pixels or identity fixtures."""
import copy
import cv2
import numpy as np
import pytest
from passport_quality_gate import Analyzer,ProxyLocalizer
from passport_quality_gate.synthetic import sample,GUIDE

@pytest.fixture
def analyzer():
    cv2.setNumThreads(2)
    return Analyzer(ProxyLocalizer())

def mutate(kind):
    im,det,_=sample()
    rng=np.random.default_rng(71)
    x1,y1,x2,y2=192,184,832,584
    if kind=='blur': im=cv2.GaussianBlur(im,(0,0),3.)
    if kind=='noise_blur': im=np.clip(cv2.GaussianBlur(im,(0,0),3.).astype(float)+rng.normal(0,22,im.shape),0,255).astype(np.uint8)
    if kind=='low_contrast': im[y1:y2,x1:x2]=np.clip(160+(im[y1:y2,x1:x2].astype(float)-160)*.12,0,255)
    if kind=='missing_mrz': det.mrz_polygon=None
    if kind=='low_mrz_confidence': det.mrz_confidence=.1
    if kind=='unknown_corners': det.corners_reliable=False
    if kind=='low_page_confidence': det.confidence=.3
    if kind=='blank': im[y1:y2,x1:x2]=190
    if kind=='partial_mrz_erase': im[500:584,500:832]=215
    if kind=='mrz_only_blur': im[500:584,192:832]=cv2.GaussianBlur(im[500:584,192:832],(0,0),4.)
    if kind=='body_only_blur': im[245:490,380:810]=cv2.GaussianBlur(im[245:490,380:810],(0,0),4.)
    if kind=='low_resolution':
        im=cv2.resize(im,None,fx=.4,fy=.4);det.polygon*=.4;det.mrz_polygon*=.4
    return im,det

@pytest.mark.parametrize('kind',['blur','noise_blur','low_contrast','missing_mrz','low_mrz_confidence','low_page_confidence','blank','partial_mrz_erase','mrz_only_blur','body_only_blur','low_resolution'])
def test_bad_capture_never_becomes_ready(analyzer,kind):
    im,det=mutate(kind)
    for i in range(7): r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=i*.2,detection=det)
    assert r['state']=='ADJUST', (kind,r['quality'])
    assert not r['ready_for_capture'] and r['blocking_issues']
    assert analyzer.analyze_frame(im,GUIDE,'final',detection=det)['state']=='RETAKE'


def test_unknown_corners_are_advisory_in_preview(analyzer):
    im,det=mutate('unknown_corners')
    for i in range(7): r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=i*.2,detection=det)
    assert r['state']=='READY' and r['ready_for_capture']
    assert 'PERSPECTIVE_UNVERIFIED' in r['advisories']
    assert r['quality_evidence']['localization_grade']=='B'

def test_good_capture_can_become_ready(analyzer):
    im,det=mutate('clean')
    for i in range(7): r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=i*.2,detection=det)
    assert r['state']=='READY' and r['ready_for_capture']
    assert not r['blocking_issues']

def test_no_stale_ready_on_first_bad_frame(analyzer):
    im,det=mutate('clean')
    for i in range(7): analyzer.analyze_frame(im,GUIDE,'preview',timestamp=i*.2,detection=det)
    im,det=mutate('blur')
    r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=1.4,detection=det)
    assert r['state']!='READY' and not r['ready_for_capture']
    assert 'BLUR' in r['blocking_issues']

def test_missing_checks_do_not_pass_when_blur_disabled():
    a=Analyzer(ProxyLocalizer(),{'readability':{'enabled':False}})
    im,det=mutate('clean');r=a.analyze_frame(im,GUIDE,'final',detection=det)
    assert r['state']=='RETAKE' and 'QUALITY_UNCERTAIN' in r['blocking_issues']

def test_old_config_cannot_silently_drop_new_evidence_gate():
    a=Analyzer(ProxyLocalizer(),{'decision':{'priority':['PASSPORT_NOT_FOUND','BLUR']}})
    im,det=mutate('missing_mrz');r=a.analyze_frame(im,GUIDE,'final',detection=det)
    assert r['state']=='RETAKE' and 'MRZ_NOT_FOUND' in r['blocking_issues']

def test_reset_clears_motion_and_temporal(analyzer):
    im,det=mutate('clean');analyzer.analyze_frame(im,GUIDE,'preview',timestamp=5,detection=det)
    analyzer.reset()
    r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=0,detection=det)
    assert r['state']=='ADJUST'

def test_sharp_motion_jitter_is_not_an_immediate_capture_blocker(analyzer):
    # V4.1: geometry motion is advisory unless it is persistent/severe or is
    # accompanied by actual readability loss.  Detector jitter alone must not
    # make the user hold perfectly still forever.
    for i in range(10):
        im,det,_=sample('position',-.035 if i%2 else .035)
        r=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=.2*i,detection=det)
    assert r['quality']['blur_score'] < .6
    assert r['quality']['motion_score'] >= 0

def test_preview_final_detail_score_same_native_evidence(analyzer):
    im,det=mutate('blur');p=analyzer.analyze_frame(im,GUIDE,'preview',timestamp=0,detection=det)
    f=analyzer.analyze_frame(im,GUIDE,'final',detection=det)
    assert p['raw_metrics']['readability']['blur_score']==f['raw_metrics']['readability']['blur_score']

@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_timestamp_finite(analyzer,bad):
    im,det=mutate('clean')
    with pytest.raises(ValueError): analyzer.analyze_frame(im,GUIDE,'preview',timestamp=bad,detection=det)
