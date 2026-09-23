"""V4.2: ideal guide composition must not be confused with OCR capture-worthiness."""
import numpy as np
from passport_quality_gate import Analyzer, ProxyLocalizer
from passport_quality_gate.synthetic import sample, GUIDE


def settle(analyzer, image, detection, guide=GUIDE):
    result=None
    for i in range(8):
        result=analyzer.analyze_frame(image,guide,'preview',timestamp=.2*i,detection=detection)
    return result


def test_too_far_from_ideal_guide_is_ready_with_advisory_when_quality_is_good():
    im,det,_=sample()
    guide=dict(x=.02,y=.02,w=.96,h=.96)
    r=settle(Analyzer(ProxyLocalizer()),im,det,guide)
    assert r['capture_allowed']
    assert r['capture_quality_state']=='READY'
    assert not r['blocking_issues']
    assert 'PASSPORT_TOO_FAR' in r['advisories']
    assert r['recommended_adjustment']['code']=='MOVE_CLOSER'
    issue={x['code']:x for x in r['all_issues']}['PASSPORT_TOO_FAR']
    assert issue['active'] and not issue['blocking']


def test_too_close_to_ideal_guide_is_ready_if_page_is_not_physically_cropped():
    im,det,_=sample()
    guide=dict(x=.32,y=.30,w=.36,h=.38)
    r=settle(Analyzer(ProxyLocalizer()),im,det,guide)
    assert r['capture_allowed']
    assert not r['blocking_issues']
    assert 'PASSPORT_TOO_CLOSE' in r['advisories']
    assert r['recommended_adjustment']['code']=='MOVE_FARTHER'


def test_offcentre_inside_camera_is_advisory_not_capture_veto():
    im,det,_=sample()
    guide=dict(x=.05,y=.18,w=.68,h=.64)
    r=settle(Analyzer(ProxyLocalizer()),im,det,guide)
    assert r['capture_allowed']
    assert 'MOVE_LEFT' in r['advisories']
    assert r['recommended_adjustment']['code']=='MOVE_LEFT'


def test_quality_failure_still_blocks_even_when_too_far_is_only_advisory():
    im,det,_=sample('scale',.5)
    r=settle(Analyzer(ProxyLocalizer()),im,det,GUIDE)
    assert not r['capture_allowed']
    assert 'LOW_RESOLUTION' in r['blocking_issues']
    assert 'PASSPORT_TOO_FAR' in r['advisories']
    assert r['guidance_code']=='MOVE_CLOSER'


def test_ideal_clean_frame_is_optimal():
    im,det,_=sample()
    r=settle(Analyzer(ProxyLocalizer()),im,det,GUIDE)
    assert r['capture_allowed']
    assert r['capture_quality_state']=='OPTIMAL'
    assert r['guide_alignment_ideal']
    assert not r['advisories']


def test_true_physical_left_crop_still_blocks_but_guidance_is_directional():
    im,det,_=sample()
    det.polygon=np.array([[1,184],[641,184],[641,584],[1,584]],np.float32)
    det.corners_reliable=False
    r=settle(Analyzer(ProxyLocalizer()),im,det,GUIDE)
    assert not r['capture_allowed']
    assert 'CROPPED' in r['blocking_issues']
    assert r['guidance_code']=='MOVE_RIGHT'
