"""V4.3-fp1 regression tests for partial-page false READY.

These tests do not calibrate real passport accuracy.  They assert the product
invariant that a sharp visible fragment must not become READY merely because
MRZ/page detectors still fire on that fragment.
"""
import numpy as np
from passport_quality_gate import Analyzer, ProxyLocalizer
from passport_quality_gate.geometry import box_quad
from passport_quality_gate.synthetic import sample, GUIDE


def settle(analyzer, image, detection):
    result=None
    for i in range(8):
        result=analyzer.analyze_frame(image,GUIDE,'preview',timestamp=.2*i,detection=detection)
    return result


def _partial_detection(side):
    image,det,_=sample()
    p=np.asarray(det.polygon,np.float32).copy()
    m=np.asarray(det.mrz_polygon,np.float32).copy()
    x1=float(p[:,0].min()); x2=float(p[:,0].max()); width=x2-x1
    if side=='right':
        cut=x1+.62*width
        p=box_quad([x1,float(p[:,1].min()),cut,float(p[:,1].max())])
        m[:,0]=np.minimum(m[:,0],cut-3)
    elif side=='left':
        cut=x2-.62*width
        p=box_quad([cut,float(p[:,1].min()),x2,float(p[:,1].max())])
        m[:,0]=np.maximum(m[:,0],cut+3)
    else:
        raise ValueError(side)
    det.polygon=p
    det.mrz_polygon=m
    det.corners_reliable=False
    return image,det


def test_cornerless_complete_page_remains_capture_worthy():
    image,det,_=sample();det.corners_reliable=False
    r=settle(Analyzer(ProxyLocalizer()),image,det)
    assert r['capture_allowed']
    assert 'DOCUMENT_INCOMPLETE' not in r['blocking_issues']
    assert r['quality_evidence']['document_completeness_ok']


def test_internal_right_fragment_cannot_ready_even_if_visible_part_is_sharp():
    image,det=_partial_detection('right')
    r=settle(Analyzer(ProxyLocalizer()),image,det)
    assert not r['capture_allowed']
    assert 'DOCUMENT_INCOMPLETE' in r['blocking_issues']
    assert not r['quality_evidence']['document_completeness_ok']
    assert r['guidance_code']=='SHOW_ALL_EDGES'


def test_internal_left_fragment_cannot_ready_even_if_visible_part_is_sharp():
    image,det=_partial_detection('left')
    r=settle(Analyzer(ProxyLocalizer()),image,det)
    assert not r['capture_allowed']
    assert 'DOCUMENT_INCOMPLETE' in r['blocking_issues']
    assert not r['quality_evidence']['document_completeness_ok']
    assert r['guidance_code']=='SHOW_ALL_EDGES'
