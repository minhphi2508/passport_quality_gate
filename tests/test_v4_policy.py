import numpy as np
import cv2
from passport_quality_gate import Analyzer, ProxyLocalizer
from passport_quality_gate.synthetic import sample, GUIDE


def test_final_guide_alignment_is_not_a_retake_reason():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample()
    # Deliberately tiny guide: diagnostic TOO_CLOSE may be active, but guide
    # alignment is a preview concern and must not reject a final still.
    tiny=dict(x=.35,y=.35,w=.3,h=.3)
    r=a.analyze_frame(im,tiny,'final',detection=d,capture_context='full_frame_final')
    issues={x['code']:x for x in r['all_issues']}
    assert issues['PASSPORT_TOO_CLOSE']['active']
    assert not issues['PASSPORT_TOO_CLOSE']['blocking']
    assert r['state']=='ACCEPT'


def test_document_crop_ignores_capture_framing():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample()
    d.corners_reliable=False
    r=a.analyze_frame(im,None,'final',detection=d,capture_context='document_crop')
    assert r['capture_context']=='document_crop'
    assert r['state']=='ACCEPT'
    assert 'PERSPECTIVE_UNVERIFIED' in r['advisories']


def test_full_frame_cornerless_near_border_is_blocked():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample()
    # Move the supplied fallback page box very near the physical frame edge;
    # without corners V4 cannot prove that all page edges are visible.
    h,w=im.shape[:2]
    d.polygon=np.array([[2,2],[w-2,2],[w-2,h-2],[2,h-2]],np.float32)
    d.corners_reliable=False
    # Keep MRZ inside the supplied page.
    r=a.analyze_frame(im,GUIDE,'final',detection=d,capture_context='full_frame_final')
    assert r['state']=='RETAKE'
    assert 'LOCALIZATION_UNCERTAIN' in r['blocking_issues']
    assert r['guidance_code']=='SHOW_ALL_EDGES'


def test_preview_cornerless_grade_b_can_ready():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample();d.corners_reliable=False
    for i in range(8):
        r=a.analyze_frame(im,GUIDE,'preview',timestamp=.2*i,detection=d,capture_context='live_preview')
    assert r['ready_for_capture']
    assert r['quality_evidence']['localization_grade']=='B'
    assert 'PERSPECTIVE_UNVERIFIED' in r['advisories']


def test_partner_guidance_is_action_not_internal_diagnostic():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample();d.confidence=.1
    r=a.analyze_frame(im,GUIDE,'final',detection=d,capture_context='full_frame_final')
    assert 'LOCALIZATION_UNCERTAIN' in r['blocking_issues']
    assert r['guidance_code']=='SHOW_ALL_EDGES'
    assert r['guidance']['source_issue']=='LOCALIZATION_UNCERTAIN'


def test_edge_gap_metrics_are_exposed():
    a=Analyzer(ProxyLocalizer())
    im,d,_=sample()
    r=a.analyze_frame(im,GUIDE,'final',detection=d)
    gaps=r['geometry']['edge_gaps']
    assert set(gaps)=={'left','right','top','bottom'}
    assert all(np.isfinite(list(gaps.values())))


def test_one_camera_edge_prefers_directional_recovery_over_vertical_offset():
    from passport_quality_gate.config import load_config
    from passport_quality_gate.decision import DecisionEngine
    engine=DecisionEngine(load_config())
    g=dict(
        fill_w=.9,fill_h=.9,center_offset_x=0.0,center_offset_y=-.30,
        crop_risk=.95,rotation_deg=0.0,perspective_score=None,border_margin=.005,
        frame_edge_gaps=dict(left=.002,right=.25,top=.18,bottom=.20),
    )
    q=dict(too_dark_score=0.,too_bright_score=0.,blur_score=0.,glare_score=0.,
           low_resolution_score=0.,low_contrast_score=0.,noise_score=0.,motion_score=0.)
    ev=dict(mrz_present=True,corners_verified=False,page_confident=True,
            text_detail_observed=True,blur_observed=True,glare_observed=True,
            localization_grade='B')
    scores=engine.scores(True,g,q,ev,'live_preview')
    assert engine.active('MOVE_RIGHT',scores['MOVE_RIGHT'])
    assert scores['MOVE_UP']==0.0
    assert engine.active('CROPPED',scores['CROPPED'])
