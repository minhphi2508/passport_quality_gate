"""Targeted regressions layered on the stable V4.3-FP2 baseline.

These tests cover only the four user-observed regressions: partial-page false
READY, MRZ glare false negative, strong-shake flicker, and a slightly stricter
far-but-readable capture floor.  They are not OCR calibration tests.
"""
from passport_quality_gate import Analyzer, ProxyLocalizer
from passport_quality_gate.analyzer import _confirmed_glare_score
from passport_quality_gate.config import load_config
from passport_quality_gate.decision import DecisionEngine, TemporalStabilizer
from passport_quality_gate.synthetic import sample, GUIDE


def settle(analyzer, image, detection):
    r=None
    for i in range(8):
        r=analyzer.analyze_frame(image,GUIDE,'preview',timestamp=.2*i,detection=detection)
    return r


def test_asymmetric_missing_boundary_is_incomplete_without_rejecting_uniformly_weak_edges():
    engine=DecisionEngine(load_config())
    g=dict(page_aspect_ratio=1.42,mrz_aspect_ratio=9.0,mrz_width_ratio=.88,crop_risk=0.0)
    missing={'corners_verified':False,'corner_evidence':{
        'bbox_side_support':[.90,.18,.88,.86],
        'bbox_side_max_gap':[.05,.55,.05,.08],
        'bbox_corner_support':[.86,.16,.22,.84],
    }}
    weak={'corners_verified':False,'corner_evidence':{
        'bbox_side_support':[.22,.24,.20,.23],
        'bbox_side_max_gap':[.45,.42,.48,.44],
        'bbox_corner_support':[.18,.20,.17,.19],
    }}
    assert engine.document_incomplete_score(g,missing) >= engine.enter_threshold('DOCUMENT_INCOMPLETE')
    assert engine.document_incomplete_score(g,weak) < engine.enter_threshold('DOCUMENT_INCOMPLETE')


def test_connected_mrz_glare_component_blocks_but_old_broad_candidate_stays_permissive():
    cfg=load_config()['glare']
    broad={
        'glare_score':1.0,'area_ratio':.20,'mrz_overlap':.09,
        'critical_region_overlap':.33,'clipping_inside_glare':.39,'regions':[]
    }
    rd={'blur_score':0.0,'low_contrast_score':0.0}
    assert _confirmed_glare_score(broad,rd,cfg) < .10

    mrz_glare={**broad,'regions':[{
        'mrz_overlap_ratio':.05,'mrz_span_w':.18,'mrz_span_h':.62,
        'mean_context_delta':22.0,'clipping_ratio':.35,
    }]}
    assert _confirmed_glare_score(mrz_glare,rd,cfg) >= .60


def test_far_guard_moves_capture_floor_slightly_closer_without_restoring_guide_hard_gate():
    im,det,_=sample('scale',.65)
    r=settle(Analyzer(ProxyLocalizer()),im,det)
    assert r['capture_allowed']  # still readable enough; advisory remains acceptable

    im,det,_=sample('scale',.60)
    r=settle(Analyzer(ProxyLocalizer()),im,det)
    assert not r['capture_allowed']
    assert 'LOW_RESOLUTION' in r['blocking_issues']
    assert 'PASSPORT_TOO_FAR' in r['advisories']


def test_confirmed_hold_steady_does_not_flicker_ready_during_short_calm_dip():
    engine=DecisionEngine(load_config())
    temporal=TemporalStabilizer(engine)
    geom={'center_offset_x':0.,'center_offset_y':0.,'fill_ratio':1.0}
    high={'HOLD_STEADY':1.0,'PASSPORT_NOT_FOUND':0.0}
    low={'HOLD_STEADY':0.0,'PASSPORT_NOT_FOUND':0.0}
    r=None
    for t in [0.,.2,.4,.6,.8]:
        r=temporal.update(high,t,geom,blocking_codes=['HOLD_STEADY'])
    assert r['primary_issue']=='HOLD_STEADY'
    for t in [1.0,1.2,1.4,1.6,1.8,2.0]:
        r=temporal.update(low,t,geom,blocking_codes=['HOLD_STEADY'])
        assert r['state']=='ADJUST'
        assert r['primary_issue']=='HOLD_STEADY'
