"""V4.3-FP2 regression tests for glare false blocking.

The old highlight mask is intentionally broad.  These tests assert that bright,
low-texture candidate regions do not block capture unless they also show OCR-
destructive evidence (MRZ overlap/clipping or readability damage).
"""
from passport_quality_gate.analyzer import _confirmed_glare_score


CFG={
    'impact_confirmation': {
        'enabled': True,
        'mrz_overlap_soft': .08,
        'mrz_overlap_hard': .35,
        'clip_soft': .25,
        'clip_hard': .75,
        'critical_overlap_soft': .20,
        'critical_overlap_hard': .60,
        'area_soft': .20,
        'area_hard': .45,
        'extreme_critical_soft': .45,
        'extreme_critical_hard': .75,
        'extreme_clip_soft': .55,
        'extreme_clip_hard': .85,
    }
}


def test_broad_bright_candidate_with_good_readability_does_not_block():
    gl={
        'glare_score':1.0,
        'area_ratio':.21,
        'mrz_overlap':.09,
        'critical_region_overlap':.33,
        'clipping_inside_glare':.39,
    }
    rd={'blur_score':0.0,'low_contrast_score':0.0}
    assert _confirmed_glare_score(gl,rd,CFG) < .10


def test_heavy_mrz_clipped_glare_is_blocking():
    gl={
        'glare_score':1.0,
        'area_ratio':.18,
        'mrz_overlap':.40,
        'critical_region_overlap':.25,
        'clipping_inside_glare':.85,
    }
    rd={'blur_score':0.0,'low_contrast_score':0.0}
    assert _confirmed_glare_score(gl,rd,CFG) >= .95


def test_body_glare_blocks_when_readability_is_damaged():
    gl={
        'glare_score':1.0,
        'area_ratio':.30,
        'mrz_overlap':0.0,
        'critical_region_overlap':.55,
        'clipping_inside_glare':.60,
    }
    rd={'blur_score':0.75,'low_contrast_score':0.80}
    assert _confirmed_glare_score(gl,rd,CFG) >= .70
