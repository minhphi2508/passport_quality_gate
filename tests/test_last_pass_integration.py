from copy import deepcopy
from types import MappingProxyType
import cv2
import numpy as np
import pytest
from passport_quality_gate.api import PassportQualityGate, to_public_result
from passport_quality_gate.analyzer import Analyzer
from passport_quality_gate.localization import Detection
from passport_quality_gate.synthetic import sample, GUIDE
from passport_quality_gate.frame_selector import BestFrameSelector, BestFrameConfig

cv2.setNumThreads(2)

class Localizer:
    def __init__(self, detection): self.detection = detection
    def locate(self, frame, mode):
        return deepcopy(self.detection) if frame.any() else Detection()

@pytest.mark.parametrize('crop', [False, True])
@pytest.mark.parametrize('corners', [False, True])
def test_positive_final_contract_and_golden_decision(crop, corners):
    frame, det, _ = sample()
    det.corners_reliable = corners
    gate = PassportQualityGate(localizer=Localizer(det), device='cpu')
    context = 'document_crop' if crop else 'full_frame_final'
    raw = Analyzer(Localizer(det), gate.config).analyze_frame(frame, None if crop else GUIDE,
            'final', timestamp=1., capture_context=context)
    result = gate.analyze_document_crop(frame, timestamp=1.) if crop else gate.analyze_final(frame, GUIDE, timestamp=1.)
    assert raw['state'] == result['state'] == 'ACCEPT'
    # Reproduce immutable Golden defect; ensure wrapper corrects only aliases.
    assert raw['capture_allowed'] is False and raw['capture_quality_state'] == 'NOT_READY'
    assert result['capture_allowed'] and result['ready_for_capture']
    assert result['capture_quality_state'] == ('OPTIMAL' if corners else 'READY')
    assert result['blocking_issues'] == raw['blocking_issues'] == []
    assert result['quality'] == raw['quality']
    assert result['all_issues'] == raw['all_issues']
    public = to_public_result(raw)
    assert public['capture_allowed'] and public['workflow_state'] == 'ACCEPT'
    assert raw['capture_allowed'] is False  # no in-place mutation
    selector = BestFrameSelector()
    selector.push(frame, result, timestamp=1.)
    assert selector.select_recent(1.1) is not None

@pytest.mark.parametrize('crop', [False, True])
def test_public_methods_retake_and_accept(crop):
    frame, det, _ = sample()
    gate = PassportQualityGate(localizer=Localizer(det), device='cpu')
    analyze = (lambda f: gate.analyze_document_crop_public(f)) if crop else (lambda f: gate.analyze_final_public(f, GUIDE))
    assert analyze(frame)['capture_allowed']
    bad = analyze(np.zeros_like(frame))
    assert bad['workflow_state'] == 'RETAKE'
    assert not bad['capture_allowed'] and bad['capture_quality_state'] == 'NOT_READY'

@pytest.mark.parametrize('crop', [False, True])
def test_final_does_not_mutate_preview_state(crop):
    frame, det, _ = sample()
    gate = PassportQualityGate(localizer=Localizer(det), device='cpu')
    control = PassportQualityGate(localizer=Localizer(det), device='cpu')
    for t in [0., .2, .4]:
        gate.analyze_preview(frame, GUIDE, timestamp=t)
        control.analyze_preview(frame, GUIDE, timestamp=t)
    if crop: gate.analyze_document_crop(np.zeros_like(frame), timestamp=.5)
    else: gate.analyze_final(np.zeros_like(frame), GUIDE, timestamp=.5)
    a = gate.analyze_preview(frame, GUIDE, timestamp=.6)
    b = control.analyze_preview(frame, GUIDE, timestamp=.6)
    assert a['raw_metrics']['motion'] == b['raw_metrics']['motion']
    assert a['temporal'] == b['temporal']
    gate.reset()
    assert gate._analyzer.motion.previous is None
    assert gate._final_analyzer.debug == {}
    assert gate.analyze_preview(frame, GUIDE, timestamp=0.)['state'] == 'ADJUST'

def test_mapping_config_matches_dict():
    frame, det, _ = sample()
    cfg = {'normalization': {'preview_width': 360}}
    a = PassportQualityGate(localizer=Localizer(det), device='cpu', config=MappingProxyType(cfg))
    b = PassportQualityGate(localizer=Localizer(det), device='cpu', config=cfg)
    assert a.config == b.config
    assert cfg == {'normalization': {'preview_width': 360}}

@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_selector_rejects_invalid_clock_without_mutation(bad):
    s = BestFrameSelector()
    frame = np.zeros((20, 20, 3), np.uint8)
    s.push(frame, {'capture_allowed': True}, timestamp=1.)
    with pytest.raises(ValueError): s.push(frame, {'capture_allowed': True}, timestamp=bad)
    with pytest.raises(ValueError): s.select_recent(bad)
    assert s.buffered_frames == 1
    assert s.select_recent(1.1).timestamp == 1.
    assert s.select_recent(2.) is None

@pytest.mark.parametrize('field', ['window_ms','max_age_ms','max_frames','max_memory_mb','min_bbox_iou','recency_weight'])
def test_selector_config_finite(field):
    with pytest.raises(ValueError): BestFrameConfig(**{field: float('nan')})

def test_selector_rejects_reversed_clock_and_clear_starts_new_session():
    s = BestFrameSelector(); f = np.zeros((20,20,3),np.uint8)
    s.push(f, {'capture_allowed':True}, 2.)
    for t in [1.,2.]:
        with pytest.raises(ValueError): s.push(f, {'capture_allowed':True}, t)
    with pytest.raises(ValueError): s.select_recent(1.)
    s.clear(); s.push(f, {'capture_allowed':True}, 0.)
    assert s.select_recent(0.).timestamp == 0.
