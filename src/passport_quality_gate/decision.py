from .geometry import clip


class DecisionEngine:
    """Turn measurements into issue severities and policy-aware decisions.

    Scores are diagnostic severities in [0, 1], not probabilities.  Whether an
    issue blocks capture depends on capture context (preview vs final vs a
    document crop), so the same measurements can be interpreted differently
    without changing the quality algorithms themselves.
    """

    def __init__(self, config):
        self.config = config

    def enter_threshold(self, code):
        t = self.config['decision'].get('thresholds', {}).get(code, {})
        return float(t.get('enter', self.config['decision']['enter']))

    def exit_threshold(self, code):
        t = self.config['decision'].get('thresholds', {}).get(code, {})
        if 'exit' in t:
            return float(t['exit'])
        old = self.config['decision'].get('exit_by_issue', {}).get(code)
        return float(self.config['decision']['exit'] if old is None else old)

    def active(self, code, value, *, exit_level=False):
        if value is None:
            return False
        threshold = self.exit_threshold(code) if exit_level else self.enter_threshold(code)
        return float(value) >= threshold

    def blocking_codes(self, capture_context):
        return list(self.config['policy'][capture_context]['blocking'])


    def document_incomplete_score(self, g, evidence=None, capture_context='live_preview'):
        """Conservative TD3 completeness check for false-READY prevention.

        This is deliberately separate from guide alignment.  A page may be
        smaller/larger than the UI guide and still be capture-worthy, but a
        fragment of the data page must never become READY merely because the
        visible fragment is sharp.  The score uses broad TD3 geometry priors;
        it is not a document-authenticity signal.
        """
        if not g:
            return None
        cfg=self.config.get('completeness', {})
        if not cfg.get('enabled', True):
            return 0.0

        def low_score(value, good, fail):
            if value is None:
                return 0.0
            value=float(value); good=float(good); fail=float(fail)
            if good <= fail:
                return 0.0
            return clip((good-value)/(good-fail))

        page_aspect=low_score(g.get('page_aspect_ratio'),
                              cfg.get('page_aspect_good_min',1.30),
                              cfg.get('page_aspect_fail',1.16))
        mrz_aspect=low_score(g.get('mrz_aspect_ratio'),
                             cfg.get('mrz_aspect_good_min',7.0),
                             cfg.get('mrz_aspect_fail',5.3))
        mrz_width=low_score(g.get('mrz_width_ratio'),
                            cfg.get('mrz_width_ratio_good_min',0.76),
                            cfg.get('mrz_width_ratio_fail',0.58))

        # An actual physical-camera crop is already captured by CROPPED.  Feed
        # a bounded form into completeness too so the two diagnostics agree,
        # while still allowing guide-only mismatch to remain advisory.
        crop=float(g.get('crop_risk') or 0.0)
        edge_component=min(1.0, crop)

        # A single weak cue is not enough to reject a real passport.  Require a
        # strong page-shape failure or corroboration from MRZ geometry.  This
        # catches left/right fragments where YOLO and MRZ still fire on the
        # visible portion, which was the V4.2 false-READY failure mode.
        corroborated=clip(0.72*page_aspect + 0.45*max(mrz_aspect,mrz_width))
        severe_page=clip((page_aspect-float(cfg.get('single_cue_grace',0.45))) /
                         max(.05,1.0-float(cfg.get('single_cue_grace',0.45))))

        # V4.3-FP2.1: when verified corners are unavailable, use weak current-
        # frame boundary support around the detector box as a *corroborating*
        # completeness cue.  One weak corner/side while the others are strongly
        # supported is characteristic of a page fragment; uniformly weak edges
        # are treated as uncertainty rather than an automatic failure so a clean
        # passport on a low-contrast background is not rejected just because the
        # classical corner verifier is cautious.
        boundary_score=0.0
        if evidence and not evidence.get('corners_verified'):
            ce=evidence.get('corner_evidence') or {}
            side=ce.get('bbox_side_support') or []
            gaps=ce.get('bbox_side_max_gap') or []
            corners=ce.get('bbox_corner_support') or []
            if len(side)==4 and len(corners)==4:
                side=sorted(float(v) for v in side)
                corners=sorted(float(v) for v in corners)
                side_med=(side[1]+side[2])/2.0
                corner_med=(corners[1]+corners[2])/2.0
                # Asymmetry is more trustworthy than an absolute weak edge.
                # The thresholds are intentionally conservative and can be
                # inspected in corner_evidence during field tests.
                side_drop=max(0.0, side_med-side[0])
                corner_drop=max(0.0, corner_med-corners[0])
                side_asym=clip((side_drop-float(cfg.get('boundary_side_drop_good',0.28))) /
                               max(.05,float(cfg.get('boundary_side_drop_fail',0.62))-float(cfg.get('boundary_side_drop_good',0.28))))
                corner_asym=clip((corner_drop-float(cfg.get('boundary_corner_drop_good',0.30))) /
                                 max(.05,float(cfg.get('boundary_corner_drop_fail',0.62))-float(cfg.get('boundary_corner_drop_good',0.30))))
                support_gate=clip((max(side_med,corner_med)-float(cfg.get('boundary_median_support_min',0.30))) / .35)
                gap_risk=0.0
                if len(gaps)==4:
                    max_gap=max(float(v) for v in gaps)
                    gap_risk=clip((max_gap-float(cfg.get('boundary_gap_good',0.28))) /
                                  max(.05,float(cfg.get('boundary_gap_fail',0.68))-float(cfg.get('boundary_gap_good',0.28))))
                boundary_score=max(side_asym,corner_asym,gap_risk)*support_gate

        score=max(edge_component, corroborated, severe_page, boundary_score)

        # Verified corners reduce ambiguity, but do not zero the score: a
        # reliable quadrilateral with a wildly non-TD3 aspect still deserves a
        # completeness warning.  Reduce only moderate heuristic uncertainty.
        if evidence and evidence.get('corners_verified') and score < .85:
            score*=float(cfg.get('verified_corners_discount',0.55))

        # document_crop means guide framing is irrelevant, not that truncation
        # is acceptable.  Keep the same completeness semantics in every mode.
        return clip(score)

    def scores(self, found, g, q, evidence=None, capture_context='live_preview'):
        scores = {code: 0. for code in self.config['decision']['priority']}
        if not found:
            scores = {code: None for code in self.config['decision']['priority']}
            scores['PASSPORT_NOT_FOUND'] = 1.
            return scores

        c = self.config['geometry']
        fw = float(g.get('fill_w', 0.))
        fh = float(g.get('fill_h', 0.))
        min_dim = min(fw, fh)
        max_dim = max(fw, fh)
        min_target = min(float(c['preview_min_fill_w']), float(c['preview_min_fill_h']))
        max_target = max(float(c['preview_max_fill_w']), float(c['preview_max_fill_h']))
        too_far = clip((min_target - min_dim) / max(.01, c['preview_scale_transition']))
        too_close = clip((max_dim - max_target) / max(.01, c['preview_scale_transition']))

        # If the page is materially off-centre, the useful first action is a
        # translation rather than a scale instruction.  Keep scale diagnostics
        # but suppress their blocking strength until centring is approximately OK.
        off = max(abs(float(g.get('center_offset_x', 0.))), abs(float(g.get('center_offset_y', 0.))))
        if off > c['offset_tolerance'] * 1.25:
            too_far *= .45
            too_close *= .45

        scores.update(
            PASSPORT_TOO_FAR=too_far,
            PASSPORT_TOO_CLOSE=too_close,
            CROPPED=g.get('crop_risk'),
            DOCUMENT_INCOMPLETE=self.document_incomplete_score(g,evidence,capture_context),
            ROTATED=clip(abs(float(g.get('rotation_deg', 0.))) / c['rotation_tolerance_deg'] - 1),
            PERSPECTIVE_TOO_HIGH=None if g.get('perspective_score') is None else
                clip(float(g['perspective_score']) / c['perspective_tolerance'] - 1),
        )

        for axis, pos, neg in [('x', 'MOVE_LEFT', 'MOVE_RIGHT'), ('y', 'MOVE_UP', 'MOVE_DOWN')]:
            v = float(g.get('center_offset_' + axis, 0.))
            scores[pos if v > 0 else neg] = clip(abs(v) / c['offset_tolerance'] - 1)

        # Live preview: if a detected page is very near one physical camera
        # edge, recovery from that edge is more useful than a centre-of-visible-
        # fragment instruction.  This fixes cases where a partial left/right
        # passport incorrectly produced MOVE_UP / MOVE_DOWN.
        if capture_context == 'live_preview':
            fg = g.get('frame_edge_gaps') or {}
            margin = float(c.get('recovery_border_margin', 0.045))
            candidates = []
            mapping = {
                'left': 'MOVE_RIGHT',
                'right': 'MOVE_LEFT',
                'top': 'MOVE_DOWN',
                'bottom': 'MOVE_UP',
            }
            for edge, code in mapping.items():
                gap = fg.get(edge)
                if gap is not None and float(gap) < margin:
                    sev = clip((margin - float(gap)) / max(.005, margin))
                    candidates.append((sev, code, edge))
            if candidates:
                candidates.sort(reverse=True)
                sev, code, _ = candidates[0]
                # One dominant edge => give a directional action first.  If
                # several edges are simultaneously clipped, keep CROPPED.
                dominant = [x for x in candidates if x[0] >= sev - .15]
                if len(dominant) == 1:
                    for move in ('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN'):
                        scores[move] = 0.0
                    scores[code] = max(float(scores.get(code) or 0.0), sev)
                    # Do not suppress CROPPED.  In V4.2 guide-position MOVE
                    # issues are advisory, while a true physical crop remains a
                    # blocker.  Analyzer._guidance converts that blocker into a
                    # directional partner action when one missing edge is clear.

        for code, key in [('TOO_DARK', 'too_dark_score'), ('TOO_BRIGHT', 'too_bright_score'),
                          ('BLUR', 'blur_score'), ('GLARE', 'glare_score')]:
            scores[code] = q.get(key)
        for code, key in [('LOW_RESOLUTION', 'low_resolution_score'), ('LOW_CONTRAST', 'low_contrast_score'),
                          ('NOISE', 'noise_score')]:
            if code in scores:
                scores[code] = q.get(key)

        # FP2.1: keep guide mismatch advisory, but tighten the READY floor a
        # little when the page is both visibly far from the ideal guide and the
        # OCR-critical resolution metric is already borderline.  This avoids
        # turning the guide into a hard gate while still asking users to come
        # closer before the text scale becomes fragile.
        if capture_context == 'live_preview' and scores.get('LOW_RESOLUTION') is not None:
            guard=self.config.get('readability', {}).get('far_readability_guard', {})
            if guard.get('enabled', True):
                far_active=self.active('PASSPORT_TOO_FAR', scores.get('PASSPORT_TOO_FAR'))
                low=float(scores['LOW_RESOLUTION'])
                if far_active and low >= float(guard.get('min_low_resolution_score',0.50)):
                    scores['LOW_RESOLUTION']=max(low,self.enter_threshold('LOW_RESOLUTION'))

        # Motion is a capture risk only when it is either very strong by itself
        # or is accompanied by actual loss of detail.  Detector-box jitter must
        # not force HOLD_STEADY while the current frame is still sharp.
        motion = q.get('motion_score')
        if 'HOLD_STEADY' in scores:
            if motion is None:
                scores['HOLD_STEADY'] = None
            else:
                m = clip(float(motion))
                mc = self.config.get('motion', {})
                floor = float(mc.get('quality_coupling_floor', 0.20))
                blur_ref = max(.05, float(mc.get('blur_coupling_reference', 0.55)))
                b = clip(float(q.get('blur_score') or 0.0) / blur_ref)
                coupled = m * (floor + (1.0 - floor) * b)
                hard_start = float(mc.get('hard_block_start', 0.80))
                hard = clip((m - hard_start) / max(.05, 1.0 - hard_start))
                scores['HOLD_STEADY'] = max(coupled, hard)

        if evidence is not None:
            policy = self.config['policy'][capture_context]
            scores['MRZ_NOT_FOUND'] = float(policy.get('require_mrz', True) and not evidence.get('mrz_present', False))

            page_bad = not evidence.get('page_confident', False)
            grade = evidence.get('localization_grade', 'D')
            corners = evidence.get('corners_verified', False)
            border = float(g.get('border_margin', 0.))
            safe_border = float(policy.get('fallback_safe_border_margin', 0.0))

            # V4: absence of verified corners is not itself a failure.  A strong
            # page+MRZ localization is grade B and is usable for preview.  For a
            # final full-frame capture we only block a corner-less fallback when
            # it is also too close to the camera boundary to prove completeness.
            loc_bad = page_bad or grade in ('C', 'D', 'NONE')
            if capture_context == 'full_frame_final' and not corners and border < safe_border:
                loc_bad = True
            if capture_context == 'document_crop' and grade == 'B':
                loc_bad = False
            scores['LOCALIZATION_UNCERTAIN'] = float(loc_bad)

            scores['QUALITY_UNCERTAIN'] = float(
                not evidence.get('blur_observed', False)
                or not evidence.get('glare_observed', False)
                or (policy.get('require_text_evidence', True) and not evidence.get('text_detail_observed', False))
            )

        # Framing scores remain in diagnostics in every context.  The policy's
        # blocking list decides whether they can reject; final/document-crop
        # therefore keep useful measurements without confusing them with a gate.
        if capture_context == 'document_crop' and scores.get('CROPPED') is not None:
            scores['CROPPED'] = 0.
        return scores

    def primary(self, scores, threshold=None, allowed=None):
        allowed_set = None if allowed is None else set(allowed)
        for code in self.config['decision']['priority']:
            if allowed_set is not None and code not in allowed_set:
                continue
            value = scores.get(code)
            if value is None:
                continue
            if threshold is not None:
                active = value >= threshold
            else:
                active = self.active(code, value)
            if active:
                return code
        return None


class TemporalStabilizer:
    """One instance per camera stream. Timestamps are monotonic seconds."""
    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.config['temporal']
        self.reset()

    def reset(self):
        self.ema = {}
        self.stable = None
        self.candidate = None
        self.since = None
        self.last = None
        self.last_geometry = None
        self.recovered_since = None
        self.raw_clean_since = None
        self.hold_release_since = None

    def update(self, scores, timestamp, geometry, blocking_codes=None):
        if self.last is not None and timestamp <= self.last:
            raise ValueError('Preview timestamps must strictly increase')
        if self.last is not None and timestamp - self.last > self.cfg['reset_gap_s']:
            self.reset()
        g = [geometry.get('center_offset_x', 0), geometry.get('center_offset_y', 0), geometry.get('fill_ratio', 0)]
        if self.last_geometry is not None and max(abs(a-b) for a, b in zip(g, self.last_geometry)) > self.cfg['geometry_jump']:
            self.reset()
        self.last = timestamp
        self.last_geometry = g
        allowed = blocking_codes

        if self.engine.active('PASSPORT_NOT_FOUND', scores.get('PASSPORT_NOT_FOUND')):
            self.reset()
            self.last = timestamp
            self.stable = 'PASSPORT_NOT_FOUND'
            return dict(state='ADJUST', primary_issue=self.stable, smoothed_scores={}, pending=False)

        raw_issue = self.engine.primary(scores, allowed=allowed)
        # HOLD_STEADY is itself a temporal signal.  A one-frame motion spike
        # should not repeatedly restart the whole READY timer; persistence below
        # decides whether it becomes a stable blocker.
        raw_non_motion_issue = None if raw_issue == 'HOLD_STEADY' else raw_issue
        if raw_non_motion_issue is not None:
            self.raw_clean_since = None
        elif self.raw_clean_since is None:
            self.raw_clean_since = timestamp

        if self.stable and self.stable != 'READY' and not self.engine.active(self.stable, scores.get(self.stable)):
            if self.recovered_since is None:
                self.recovered_since = timestamp
        else:
            self.recovered_since = None
        recovered = self.recovered_since is not None and timestamp - self.recovered_since >= self.cfg.get('recovery_s', .8)

        a = self.cfg['ema_alpha']
        self.ema = {
            k: None if v is None else a*v + (1-a)*self.ema.get(k, v) if self.ema.get(k) is not None else v
            for k, v in scores.items()
        }
        candidate = self.engine.primary(self.ema, allowed=allowed)

        # FP2.1: once strong-shake HOLD_STEADY has become the stable state,
        # require a continuous calm interval before releasing it.  This prevents
        # green/yellow flicker when the motion estimate briefly dips below its
        # threshold during an ongoing shake.  Other higher-priority blockers can
        # still replace HOLD_STEADY immediately.
        if self.stable == 'HOLD_STEADY':
            hv=self.ema.get('HOLD_STEADY')
            calm=hv is None or hv < self.engine.exit_threshold('HOLD_STEADY')
            if calm:
                if self.hold_release_since is None:
                    self.hold_release_since=timestamp
            else:
                self.hold_release_since=None
            release_s=float(self.engine.config.get('motion',{}).get('release_s',1.20))
            higher_issue=candidate not in (None,'READY','HOLD_STEADY')
            released=self.hold_release_since is not None and timestamp-self.hold_release_since >= release_s
            if not higher_issue and not released:
                candidate='HOLD_STEADY'
        else:
            self.hold_release_since=None

        priority = self.engine.config['decision']['priority']
        if self.stable in priority:
            stable_value = self.ema.get(self.stable)
            above_exit = stable_value is not None and stable_value >= self.engine.exit_threshold(self.stable)
            if not recovered and above_exit:
                if candidate is None or priority.index(candidate) >= priority.index(self.stable):
                    candidate = self.stable

        candidate = candidate or 'READY'
        if candidate != self.candidate:
            self.candidate = candidate
            self.since = timestamp
        duration = self.cfg['ready_s'] if candidate == 'READY' else self.cfg['persist_s']
        if timestamp - self.since >= duration - 1e-9:
            if self.stable != candidate:
                self.recovered_since = None
            self.stable = candidate
        return dict(
            state='READY' if self.stable == 'READY' else 'ADJUST',
            primary_issue=self.stable,
            smoothed_scores=self.ema.copy(),
            pending=self.stable != candidate,
            clean_duration_s=0. if self.raw_clean_since is None else timestamp - self.raw_clean_since,
        )
