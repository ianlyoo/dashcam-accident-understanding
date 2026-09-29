

# S185: S178 original-S161 crossing side, S180 S174-only shift, and S181
# fallback-gap eligibility on exact S174. No S176 bucket rule or source.
_s185_previous_analyze = Tracker.analyze
_s185_s161_analyze = _s174_original_analyze
_s185_s174_fallback = _s174_fallback
_s185_gap_reasons = frozenset(('short_track', 'not_in_lane_at_collision',
                              'far_inside_when_first_seen'))


def _s185_capture_s161(self, paths, numbers, collision_index):
    result = _s185_s161_analyze(self, paths, numbers, collision_index)
    self._s185_s161_result = result
    return result


def _s185_capture_s174(self, paths, numbers, collision_index):
    result = _s185_s174_fallback(self, paths, numbers, collision_index)
    self._s185_s174_completed = True
    return result


_s174_original_analyze = _s185_capture_s161
_s174_fallback = _s185_capture_s174


def _s185_analyze(self, paths, numbers, collision_index):
    self._s185_s161_result = None
    self._s185_s174_completed = False
    result = _s185_previous_analyze(self, paths, numbers, collision_index)
    original = self._s185_s161_result
    out = Result(result)
    old_position = decide(original, 's109')[0] if original is not None else None
    new_position = decide(result, 's109')[0]
    if new_position is not None and result.get('reason') == 'crossing':
        out['s185_source'] = 'S161' if old_position is not None else 'S174'
    else:
        out['s185_source'] = 'S109'
    out['s185_s161_reason'] = original.get('reason') if original is not None else None
    out['s185_s174_completed'] = bool(self._s185_s174_completed)
    side = result.get('side')
    out['s185_side'] = (side if out['s185_source'] == 'S161' and
                        side in ('LEFT', 'RIGHT') else None)
    if (out['s185_source'] == 'S109' and self._s185_s174_completed and
            out['s185_s161_reason'] in _s185_gap_reasons and numbers):
        collision = max(0, min(len(numbers)-1, int(collision_index)))
        out['s185_gap_collision_index'] = collision
        out['s185_gap_fps'] = 30.0 if len(paths) > self.params.long_n else 10.0
    return out


Tracker.analyze = _s185_analyze
