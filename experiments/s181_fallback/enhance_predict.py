# S181: a collision-relative fallback only for S161 reasons with labeled,
# panel-held-out support. The 0.85-second gap is the median of 52 accepted
# S161 crossing-to-collision gaps on public clips, independent of fallback
# labels. S174 must have completed and abstained, and S176 must have abstained.
try:
    _s181_previous_analyze = Tracker.analyze
    _s181_previous_decide = decide
    _s181_reasons = frozenset(('short_track', 'not_in_lane_at_collision',
                               'far_inside_when_first_seen'))

    def _s181_analyze(self, paths, numbers, collision_index):
        result = _s181_previous_analyze(self, paths, numbers, collision_index)
        try:
            if (result.get('reason') not in _s181_reasons
                    or not getattr(self, '_s176_fallback_completed', False)
                    or _s181_previous_decide(result, 's109')[0] is not None):
                return result
            if not numbers:
                return result
            collision = max(0, min(len(numbers)-1, int(collision_index)))
            fps = 30.0 if len(paths) > self.params.long_n else 10.0
            out = Result(result)
            out['s181_collision_index'] = collision
            out['s181_fps'] = fps
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return result

    def decide(result, fallback):
        previous = _s181_previous_decide(result, fallback)
        try:
            if (previous[0] is not None or result.get('reason') not in _s181_reasons
                    or 's181_collision_index' not in result):
                return previous
            collision = result['s181_collision_index']
            fps = result['s181_fps']
            if (not isinstance(collision, int) or collision < 0
                    or fps not in (10.0, 30.0)):
                return previous
            entry = max(0, collision-int(round(0.85*fps)))
            return entry, None
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return previous

    Tracker.analyze = _s181_analyze
except Exception:
    pass
