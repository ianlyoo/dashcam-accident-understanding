# S176 extension on the exact S174 release. Apply only to two S161 abstentions
# after S174 itself has completed and abstained. No new detector or model.
try:
    _s176_previous_analyze = Tracker.analyze
    _s176_original_s161 = _s174_original_analyze
    _s176_original_fallback = _s174_fallback
    _s176_previous_decide = decide

    def _s176_capture_original(self, paths, numbers, collision_index):
        result = _s176_original_s161(self, paths, numbers, collision_index)
        self._s176_original_result = result
        return result

    def _s176_capture_fallback(self, paths, numbers, collision_index):
        result = _s176_original_fallback(self, paths, numbers, collision_index)
        self._s176_fallback_completed = True
        return result

    def _s176_analyze(self, paths, numbers, collision_index):
        self._s176_original_result = None
        self._s176_fallback_completed = False
        s174 = _s176_previous_analyze(self, paths, numbers, collision_index)
        try:
            # An accepted S161 or S174 decision keeps exactly its prior result.
            if _s176_previous_decide(s174, 's109')[0] is not None:
                return s174
            if not self._s176_fallback_completed:
                return s174  # S174 faulted or never ran: fail closed to S174.
            original = self._s176_original_result
            if original is None or not numbers:
                return s174
            reason = original.get('reason')
            if reason == 'inside_from_start':
                index = 0
                tag = 's176_inside_from_start'
            elif reason == 'inside_when_first_seen':
                first = int(original['entry_index'])
                fps = 30.0 if len(paths) > self.params.long_n else 10.0
                index = max(0, first - int(round(0.2 * fps)))
                tag = 's176_inside_first_minus_0p2s'
            else:
                return s174
            c = max(0, min(len(numbers)-1, int(collision_index)))
            if index > c:
                return s174
            return Result(reason=tag, entry_index=int(index),
                          collision_index=c, s176_s161_reason=reason)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return s174

    def decide(result, fallback):
        if result.get('reason') in ('s176_inside_from_start',
                                    's176_inside_first_minus_0p2s'):
            index = result.get('entry_index')
            collision = result.get('collision_index')
            if isinstance(index, int) and isinstance(collision, int) and 0 <= index <= collision:
                return index, None
            return None, None
        return _s176_previous_decide(result, fallback)

    _s174_original_analyze = _s176_capture_original
    _s174_fallback = _s176_capture_fallback
    Tracker.analyze = _s176_analyze
except Exception:
    # An initialization issue retains the exact S174 behavior.
    pass
