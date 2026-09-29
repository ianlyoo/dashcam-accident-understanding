# S178 side-only extension on exact S176. The entry decision and detector path
# remain S176; expose a side only for supported original S161 crossings or
# lateral collision anchors in the two accepted S176 already-inside buckets.
try:
    _s178_previous_analyze = Tracker.analyze

    def _s178_analyze(self, paths, numbers, collision_index):
        result = _s178_previous_analyze(self, paths, numbers, collision_index)
        side = None
        try:
            reason = result.get('reason')
            original = getattr(self, '_s176_original_result', None)
            if reason == 'crossing' and original is not None:
                # S174-only added crossings lost a confident CCD side panel.
                # Keep its inherited S109 side; retain the proven S161 boundary.
                if _s176_previous_decide(original, 's109')[0] is not None:
                    candidate = result.get('side')
                    if candidate in ('LEFT', 'RIGHT'):
                        side = candidate
            elif reason in ('s176_inside_from_start',
                            's176_inside_first_minus_0p2s') and original is not None:
                box = original.get('anchor_box')
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    center = (float(box[0]) + float(box[2])) / 2.0
                    if 0.0 <= center <= 0.35:
                        side = 'LEFT'
                    elif 0.65 <= center <= 1.0:
                        side = 'RIGHT'
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            side = None
        # The existing decide() does not consult side. Returning a copy keeps
        # the S176 entry path unchanged and makes an error fail closed per clip.
        out = Result(result)
        out['side'] = side
        return out

    Tracker.analyze = _s178_analyze
except Exception:
    pass
