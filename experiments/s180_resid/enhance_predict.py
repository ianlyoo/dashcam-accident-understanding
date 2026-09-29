# S180: S174-only geometric crossings tend to precede annotated tire contact.
# S178 clears the side of S174-only crossings, while retaining the side of
# accepted S161 crossings. Apply one 0.2-second entry shift to that exact
# source; retain the S178 decision on every uncertainty or per-clip error.
try:
    _s180_previous_decide = decide

    def decide(result, fallback):
        previous = _s180_previous_decide(result, fallback)
        try:
            position, side = previous
            if (position is None or result.get('reason') != 'crossing'
                    or result.get('side') is not None):
                return previous
            fps = float(result['fps'])
            collision = int(result['collision_index'])
            if not (1.0 <= fps <= 120.0 and 0 <= position <= collision):
                return previous
            offset = int(round(0.2 * fps))
            return min(collision, position + offset), side
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return previous
except Exception:
    pass
