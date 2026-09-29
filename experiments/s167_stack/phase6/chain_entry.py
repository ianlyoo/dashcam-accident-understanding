"""Replay the S178/S180/S181 chain after exact S177 on S172 anchors."""

REASONS = frozenset(('s176_inside_from_start', 's176_inside_first_minus_0p2s'))
GAP_REASONS = frozenset(('short_track', 'not_in_lane_at_collision',
                         'far_inside_when_first_seen'))


def apply(ns, base, data_dir, model_dir, *, full):
    import ctypes
    import gc
    import importlib.util
    import time
    from pathlib import Path

    tag = '_S182_DIAGNOSTICS' if full else '_S179_DIAGNOSTICS'
    diag = {'load_error': None, 'close_error': None, 'clips': {},
            'anchor': 'S172 final collision', 'fallback': 'exact S177 row'}
    ns[tag] = diag
    if base.empty:
        return base
    root = Path(__file__).resolve().parents[2]
    tracker = None
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        source = root / ('model/stage2/s181_predict.py' if full else
                         'model/stage2/s178_predict.py')
        spec = importlib.util.spec_from_file_location('_s182_chain' if full else '_s179_side', source)
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        if full:
            assert package.Tracker.analyze is package._s181_analyze
            assert package._s181_previous_analyze is package._s178_analyze
            assert package._s180_previous_decide is not package.decide
        else:
            assert package.Tracker.analyze is package._s178_analyze
        tracker = package.Tracker(root / 'model/stage2/s161')
    except Exception as exc:
        diag['load_error'] = f'{type(exc).__name__}: {exc}'
        return base

    entries, sides = [], []
    for row in base.itertuples(index=False):
        old_entry = int(row.entry_frame)
        old_side = row.entry_side
        collision = int(row.collision_frame)
        entry, side = old_entry, old_side
        record = {'s177_entry': old_entry, 's177_side': old_side,
                  's172_collision': collision, 'error': None, 'rule': None}
        try:
            paths = list(ns['_s008_frame_paths'](Path(data_dir) / 'images' / str(row.ID)))
            numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
            if numbers:
                anchor = ns['_s118_adapter']().position_of(numbers, collision)
                start = time.perf_counter()
                result = tracker.analyze(paths, numbers, anchor)
                record['seconds'] = time.perf_counter() - start
                reason = result.get('reason')
                record['reason'] = reason
                record['anchor_index'] = int(anchor)
                proposed_side = result.get('side')
                if reason in REASONS | {'crossing'} and proposed_side in ('LEFT', 'RIGHT'):
                    side = proposed_side
                    record['rule'] = 'S178 side'
                if full:
                    shifted = (reason == 'crossing' and proposed_side is None)
                    gap = reason in GAP_REASONS and 's181_collision_index' in result
                    if shifted or gap:
                        position, _ = package.decide(result, 's109')
                        prior_position = (package._s180_previous_decide(result, 's109')[0]
                                          if shifted else None)
                        if position is not None and (gap or position != prior_position):
                            if not 0 <= int(position) < len(numbers):
                                raise ValueError('invalid chained entry position')
                            entry = min(int(numbers[int(position)]), collision)
                            record['rule'] = 'S180 shift' if shifted else 'S181 gap'
        except Exception as exc:
            entry, side = old_entry, old_side
            record['error'] = f'{type(exc).__name__}: {exc}'
        entries.append(entry)
        sides.append(side)
        record['final_entry'] = entry
        record['final_side'] = side
        diag['clips'][str(row.ID)] = record

    try:
        tracker.close()
    except Exception as exc:
        diag['close_error'] = f'{type(exc).__name__}: {exc}'
        return base
    out = base.copy()
    out['entry_frame'] = entries
    out['entry_side'] = sides
    return out
