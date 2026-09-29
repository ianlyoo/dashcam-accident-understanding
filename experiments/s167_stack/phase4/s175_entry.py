"""Run exact S174 entry policy on final S172 collisions, preserving S172 on error."""

def apply(ns, base, data_dir, model_dir):
    from pathlib import Path
    import ctypes
    import gc
    import time

    diag = {'load_error': None, 'clips': {}, 'track': True,
            'anchor': 'S172 final collision', 'fallback': 'exact S172 entry'}
    ns['_S175_DIAGNOSTICS'] = diag
    if len(base) == 0:
        return base
    root = Path(__file__).resolve().parents[2]
    tracker = None
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        pkg = ns['_s118_adapter']().load_package(root, 'model/stage2/s161')
        if getattr(pkg, '_s174_analyze', None) is not pkg.Tracker.analyze:
            raise RuntimeError('S174 entry extension did not initialize')
        tracker = pkg.Tracker(root / 'model/stage2/s161')
    except Exception as exc:
        diag['load_error'] = f'{type(exc).__name__}: {exc}'
        return base

    entries = []
    for row in base.itertuples(index=False):
        original = int(row.entry_frame)
        collision = int(row.collision_frame)
        record = {'s172_collision': collision, 's172_entry': original,
                  'accepted': False, 'error': None}
        entry = original
        try:
            paths = list(ns['_s008_frame_paths'](Path(data_dir) / 'images' / str(row.ID)))
            numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
            if len(numbers) > 0:
                adapter = ns['_s118_adapter']()
                anchor = adapter.position_of(numbers, collision)
                start = time.perf_counter()
                result = tracker.analyze(paths, numbers, anchor)
                record['seconds'] = time.perf_counter() - start
                position, _side = pkg.decide(result, 's109')
                if position is not None:
                    entry = int(numbers[max(0, min(int(position), len(numbers)-1))])
                    record['accepted'] = True
                entry = min(entry, collision)
                record['reason'] = result.get('reason')
                record['anchor_index'] = int(anchor)
        except Exception as exc:
            entry = original
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['final_entry'] = int(entry)
        entries.append(int(entry))
        diag['clips'][str(row.ID)] = record
    try:
        tracker.close()
    except Exception as exc:
        diag['close_error'] = f'{type(exc).__name__}: {exc}'
        return base
    out = base.copy()
    out['entry_frame'] = entries
    return out
