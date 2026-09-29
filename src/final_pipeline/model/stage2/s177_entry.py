"""Apply S176 abstention buckets to S175 rows on final S172 collision anchors."""


def apply(ns, base, data_dir, model_dir):
    import ctypes
    import gc
    import importlib.util
    import time
    from pathlib import Path

    diag = {'load_error': None, 'close_error': None, 'clips': {},
            'anchor': 'S172 final collision', 'fallback': 'exact S175 entry'}
    ns['_S177_DIAGNOSTICS'] = diag
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
        source = root / 'model/stage2/s176_predict.py'
        spec = importlib.util.spec_from_file_location('_s177_pinned_s176', source)
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        if package.Tracker.analyze is not package._s176_analyze:
            raise RuntimeError('S176 bucket extension did not initialize')
        tracker = package.Tracker(root / 'model/stage2/s161')
    except Exception as exc:
        diag['load_error'] = f'{type(exc).__name__}: {exc}'
        return base

    entries = []
    bucket_reasons = {'s176_inside_from_start',
                      's176_inside_first_minus_0p2s'}
    for row in base.itertuples(index=False):
        old_entry = int(row.entry_frame)
        collision = int(row.collision_frame)
        entry = old_entry
        record = {'s175_entry': old_entry, 's172_collision': collision,
                  'bucket': None, 'error': None}
        try:
            paths = list(ns['_s008_frame_paths'](Path(data_dir) / 'images' / str(row.ID)))
            numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
            if numbers:
                anchor = ns['_s118_adapter']().position_of(numbers, collision)
                start = time.perf_counter()
                result = tracker.analyze(paths, numbers, anchor)
                record['seconds'] = time.perf_counter() - start
                record['reason'] = result.get('reason')
                record['anchor_index'] = int(anchor)
                if result.get('reason') in bucket_reasons:
                    position, _ = package.decide(result, 's109')
                    if position is None or not 0 <= int(position) < len(numbers):
                        raise ValueError('invalid S176 bucket position')
                    entry = min(int(numbers[int(position)]), collision)
                    record['bucket'] = result['reason']
                    record['s161_reason'] = result.get('s176_s161_reason')
        except Exception as exc:
            entry = old_entry
            record['error'] = f'{type(exc).__name__}: {exc}'
        entries.append(entry)
        record['final_entry'] = entry
        diag['clips'][str(row.ID)] = record

    try:
        tracker.close()
    except Exception as exc:
        diag['close_error'] = f'{type(exc).__name__}: {exc}'
        return base
    out = base.copy()
    out['entry_frame'] = entries
    return out
