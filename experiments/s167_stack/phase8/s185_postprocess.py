"""Apply the S178/S180/S181 rules without S176 after exact S175 rows."""


def apply(ns, base, data_dir, model_dir):
    import ctypes
    import gc
    import importlib.util
    import time
    from pathlib import Path

    diag = {'load_error': None, 'close_error': None, 'clips': {},
            'anchor': 'S172 final collision', 'fallback': 'exact S175 row'}
    ns['_S185_DIAGNOSTICS'] = diag
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
        source = root / 'model/stage2/s185_predict.py'
        spec = importlib.util.spec_from_file_location('_s185_no_bucket', source)
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        if package.Tracker.analyze is not package._s185_analyze:
            raise RuntimeError('S185 extension did not initialize')
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
        record = {'s175_entry': old_entry, 's175_side': old_side,
                  's172_collision': collision, 'error': None, 'rule': None}
        try:
            paths = list(ns['_s008_frame_paths'](Path(data_dir) / 'images' / str(row.ID)))
            numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
            if numbers:
                anchor = ns['_s118_adapter']().position_of(numbers, collision)
                start = time.perf_counter()
                result = tracker.analyze(paths, numbers, anchor)
                record['seconds'] = time.perf_counter() - start
                record['anchor_index'] = int(anchor)
                record['source'] = result.get('s185_source')
                record['s161_reason'] = result.get('s185_s161_reason')
                record['s174_completed'] = result.get('s185_s174_completed')
                record['reason'] = result.get('reason')
                if result.get('s185_side') in ('LEFT', 'RIGHT'):
                    side = result['s185_side']
                    record['rule'] = 'S178 crossing side'
                if result.get('s185_source') == 'S174':
                    position, _ = package.decide(result, 's109')
                    if position is None or not 0 <= int(position) < len(numbers):
                        raise ValueError('invalid S174 crossing position')
                    fps = float(result['fps'])
                    if 1.0 <= fps <= 120.0 and 0 <= position <= anchor:
                        offset = int(round(0.2 * fps))
                        shifted = min(anchor, int(position) + offset)
                        entry = min(int(numbers[shifted]), collision)
                        record['rule'] = 'S180 S174-only shift'
                elif 's185_gap_collision_index' in result:
                    c = int(result['s185_gap_collision_index'])
                    fps = float(result['s185_gap_fps'])
                    if c < 0 or fps not in (10.0, 30.0):
                        raise ValueError('invalid S181 gap anchor')
                    position = max(0, c - int(round(0.85 * fps)))
                    entry = min(int(numbers[position]), collision)
                    record['rule'] = 'S181 gap'
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
