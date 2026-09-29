"""Bounded CPU-only LK prefetch; original decision functions stay in the parent.

Workers are standalone interpreters, never spawn/fork the CUDA inference harness.
Only per-file arrays cross the boundary. All hooks and caches are call-scoped.
"""
import ast
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import OrderedDict

RUNTIME_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def motion_scope(source, stage):
    """Load exact CPU functions from this artifact, without torch/model imports."""
    raw = Path(source).read_text(encoding='utf-8')
    tree = ast.parse(raw)
    scope = {'__name__': '_s164_cpu', '__file__': str(source)}
    if stage == 2:
        texts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str) and 'def extract_folder_features(' in n.value]
        if len(texts) != 1:
            raise ValueError('expected one embedded Stage2 motion source')
        embedded = ast.parse(texts[0])
        # DataFrame assembly stays in the parent. Avoid importing pandas into
        # every CPU worker solely for the unused predictor annotations.
        embedded.body = [n for n in embedded.body
                         if not (isinstance(n, ast.Import) and any(a.name == 'pandas' for a in n.names))
                         and not (isinstance(n, ast.FunctionDef) and n.name == 'predict_stage2')
                         and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                                  and t.id == '_S2_MOTION_PREDICT_STAGE2' for t in n.targets))]
        exec(compile(embedded, '<s164-original-stage2>', 'exec'), scope)
    else:
        import cv2
        import numpy as np
        scope.update(cv2=cv2, np=np, Path=Path)
        wanted = {'_prepare_gray', '_feature_mask', '_fit_similarity', 'extract_video_features'}
        selected = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted}
        if set(selected) != wanted:
            raise ValueError('missing Stage3 motion functions')
        exec(compile(ast.Module(body=list(selected.values()), type_ignores=[]), str(source), 'exec'), scope)
    return scope


def worker_main():
    # Set before numpy/OpenCV imports; this process never imports torch.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    scope = motion_scope(sys.argv[2], int(sys.argv[3]))
    fn = scope['extract_folder_features' if int(sys.argv[3]) == 2 else 'extract_video_features']
    for line in sys.stdin:
        try:
            request = json.loads(line)
            start = time.perf_counter()
            result = fn(request['path'])
            buf = io.BytesIO()
            np.savez(buf, **result)
            response = {'arrays': base64.b64encode(buf.getvalue()).decode('ascii'),
                        'seconds': time.perf_counter() - start}
            if sys.platform == 'linux':
                import resource
                response['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception as exc:
            response = {'error': type(exc).__name__ + ': ' + str(exc)}
        print(json.dumps(response), flush=True)


def worker_count():
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(7, cpus, int(os.environ.get('S164_WORKERS', '6'))))


def prefetch(source, stage, paths, original, diag):
    """Parent participates, so S164_WORKERS=2 means exactly two processes."""
    import numpy as np
    count = min(worker_count(), len(paths))
    if count <= 1:
        return {}
    workers, cache = [], {}
    limit = max(0, min(256, int(os.environ.get('S164_FEATURE_MB', '128')))) * 1024**2
    used = 0
    start = time.perf_counter()
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        for _ in range(count - 1):
            workers.append(subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                              '--worker', str(source), str(stage)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, text=True, env=env,
                                             creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)))
        for start_index in range(0, len(paths), count):
            if used >= limit:
                diag['feature_budget_reached'] = True
                break
            batch = paths[start_index:start_index + count]
            pending = []
            for proc, path in zip(workers, batch[1:]):
                try:
                    proc.stdin.write(json.dumps({'path': str(path)}) + '\n')
                    proc.stdin.flush()
                    pending.append((proc, path))
                except (OSError, ValueError) as exc:
                    diag.setdefault('errors', []).append(str(exc))
            results = []
            try:
                results.append((batch[0], original(batch[0])))
            except Exception as exc:
                diag.setdefault('errors', []).append(str(exc))
            for proc, path in pending:
                try:
                    response = json.loads(proc.stdout.readline())
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    diag['max_worker_rss_bytes'] = max(diag.get('max_worker_rss_bytes', 0), response.get('peak_rss_bytes', 0))
                    with np.load(io.BytesIO(base64.b64decode(response['arrays'])), allow_pickle=False) as blob:
                        arrays = {k: blob[k] for k in blob.files}
                    results.append((path, arrays))
                except Exception as exc:
                    diag.setdefault('errors', []).append(str(exc))
            for path, arrays in results:
                size = sum(a.nbytes for a in arrays.values())
                if used + size <= limit:
                    cache[str(Path(path).resolve())] = arrays
                    used += size
    except Exception as exc:
        diag.setdefault('errors', []).append(str(exc))
    finally:
        for proc in workers:
            try:
                proc.stdin.close()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()  # only the subprocess created in this call
                proc.wait()
            proc.stdout.close()
        diag.update(prefetch_seconds=time.perf_counter() - start, workers=count,
                    prefetched=len(cache), feature_bytes=used)
    return cache


class DecodeCache:
    """Exact cv2 decode reuse, byte-bounded; returns copies to isolate mutations."""
    def __init__(self, original, limit):
        self.original, self.limit = original, limit
        self.cache = OrderedDict()
        self.size = self.hits = self.misses = 0

    def __call__(self, filename, flags=1, *args, **kwargs):
        if args or kwargs:
            return self.original(filename, flags, *args, **kwargs)
        path = Path(filename)
        try:
            stat = path.stat()
            key = (str(path.resolve()), flags, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return self.original(filename, flags)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        self.misses += 1
        value = self.original(filename, flags)
        if value is not None and value.nbytes <= self.limit:
            while self.cache and self.size + value.nbytes > self.limit:
                self.size -= self.cache.popitem(last=False)[1].nbytes
            self.cache[key] = value.copy()
            self.size += value.nbytes
        return value


def lazy_scene(ns, base, data_dir, model_dir, diag):
    """Run S109's learned scene fallback only for unresolved side/evasion rows.

    S006/S007 overwrite the old collision/entry; S008/S012 overwrite decisive
    scene fields. Only the remaining fields need the expensive ResNet/GRU.
    This wraps the S109 base, below S118/other arms, never their final fields.
    """
    code = getattr(base, '__code__', None)
    if code is None or '_s012_capture_predict_sides' not in code.co_names:
        return base(data_dir, model_dir)
    old_scene, old_sides = ns['_S2_CHAMPION_PREDICT_STAGE2'], ns['_s008_predict_sides']
    sides = {}

    def placeholder(data, model):
        rows = []
        for folder in sorted(p for p in (Path(data) / 'images').iterdir() if p.is_dir()):
            paths = [p for p in ns['_s008_frame_paths'](folder) if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            if paths:
                first = ns['_s008_frame_number'](paths[0])
                rows.append(dict(ID=folder.name, collision_frame=first, entry_frame=first,
                                 evasion_space=0, entry_side='LEFT'))
        return ns['pd'].DataFrame(rows, columns=['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side'])

    def capture_sides(*args, **kwargs):
        value = old_sides(*args, **kwargs)
        sides.update(value)
        return value

    failed = False
    try:
        ns['_S2_CHAMPION_PREDICT_STAGE2'], ns['_s008_predict_sides'] = placeholder, capture_sides
        out = base(data_dir, model_dir)
        evidence = ns.get('_S012_LAST_DIAGNOSTICS', {})
        needed = {str(row.ID) for row in out.itertuples(index=False)
                  if str(row.ID) not in sides or not evidence.get(str(row.ID), {}).get('decisive', False)}
        diag['lazy_scene_required'] = sorted(needed)
        diag['lazy_scene_skipped'] = sorted(set(out.ID.astype(str)) - needed)
        if needed:
            # Clone the original baseline function, filtering only its folder
            # list. Batch size, frame ordering, transforms and weights stay exact.
            tree = ast.parse(Path(ns['__file__']).read_text(encoding='utf-8'))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_stage2')
            node.name = '_s164_scene_subset'
            index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'folders' for t in n.targets))
            node.body.insert(index + 1, ast.parse('folders = [p for p in folders if p.name in _S164_SCENE_IDS]').body[0])
            scope = dict(ns, _S164_SCENE_IDS=needed)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(ns['__file__']), 'exec'), scope)
            start = time.perf_counter()
            reference = scope['_s164_scene_subset'](data_dir, model_dir)
            diag['lazy_scene_seconds'] = time.perf_counter() - start
            by_id = {str(row.ID): row for row in reference.itertuples(index=False)}
            if set(by_id) != needed:
                raise ValueError('scene fallback rows do not match unresolved clips')
            for index, row in out.iterrows():
                ident = str(row['ID'])
                if ident not in needed:
                    continue
                if ident not in sides:
                    out.at[index, 'entry_side'] = by_id[ident].entry_side
                if not evidence.get(ident, {}).get('decisive', False):
                    out.at[index, 'evasion_space'] = int(by_id[ident].evasion_space)
    except Exception as exc:
        diag['lazy_scene_error'] = type(exc).__name__ + ': ' + str(exc)
        failed = True
    finally:
        ns['_S2_CHAMPION_PREDICT_STAGE2'], ns['_s008_predict_sides'] = old_scene, old_sides
    # Outside except: release traceback-held tensors before original fallback.
    return base(data_dir, model_dir) if failed else out


def predict(ns, original_predict, data_dir, model_dir, stage):
    import cv2
    diag = {'stage': stage, 'runtime_sha256': RUNTIME_SHA256}
    ns['_S164_LAST_DIAGNOSTICS'] = diag
    if stage == 2:
        scope = ns['_S2_COLLISION_NAMESPACE']
        key = 'extract_folder_features'
        root = Path(data_dir) / 'images'
        paths = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    else:
        scope, key = ns, 'extract_video_features'
        paths = list(ns['video_paths'](Path(data_dir) / 'videos'))
    original = scope[key]
    default_motion = dict(vars(scope['DEFAULT_MOTION'])) if stage == 2 else None
    # Capture only pure motion arrays. The original hooks/locators still execute
    # in their original sorted order, including S010/S144 feature capture.
    cache = prefetch(ns['__file__'], stage, paths, original, diag)

    def extract(path, *args, **kwargs):
        identity = str(Path(path).resolve())
        # Explicit parameter overrides must use the original implementation.
        compatible = not args and not kwargs
        if stage == 2 and len(args) == 1 and not kwargs:
            compatible = vars(args[0]) == default_motion
        if compatible and identity in cache:
            # S108/S109 wrap the parent's cv2 namespace to observe decoded
            # frames for V-JEPA. Cached motion must still feed that observer;
            # otherwise its alignment guard silently invokes legacy fallback.
            if stage == 3 and scope['cv2'] is not cv2:
                capture = scope['cv2'].VideoCapture(str(path))
                count = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            break
                        count += 1
                finally:
                    capture.release()
                if any(len(v) != count for v in cache[identity].values()):
                    raise ValueError('S164 observer replay/motion alignment differs')
                diag['observer_replay_frames'] = diag.get('observer_replay_frames', 0) + count
            return cache.pop(identity)
        return original(path, *args, **kwargs)

    decode = DecodeCache(cv2.imread, max(0, min(256, int(os.environ.get('S164_DECODE_MB', '96')))) * 1024**2)
    scope[key] = extract
    old_base = ns.get('_S118_BASE_PREDICT_STAGE2')
    if stage == 2:
        cv2.imread = decode
        if old_base is not None and os.environ.get('S164_LAZY_SCENE', '1') == '1':
            ns['_S118_BASE_PREDICT_STAGE2'] = lambda d, m: lazy_scene(ns, old_base, d, m, diag)
    try:
        return original_predict(data_dir, model_dir)
    finally:
        scope[key] = original
        if stage == 2:
            cv2.imread = decode.original
            if old_base is not None:
                ns['_S118_BASE_PREDICT_STAGE2'] = old_base
        diag.update(unused_features=len(cache), decode_hits=decode.hits, decode_misses=decode.misses,
                    decode_bytes=decode.size)
        cache.clear()
        decode.cache.clear()


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--worker':
    worker_main()
