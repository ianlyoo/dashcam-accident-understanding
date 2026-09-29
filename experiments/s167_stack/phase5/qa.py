"""Actual offline CUDA parity and fault QA for the S177 bucket overlay."""
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import types
import zipfile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', S164_WORKERS='2')
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
REPO = H.parents[2]
sys.path.insert(0, str(REPO))
from candidates.s108_export.harness_common import framework_setup, THREAD_ENV
from candidates.s167_stack.monitor_phase3 import monitor_resources, atomic_json
os.environ.update(THREAD_ENV)
D = Path('$DATA_DIR')
W = D/'s167_stack/phase5'
ROOT = W/'candidate'
REQUEST = Path('$GPU_REQUEST_PATH')
PANELS = {
    'public': REPO/'Baseline/sample_evaluation_data/stage2',
    'long': D/'stage2_s118/S127_nexar3',
    'cascade': D/'s144_rr3/refiner/integ_casc',
    'entry': D/'s161_entry_cross/integration',
    'bucket': D/'s176_bucket/qa_inputs',
}
BUCKETS = {'s176_inside_from_start', 's176_inside_first_minus_0p2s'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(label):
    if threading.current_thread() is threading.main_thread() and REQUEST.exists():
        print('GPU_YIELD', label, flush=True)
        raise SystemExit(75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('stage2', 'independent', 'fault'), required=True)
    parser.add_argument('--panel', choices=tuple(PANELS), required=True)
    args = parser.parse_args()
    checkpoint('before framework')
    label = f'{args.mode}_{args.panel}'
    output = W/'qa'
    output.mkdir(parents=True, exist_ok=True)
    resources, stop, monitor = monitor_resources(output, label)
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(1500)
    environment = framework_setup(D/'stage3_s107/vjepa/deps/site', gpu=True)
    import cv2
    import torch
    import pandas as pd
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(
        1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    artifact = load(ROOT/'inference.py', '_s177_exported')
    ns = vars(artifact)
    original_loader = torch.utils.data.DataLoader

    def serial(*a, **kw):
        kw['num_workers'] = 0
        return original_loader(*a, **kw)

    ns['DataLoader'] = serial
    attempts = []

    def audit(event, _args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            attempts.append(event)
            raise RuntimeError('offline QA blocked '+event)

    sys.addaudithook(audit)
    reference_path = (W/'qa/stage2_bucket_baseline.csv' if args.panel == 'bucket'
                      else D/'s167_stack/phase4/qa'/f'stage2_s175_{args.panel}.csv')
    baseline = (None if args.mode == 'stage2' and args.panel == 'bucket'
                else pd.read_csv(reference_path, dtype={'ID': str}))
    inputs = PANELS[args.panel]

    def trim():
        gc.collect()
        torch.cuda.empty_cache()
        ctypes.CDLL('libc.so.6').malloc_trim(0)

    if args.mode == 'stage2':
        adapter = ns['_s118_adapter']()
        prior = adapter.load_package(ROOT, 'model/stage2/s144')
        old_init = prior.Reranker.__init__

        def rerank_init(self, *a, **kw):
            trim()
            return old_init(self, *a, **kw)

        prior.Reranker.__init__ = rerank_init
        visual = ns['_s172_runtime']()
        old_visual_init = visual.VisualCollision.__init__

        def visual_init(self, *a, **kw):
            trim()
            return old_visual_init(self, *a, **kw)

        visual.VisualCollision.__init__ = visual_init
        ns['_s172_runtime'] = lambda: visual
        old_position = visual.VisualCollision.position

        def position(self, *a, **kw):
            checkpoint('before visual clip')
            result = old_position(self, *a, **kw)
            checkpoint('after visual clip')
            return result

        visual.VisualCollision.position = position
        pkg = adapter.load_package(ROOT, 'model/stage2/s161')
        old_analyze = pkg.Tracker.analyze

        def analyze(self, *a, **kw):
            checkpoint('before S175 entry clip')
            result = old_analyze(self, *a, **kw)
            checkpoint('after S175 entry clip')
            return result

        pkg.Tracker.analyze = analyze
        pkg._s174_analyze = analyze
        old_paths = ns['_s008_frame_paths']

        def frame_paths(*a, **kw):
            checkpoint('before frame enumeration')
            result = old_paths(*a, **kw)
            checkpoint('after frame enumeration')
            return result

        ns['_s008_frame_paths'] = frame_paths
        if args.panel == 'bucket':
            base_stage2 = ns['_S177_BASE_STAGE2']

            def capture_base(*a, **kw):
                rows = base_stage2(*a, **kw)
                ns['_S177_CAPTURED_BASE'] = rows.copy()
                return rows

            ns['_S177_BASE_STAGE2'] = capture_base
        checkpoint('before stage2 prediction')
        start = time.perf_counter()
        frame = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        seconds = time.perf_counter()-start
        if args.panel == 'bucket':
            baseline = ns['_S177_CAPTURED_BASE']
            baseline.to_csv(reference_path, index=False, lineterminator='\n')
        assert len(frame) == len(baseline)
        protected = [name for name in baseline if name != 'entry_frame']
        assert frame[protected].reset_index(drop=True).equals(baseline[protected]), \
            'S175 protected output mismatch'
        assert all(frame.entry_frame <= frame.collision_frame)
        diag = ns['_S177_DIAGNOSTICS']
        assert not diag['load_error'] and not diag['close_error']
        assert not any(row.get('error') for row in diag['clips'].values())
        assert all(record['s172_collision'] == int(row.collision_frame)
                   for row, record in zip(baseline.itertuples(index=False),
                                          diag['clips'].values()))
        csv = output/f'{label}.csv'
        frame.to_csv(csv, index=False, lineterminator='\n')
        detail = dict(seconds=seconds, rows=len(frame), csv_sha256=sha(csv),
                      changed_entry_rows=int((frame.entry_frame != baseline.entry_frame).sum()),
                      buckets={key: sum(record.get('bucket') == key for record in diag['clips'].values())
                               for key in sorted(BUCKETS)},
                      s177=diag, s175=ns.get('_S175_DIAGNOSTICS'),
                      s172=ns.get('_S172_DIAGNOSTICS'))
    elif args.mode == 'independent':
        with zipfile.ZipFile(D/'releases/S176_bucket.zip') as archive:
            source = archive.read('model/stage2/s161/predict.py')
        assert hashlib.sha256(source).hexdigest() == sha(ROOT/'model/stage2/s176_predict.py')
        reference = types.ModuleType('_s177_reference_from_zip')
        exec(compile(source, 'S176_bucket.zip/model/stage2/s161/predict.py', 'exec'),
             vars(reference))
        assert reference.Tracker.analyze is reference._s176_analyze
        tracker = reference.Tracker(ROOT/'model/stage2/s161')
        candidate = pd.read_csv(output/f'stage2_{args.panel}.csv', dtype={'ID': str})
        rows = []
        start = time.perf_counter()
        try:
            for row in baseline.itertuples(index=False):
                checkpoint('before independent bucket clip')
                paths = list(ns['_s008_frame_paths'](inputs/'images'/str(row.ID)))
                numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
                anchor = ns['_s118_adapter']().position_of(numbers, int(row.collision_frame))
                result = tracker.analyze(paths, numbers, anchor)
                entry = int(row.entry_frame)
                reason = result.get('reason')
                if reason in BUCKETS:
                    position, _ = reference.decide(result, 's109')
                    assert position is not None and 0 <= int(position) < len(numbers)
                    entry = min(int(numbers[int(position)]), int(row.collision_frame))
                    if reason == 's176_inside_from_start':
                        assert int(position) == 0
                    else:
                        original = result.get('s176_s161_reason')
                        assert original == 'inside_when_first_seen'
                rows.append(dict(ID=str(row.ID), s175_entry=int(row.entry_frame),
                                 collision=int(row.collision_frame), anchor=int(anchor),
                                 reason=reason, expected_entry=entry))
                checkpoint('after independent bucket clip')
        finally:
            tracker.close()
        seconds = time.perf_counter()-start
        assert candidate.ID.tolist() == [row['ID'] for row in rows]
        assert candidate.entry_frame.tolist() == [row['expected_entry'] for row in rows], \
            'Independent S176-on-S172 entry mismatch'
        detail = dict(seconds=seconds, rows=rows,
                      changed_entry_rows=sum(row['expected_entry'] != row['s175_entry'] for row in rows),
                      pinned_s176_sha256=hashlib.sha256(source).hexdigest(),
                      candidate_csv_sha256=sha(output/f'stage2_{args.panel}.csv'))
    else:
        expected = pd.read_csv(output/f'stage2_{args.panel}.csv', dtype={'ID': str})
        ns['_S177_BASE_STAGE2'] = lambda *unused: baseline.copy()
        original_paths = ns['_s008_frame_paths']
        calls = []

        def fail(*unused):
            calls.append('injected')
            raise RuntimeError('S177 injected frame enumeration failure')

        ns['_s008_frame_paths'] = fail
        start = time.perf_counter()
        failed = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        assert calls and failed.equals(baseline), 'Error did not preserve S175 rows'
        assert all('S177 injected' in row['error']
                   for row in ns['_S177_DIAGNOSTICS']['clips'].values())
        ns['_s008_frame_paths'] = original_paths
        trim()
        recovered = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        assert recovered.equals(expected), 'Same-process S177 recovery mismatch'
        seconds = time.perf_counter()-start
        detail = dict(seconds=seconds, fault_calls=len(calls),
                      fallback_exact_s175=True, recovery_exact_s177=True,
                      changed_entry_rows=int((recovered.entry_frame != baseline.entry_frame).sum()))

    stop.set()
    monitor.join(timeout=5)
    signal.alarm(0)
    assert not monitor.is_alive() and not attempts
    assert resources['peak_aggregate_rss_bytes'] <= 4*2**30
    assert resources['peak_processes'] <= 2
    assert resources['minimum_host_free_commit_gib'] >= 12
    assert torch.cuda.max_memory_reserved() <= 2*2**30
    report = dict(passed=True, mode=args.mode, panel=args.panel, detail=detail,
                  resources=resources, cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                  network_attempts=attempts, environment=environment,
                  build_sha256=sha(W/'build.json'), qa_source_sha256=sha(__file__),
                  artifact_sha256=sha(ROOT/'inference.py'),
                  s176_sha256=sha(ROOT/'model/stage2/s176_predict.py'))
    atomic_json(output/f'{label}.json', report)
    print(json.dumps(dict(passed=True, label=label, seconds=detail['seconds'],
                          changed_entry_rows=detail['changed_entry_rows'],
                          peak_rss=resources['peak_aggregate_rss_bytes'],
                          min_free_commit=resources['minimum_host_free_commit_gib'])), flush=True)


if __name__ == '__main__':
    main()
