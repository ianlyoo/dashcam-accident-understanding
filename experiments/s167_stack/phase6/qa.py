"""Exported S182 Stage2 CUDA QA against exact S177 and arm-chain replay."""
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
W = D/'s167_stack/phase6'
ROOT = W/'candidate'
REQUEST = Path('$GPU_REQUEST_PATH')
PANELS = {
    'public': REPO/'Baseline/sample_evaluation_data/stage2',
    'long': D/'stage2_s118/S127_nexar3',
    'cascade': D/'s144_rr3/refiner/integ_casc',
    'entry': D/'s161_entry_cross/integration',
    'bucket': D/'s176_bucket/qa_inputs',
}
SIDE_REASONS = {'crossing', 's176_inside_from_start', 's176_inside_first_minus_0p2s'}
GAP_REASONS = {'short_track', 'not_in_lane_at_collision', 'far_inside_when_first_seen'}


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
    parser.add_argument('--panel', choices=tuple(PANELS), required=True)
    args = parser.parse_args()
    checkpoint('before framework')
    output = W/'qa'
    output.mkdir(parents=True, exist_ok=True)
    resources, stop, monitor = monitor_resources(output, args.panel)
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
    artifact = load(ROOT/'inference.py', '_s182_exported')
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
    reference_path = D/'s167_stack/phase5/qa'/f'stage2_{args.panel}.csv'
    baseline = pd.read_csv(reference_path, dtype={'ID': str})
    inputs = PANELS[args.panel]

    def trim():
        gc.collect()
        torch.cuda.empty_cache()
        ctypes.CDLL('libc.so.6').malloc_trim(0)

    # Preserve the working S164/S172 clipping and resource checkpoints.
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
    checkpoint('before Stage2 prediction')
    start = time.perf_counter()
    frame = artifact.predict_stage2(inputs, ROOT/'model/stage2')
    total_seconds = time.perf_counter()-start
    assert len(frame) == len(baseline)
    protected = [name for name in baseline if name not in ('entry_frame', 'entry_side')]
    assert frame[protected].reset_index(drop=True).equals(baseline[protected]), \
        'S177 protected output mismatch'
    assert all(frame.entry_frame <= frame.collision_frame)
    diag = ns['_S182_DIAGNOSTICS']
    assert not diag['load_error'] and not diag['close_error'], diag
    assert not any(row.get('error') for row in diag['clips'].values()), diag
    assert all(record['s172_collision'] == int(row.collision_frame)
               for row, record in zip(baseline.itertuples(index=False), diag['clips'].values()))
    csv = output/f'stage2_{args.panel}.csv'
    frame.to_csv(csv, index=False, lineterminator='\n')

    # Independent replay uses the exact S181 ZIP source, separate from the
    # candidate postprocessor, and only the final S172 collision anchors.
    with zipfile.ZipFile(D/'releases/S181_fallback.zip') as archive:
        source = archive.read('model/stage2/s161/predict.py')
    assert hashlib.sha256(source).hexdigest() == sha(ROOT/'model/stage2/s181_predict.py')
    reference = types.ModuleType('_s182_reference_from_zip')
    exec(compile(source, 'S181_fallback.zip/model/stage2/s161/predict.py', 'exec'),
         vars(reference))
    assert reference.Tracker.analyze is reference._s181_analyze
    tracker = reference.Tracker(ROOT/'model/stage2/s161')
    replay = []
    replay_start = time.perf_counter()
    try:
        for row in baseline.itertuples(index=False):
            checkpoint('before independent chain clip')
            paths = list(ns['_s008_frame_paths'](inputs/'images'/str(row.ID)))
            numbers = [int(ns['_s008_frame_number'](path)) for path in paths]
            collision = int(row.collision_frame)
            anchor = ns['_s118_adapter']().position_of(numbers, collision)
            result = tracker.analyze(paths, numbers, anchor)
            reason = result.get('reason')
            side = row.entry_side
            entry = int(row.entry_frame)
            if reason in SIDE_REASONS and result.get('side') in ('LEFT', 'RIGHT'):
                side = result['side']
            if reason == 'crossing' and result.get('side') is None:
                prior_position, _ = reference._s180_previous_decide(result, 's109')
                position, _ = reference.decide(result, 's109')
                if position is not None and position != prior_position:
                    assert 0 <= position < len(numbers)
                    entry = min(int(numbers[position]), collision)
            elif reason in GAP_REASONS and 's181_collision_index' in result:
                position, _ = reference.decide(result, 's109')
                assert position is not None and 0 <= position < len(numbers)
                entry = min(int(numbers[position]), collision)
            replay.append(dict(ID=str(row.ID), collision=collision, anchor=int(anchor),
                               reason=reason, expected_entry=entry, expected_side=side,
                               original_entry=int(row.entry_frame), original_side=row.entry_side))
            checkpoint('after independent chain clip')
    finally:
        tracker.close()
    replay_seconds = time.perf_counter()-replay_start
    assert frame.ID.tolist() == [r['ID'] for r in replay]
    assert frame.entry_frame.tolist() == [r['expected_entry'] for r in replay], \
        'Independent S178/S180/S181-on-S172 entry mismatch'
    assert frame.entry_side.tolist() == [r['expected_side'] for r in replay], \
        'Independent S178-on-S172 side mismatch'

    fault = None
    if args.panel == 'public':
        # Inject one clip's frame-path error; every other clip must still pass.
        ns['_S182_BASE_STAGE2'] = lambda *unused: baseline.copy()
        first_id = str(baseline.iloc[0]['ID'])
        fault_source_paths = ns['_s008_frame_paths']
        calls = []

        def fail_first(path):
            if Path(path).name == first_id:
                calls.append(first_id)
                raise RuntimeError('S182 injected frame enumeration fault')
            return fault_source_paths(path)

        ns['_s008_frame_paths'] = fail_first
        failed = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        assert calls and failed.iloc[0].equals(baseline.iloc[0]), 'S177 per-clip fallback mismatch'
        # The saved S177 CSV has object side dtype; exported inference may
        # retain a categorical dtype. Compare submission values, not dtype.
        assert failed.iloc[1:].to_numpy().tolist() == frame.iloc[1:].to_numpy().tolist(), \
            'Unaffected clip value mismatch after fault'
        assert 'S182 injected' in ns['_S182_DIAGNOSTICS']['clips'][first_id]['error']
        ns['_s008_frame_paths'] = fault_source_paths
        trim()
        recovered = artifact.predict_stage2(inputs, ROOT/'model/stage2')
        assert recovered.to_numpy().tolist() == frame.to_numpy().tolist(), \
            'Same-process S182 recovery value mismatch'
        fault = dict(calls=len(calls), fallback_exact_s177=True, recovery_exact_s182=True)

    stop.set()
    monitor.join(timeout=5)
    signal.alarm(0)
    assert not monitor.is_alive() and not attempts
    assert resources['peak_aggregate_rss_bytes'] <= 4*2**30
    assert resources['peak_processes'] <= 2
    assert resources['minimum_host_free_commit_gib'] >= 12
    assert torch.cuda.max_memory_reserved() <= 2*2**30
    report = dict(passed=True, panel=args.panel, rows=len(frame),
                  seconds=total_seconds, replay_seconds=replay_seconds,
                  changed_entry_rows=int((frame.entry_frame != baseline.entry_frame).sum()),
                  changed_side_rows=int((frame.entry_side != baseline.entry_side).sum()),
                  chain_seconds=sum(r.get('seconds', 0) for r in diag['clips'].values()),
                  rules={key: sum(r.get('rule') == key for r in diag['clips'].values())
                         for key in ('S178 side', 'S180 shift', 'S181 gap')},
                  replay=replay, fault=fault, resources=resources,
                  cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                  network_attempts=attempts, environment=environment,
                  build_sha256=sha(W/'build.json'), qa_source_sha256=sha(__file__),
                  artifact_sha256=sha(ROOT/'inference.py'),
                  candidate_csv_sha256=sha(csv))
    atomic_json(output/f'{args.panel}.json', report)
    print(json.dumps(dict(passed=True, panel=args.panel, rows=len(frame),
                          seconds=total_seconds, replay_seconds=replay_seconds,
                          changed_entry_rows=report['changed_entry_rows'],
                          changed_side_rows=report['changed_side_rows'],
                          peak_rss=resources['peak_aggregate_rss_bytes'],
                          min_free_commit=resources['minimum_host_free_commit_gib'])), flush=True)


if __name__ == '__main__':
    main()
