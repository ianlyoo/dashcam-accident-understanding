"""Fresh-process WSL/CUDA QA and timing; actual exported stage entrypoints."""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import threading
import signal
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
os.environ['S164_WORKERS'] = '2'  # mandatory preparation resource cap, including parent
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from candidates.s108_export.harness_common import THREAD_ENV, framework_setup, host_memory
os.environ.update(THREAD_ENV)


def monitor_resources(output, label):
    """Sample the actual parent + descendants, not a sum of unrelated peak RSS."""
    state = {'peak_aggregate_rss_bytes': 0, 'peak_processes': 0, 'sample_interval_seconds': 0.25}
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            pending, seen, rss = [os.getpid()], set(), 0
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                    rss += next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:')), 0)
                    seen.add(pid)
                    # Subprocesses may be launched by a thread other than main.
                    for task in Path(f'/proc/{pid}/task').iterdir():
                        pending.extend(int(n) for n in (task / 'children').read_text().split())
                except (FileNotFoundError, ProcessLookupError):
                    continue
            state['peak_aggregate_rss_bytes'] = max(rss, state['peak_aggregate_rss_bytes'])
            state['peak_processes'] = max(len(seen), state['peak_processes'])
            if rss > 3 * 2**30 or len(seen) > 2:
                state['resource_cap_exceeded'] = True
                (output / (label + '_resource_failure.json')).write_text(json.dumps(state, indent=2))
                for pid in seen - {os.getpid()}:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                os._exit(80)
            stop.wait(0.25)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return state, stop, thread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--data', default='$DATA_DIR')
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--panel', default='public', choices=['public', 'long', 'cascade', 'extras'])
    args = ap.parse_args()
    root, data, output = Path(args.candidate), Path(args.data), Path(args.output)
    if (output.parent / 'require_fair_queue').exists() and os.environ.get('S164_FAIR_QA') != '1':
        print('QUEUE_HANDOFF: release legacy batch lease; resume with the fair per-run supervisor', flush=True)
        return 75
    output.mkdir(parents=True, exist_ok=True)
    label = 'stage%d_%s' % (args.stage, args.panel)
    resources, monitor_stop, monitor_thread = monitor_resources(output, label)
    print('framework setup', flush=True)
    environment = framework_setup(data / 'stage3_s107/vjepa/deps/site', gpu=True)
    print('framework ready', flush=True)
    import cv2
    import torch
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(1.75 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    spec = importlib.util.spec_from_file_location('_s167_unregistered_qa', root / 'inference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('artifact loaded', flush=True)
    ns = vars(mod)
    # The local worker brief allows <=2 processes. Same batch tensors/order;
    # serial loading is used for both base and candidate in this constrained QA.
    ns['DataLoader'] = functools.partial(torch.utils.data.DataLoader, num_workers=0)
    original_loader = torch.utils.data.DataLoader
    def loader(*a, **kw):
        kw['num_workers'] = 0
        return original_loader(*a, **kw)
    ns['DataLoader'] = loader
    if args.panel == 'public':
        inputs = REPO / 'Baseline/sample_evaluation_data' / ('stage%d' % args.stage)
    elif args.panel == 'extras':
        assert args.stage == 1
        inputs = data / 's163_s1_cue/qa/extras'
    else:
        assert args.stage == 2
        inputs = data / ('stage2_s118/S127_nexar3' if args.panel == 'long' else 's144_rr3/refiner/integ_casc')
    times = {}
    def instrument(scope, key):
        original = scope[key]
        def wrapped(*a, **kw):
            t = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                rec = times.setdefault(key, {'seconds': 0., 'calls': 0})
                rec['seconds'] += time.perf_counter() - t
                rec['calls'] += 1
        scope[key] = wrapped
    if args.stage == 2:
        for name in ('_S2_CHAMPION_PREDICT_STAGE2', '_s008_predict_sides'):
            instrument(ns, name)
        instrument(ns['_S2_COLLISION_NAMESPACE'], 'extract_folder_features')
    if args.stage == 3:
        instrument(ns, 'extract_video_features')
    network = []
    def audit(event, event_args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            network.append(event)
            raise RuntimeError('offline QA blocked ' + event)
    sys.addaudithook(audit)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    print('inference start', args.stage, args.panel, flush=True)
    frame = ns['predict_stage%d' % args.stage](inputs, root / ('model/stage%d' % args.stage))
    elapsed = time.perf_counter() - t
    csv = frame.to_csv(index=False, lineterminator='\n').encode()
    label = 'stage%d_%s' % (args.stage, args.panel)
    (output / (label + '.csv')).write_bytes(csv)
    report = {'seconds': elapsed, 'rows': len(frame), 'csv_sha256': hashlib.sha256(csv).hexdigest(),
              'times': times, 'host_memory': host_memory(), 'cuda_peak_reserved': torch.cuda.max_memory_reserved(),
              'network_attempts': network, 'environment': environment, 'local_loader_workers': 0,
              'workers': int(os.environ['S164_WORKERS']), 'detector_batch': int(os.environ.get('S164_DETECT_BATCH', '1')),
              's164': ns.get('_S164_LAST_DIAGNOSTICS'), 's118': ns.get('_S118_LAST_DIAGNOSTICS'),
              's108': ns.get('_S108_LAST_DIAGNOSTICS')}
    report.update(resources=resources, s163=ns.get('_S163_DIAGNOSTICS'),
                  inference_sha256=hashlib.sha256((root / 'inference.py').read_bytes()).hexdigest(),
                  qa_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_path=str(inputs))
    (output / (label + '.json')).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str), flush=True)
    assert not network
    assert report['host_memory']['high_water_bytes'] <= 3 * 2**30
    assert report['cuda_peak_reserved'] <= 2 * 2**30
    if args.stage == 1:
        assert report['s163'] is not None
        assert report['s163']['model_error'] is None
        assert all(r['error'] is None for r in report['s163']['videos'].values())
    if report['s164']:
        assert not report['s164'].get('errors'), report['s164']
        assert not report['s164'].get('lazy_scene_error'), report['s164']
        assert report['s164'].get('unused_features') == 0, report['s164']
    if args.stage == 2:
        assert not any((report['s118'] or {}).get(k) for k in ('rule_error', 'rerank_error', 's142_fallbacks', 's142_load_error', 's142_capture_errors'))
    if args.stage == 3:
        assert not any(v.get('fallback') for v in (report['s108'] or {}).values()), report['s108']
    if args.stage == 1:
        reference = data / 's163_s1_cue/qa' / ('candidate_stage1_extra.csv' if args.panel == 'extras' else 'candidate_stage1.csv')
    else:
        reference = data / 's164_runtime/qa_base' / (label + '.csv')
    assert reference.is_file()
    assert csv == reference.read_bytes(), 'Output parity failed: ' + label
    report['reference_csv'] = str(reference)
    report['byte_identical_to_reference'] = True
    monitor_stop.set()
    monitor_thread.join()
    report['validation_ok'] = True
    (output / (label + '.json')).write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    sys.exit(main())
