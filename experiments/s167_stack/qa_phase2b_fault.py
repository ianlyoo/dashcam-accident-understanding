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

import gc

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

    ap.add_argument('--reference-s160', action='store_true')

    ap.add_argument('--candidate', required=True)

    ap.add_argument('--output', required=True)

    ap.add_argument('--data', default='$DATA_DIR')

    ap.add_argument('--stage', type=int, required=True)

    ap.add_argument('--panel', default='public', choices=['public', 'long', 'cascade', 'extras', 'fault'])

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

    torch.cuda.set_per_process_memory_fraction(1.65 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)

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

    if args.stage == 3:

        inputs = data / 'validation/public_stage3_10hz'

        if args.panel == 'fault':

            fault_inputs = output.parent / 'fault_inputs/videos'

            fault_inputs.mkdir(parents=True, exist_ok=True)

            for clip in ('OPEN_002', 'OPEN_004'):

                source = next(p for p in (inputs / 'videos').iterdir() if p.stem == clip)

                target = fault_inputs / source.name

                if not target.exists():

                    os.link(source, target)

            inputs = fault_inputs.parent

    elif args.panel == 'public':

        inputs = REPO / 'Baseline/sample_evaluation_data' / ('stage%d' % args.stage)

    elif args.panel == 'extras':

        assert args.stage == 1

        inputs = data / 's163_s1_cue/qa/extras'

    else:

        assert args.stage == 2

        inputs = data / ('stage2_s118/S127_nexar3' if args.panel in ('long', 'fault') else 's144_rr3/refiner/integ_casc')

    if args.stage == 2:

        # Release allocator slack before loading reranker/localizer. Do not wrap

        # _S118_BASE_PREDICT_STAGE2: S164 inspects its code for lazy scene support.

        package = ns['_s118_adapter']().load_package(root, 'model/stage2/s144')

        original_package_init = package.Reranker.__init__

        def trim_init(self, *a, **kw):

            gc.collect()

            import ctypes

            ctypes.CDLL('libc.so.6').malloc_trim(0)

            return original_package_init(self, *a, **kw)

        package.Reranker.__init__ = trim_init

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

    fault_report = None

    if args.panel == 'fault' and args.stage == 3:

        import pandas as pd

        expected = pd.read_csv(data / 's162_accel_dom/S162_acc_r3/qa/candidate_stage3.csv')

        baseline = pd.read_csv(data / 's162_accel_dom/S162_acc_r3/qa/source_stage3.csv')

        ids = [p.stem for p in ns['video_paths'](inputs / 'videos')]

        assert ids == ['OPEN_002', 'OPEN_004']

        expected = expected[expected.ID.isin(ids)].reset_index(drop=True)

        baseline = baseline[baseline.ID.isin(ids)].reset_index(drop=True)

        fault_clip = 'OPEN_004'  # 58 changed rows: fallback must observably restore S156.

        package = ns['_s118_adapter']().load_package(root, 'model/stage3/s141')

        original_decode = package.decode

        s109 = ns['_s109_adapter']()

        originals = dict(extract=ns['extract_video_features'], matrix=ns['_s019_feature_matrix'],

                         cv2=ns['cv2'], combine=s109.combine_probabilities)

        calls = []

        def fail_decode(logits, **kwargs):

            clip = ids[len(calls)]

            calls.append(dict(clip=clip, rows=len(logits), injected=clip == fault_clip))

            if clip == fault_clip:

                logits[:] = 0

                raise RuntimeError('S169 injected S162 decoder failure after partial mutation')

            return original_decode(logits, **kwargs)

        package.decode = fail_decode

        try:

            faulted = ns['predict_stage3'](inputs, root / 'model/stage3')

        finally:

            package.decode = original_decode

        assert len(calls) == 2 and sum(c['injected'] for c in calls) == 1

        want = expected.copy()

        want.loc[want.ID == fault_clip, 'accel_label'] = baseline.loc[baseline.ID == fault_clip, 'accel_label']

        assert faulted.reset_index(drop=True).equals(want)

        assert (faulted.accel_label != expected.accel_label).sum() == 58

        assert ns['extract_video_features'] is originals['extract']

        assert ns['_s019_feature_matrix'] is originals['matrix']

        assert ns['cv2'] is originals['cv2'] and s109.combine_probabilities is originals['combine']

        fault_diag = dict(ns['_S164_LAST_DIAGNOSTICS'])

        assert fault_diag['prefetched'] == 2 and fault_diag['unused_features'] == 0

        assert fault_diag['observer_replay_frames'] == len(faulted)

        assert not fault_diag.get('errors')

        assert all(not v.get('fallback') for v in ns['_S108_LAST_DIAGNOSTICS'].values())

        assert all(v['regressed'] for v in ns['_S118_STAGE3_DIAGNOSTICS']['videos'].values())

        faulted_csv = faulted.to_csv(index=False, lineterminator='\n').encode()

        (output / 'stage3_fault_injected.csv').write_bytes(faulted_csv)

        gc.collect()

        torch.cuda.empty_cache()

        recovery_start = time.perf_counter()

        frame = ns['predict_stage3'](inputs, root / 'model/stage3')

        recovery_seconds = time.perf_counter() - recovery_start

        assert frame.reset_index(drop=True).equals(expected)

        assert ns['extract_video_features'] is originals['extract']

        assert ns['_s019_feature_matrix'] is originals['matrix']

        assert ns['cv2'] is originals['cv2'] and s109.combine_probabilities is originals['combine']

        assert ns['_S164_LAST_DIAGNOSTICS']['prefetched'] == 2

        assert ns['_S164_LAST_DIAGNOSTICS']['observer_replay_frames'] == len(frame)

        fault_report = dict(calls=calls, clip=fault_clip, changed_back_to_s156_rows=58,

            other_clip_retains_s162=True, steering_preserved=True, stopped_preserved=True,

            partial_mutation=True, next_call_recovers_s162=True, all_hooks_restored=True,

            fault_s164=fault_diag, fault_csv_sha256=hashlib.sha256(faulted_csv).hexdigest(),

            recovery_seconds=recovery_seconds)

    elif args.stage == 2 and args.panel == 'fault':

        import pandas as pd

        expected = pd.read_csv(data / 's160_coll_loc/qa/candidate_long_none.csv', dtype={'ID':str})

        baseline = pd.read_csv(data / 's164_runtime/qa_base/stage2_long.csv', dtype={'ID':str})

        fault_clip = '00060'  # second clip is worker-prefetched; inject 019 as well for observable change.

        fault_ids = {'00019', '00060'}

        package = ns['_s118_adapter']().load_package(root, 'model/stage2/s144')

        original_init, original_locate = package.Reranker.__init__, package.Reranker.locate

        emb = ns['_S2_COLLISION_NAMESPACE']

        original_hooks = (emb['extract_folder_features'], emb['locate_collision'], emb['predict_folder'],

                          cv2.imread, ns['_S118_BASE_PREDICT_STAGE2'])

        active, injected = [None], []

        def init(self, *a, **kw):

            original_init(self, *a, **kw)

            assert self.loc is not None and self.loc_error is None

            actual = self.L.localize

            def fail(*a, **kw):

                if active[0] in fault_ids:

                    injected.append(active[0])

                    raise RuntimeError('S170 injected localizer failure')

                return actual(*a, **kw)

            self.L.localize = fail

        def locate(self, features, paths):

            active[0] = paths[0].parent.name

            try:

                return original_locate(self, features, paths)

            finally:

                active[0] = None

        package.Reranker.__init__, package.Reranker.locate = init, locate

        try:

            faulted = ns['predict_stage2'](inputs, root / 'model/stage2')

        finally:

            package.Reranker.__init__, package.Reranker.locate = original_init, original_locate

        def normalized(f):

            f = f.copy(); f.ID = f.ID.astype(str)

            return f.reset_index(drop=True)

        want = expected.copy()

        for ident in fault_ids:

            want.loc[want.ID == ident, :] = baseline.loc[baseline.ID == ident, :].values

        assert normalized(faulted).equals(want), 'Localizer fail-closed parity'

        assert sorted(injected) == sorted(fault_ids)

        assert int(faulted.loc[faulted.ID == '00019', 'collision_frame'].iloc[0]) == 612

        assert int(expected.loc[expected.ID == '00019', 'collision_frame'].iloc[0]) == 609

        assert int(faulted.loc[faulted.ID == '00076', 'collision_frame'].iloc[0]) == 597

        fault_diag = dict(ns['_S164_LAST_DIAGNOSTICS'])

        fault_s118 = dict(ns['_S118_LAST_DIAGNOSTICS'])

        assert fault_diag['prefetched'] == 3 and fault_diag['unused_features'] == 0

        assert not fault_diag.get('errors') and not fault_diag.get('lazy_scene_error')

        for ident in fault_ids:

            d = fault_s118['s142_clips'][ident]

            assert d.get('loc_error') and 'loc_to' not in d

        def restored():

            return original_hooks == (emb['extract_folder_features'], emb['locate_collision'], emb['predict_folder'],

                                      cv2.imread, ns['_S118_BASE_PREDICT_STAGE2'])

        assert restored()

        fault_bytes = faulted.to_csv(index=False, lineterminator='\n').encode()

        (output / 'stage2_fault_injected.csv').write_bytes(fault_bytes)

        gc.collect(); torch.cuda.empty_cache()

        # QA-only: release glibc slack before spawning the recovery prefetch worker.

        import ctypes

        ctypes.CDLL('libc.so.6').malloc_trim(0)

        frame = ns['predict_stage2'](inputs, root / 'model/stage2')

        assert normalized(frame).equals(expected) and restored()

        fault_report = dict(injected_ids=injected, prefetched_fault_clip=fault_clip,

            observable_fallback_clip='00019', s156_collision=612, s160_collision=609,

            unaffected_clip='00076', unaffected_collision=597, next_call_recovers_s160=True,

            all_hooks_restored=True, fault_s164=fault_diag, fault_s118=fault_s118,

            fault_csv_sha256=hashlib.sha256(fault_bytes).hexdigest())

    else:

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

    report.update(reference_s160=args.reference_s160,

                  s144_source_sha256=hashlib.sha256((root / 'model/stage2/s144/predict.py').read_bytes()).hexdigest(),

                  build_sha256=hashlib.sha256((output.parent / 'build.json').read_bytes()).hexdigest(),

                  resources=resources, s163=ns.get('_S163_DIAGNOSTICS'),

                  fault_injection=fault_report,

                  s162_source_sha256=hashlib.sha256((root / 'model/stage3/s141/predict.py').read_bytes()).hexdigest(),

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

    if args.stage == 2:

        import pandas as pd

        baseline = pd.read_csv(data / 's164_runtime/qa_base' / ('stage2_' + ('long' if args.panel == 'fault' else args.panel) + '.csv'), dtype={'ID':str})

        actual = frame.copy(); actual.ID = actual.ID.astype(str)

        protected = ['ID','entry_side','evasion_space']

        assert actual[protected].reset_index(drop=True).equals(baseline[protected])

        assert all(int(e) <= int(c) for e,c in zip(actual.entry_frame, actual.collision_frame))

        report['protected_fields_equal_s156'] = True

        clips = report['s118'].get('s142_clips', {})

        if args.panel != 'public': assert len(clips) == len(frame)

        for detail in clips.values():

            assert not any(detail.get(k) for k in ('refine_error','loc_error','loc_load_error'))

            assert 'loc_to' in detail

            if not args.reference_s160:

                assert detail['detector_batch'] == int(os.environ.get('S164_DETECT_BATCH','1'))

        if not args.reference_s160:

            assert report['s164']['prefetched'] == len(frame)

            assert 'lazy_scene_skipped' in report['s164']

    if args.stage == 3:

        assert not any(v.get('fallback') for v in (report['s108'] or {}).values()), report['s108']

        import pandas as pd

        baseline = pd.read_csv(data / 's162_accel_dom/S162_acc_r3/qa/source_stage3.csv')

        baseline = baseline[baseline.ID.isin(frame.ID.unique())].reset_index(drop=True)

        assert frame.drop(columns='accel_label').reset_index(drop=True).equals(baseline.drop(columns='accel_label'))

        assert (frame.accel_label == 'STOPPED').reset_index(drop=True).equals(baseline.accel_label == 'STOPPED')

        report['changed_accel_rows'] = int((frame.accel_label.reset_index(drop=True) != baseline.accel_label).sum())

        report['steering_equal_s156'] = report['stopped_equal_s156'] = True

        if args.panel == 'public':

            assert len(frame) == 2998 and report['changed_accel_rows'] == 68

            assert report['s164']['prefetched'] == 5

        assert report['s164']['observer_replay_frames'] == len(frame)

        diag = ns['_S118_STAGE3_DIAGNOSTICS']

        assert diag['rule_error'] is None and all(v['regressed'] for v in diag['videos'].values())

    if args.stage == 1:

        reference = data / 's163_s1_cue/qa' / ('candidate_stage1_extra.csv' if args.panel == 'extras' else 'candidate_stage1.csv')

    elif args.stage == 3:

        reference = data / 's162_accel_dom/S162_acc_r3/qa/candidate_stage3.csv'

    else:

        if args.reference_s160:

            assert args.stage == 2 and args.panel == 'cascade'

            reference = output / (label + '.csv')

            report['reference_generation_only'] = True

        elif args.panel == 'cascade':

            reference = output.parent / 'qa_reference/stage2_cascade.csv'

        else:

            reference = data / 's160_coll_loc/qa' / ('candidate_' + ('long' if args.panel == 'fault' else args.panel) + '_none.csv')

    assert reference.is_file()

    reference_bytes = reference.read_bytes()

    if args.panel == 'fault' and args.stage == 3:

        expected = pd.read_csv(reference)

        reference_bytes = expected[expected.ID.isin(frame.ID.unique())].to_csv(index=False, lineterminator='\n').encode()

        report['reference_subset_ids'] = sorted(frame.ID.unique().tolist())

    assert csv == reference_bytes, 'Output parity failed: ' + label

    report['reference_csv'] = str(reference)

    report['byte_identical_to_reference'] = True

    monitor_stop.set()

    monitor_thread.join()

    report['validation_ok'] = True

    (output / (label + '.json')).write_text(json.dumps(report, indent=2, default=str))





if __name__ == '__main__':

    sys.exit(main())

