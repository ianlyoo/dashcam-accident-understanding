"""Offline exported S160 Stage2 QA; one process, <=3 GiB RSS, <=2 GiB CUDA."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from candidates.s108_export.harness_common import THREAD_ENV, framework_setup, host_memory
os.environ.update(THREAD_ENV)
os.environ['MALLOC_ARENA_MAX'] = '2'
sys.dont_write_bytecode = True
DATA = Path('$DATA_DIR')
WORK = DATA / 's160_coll_loc'


def package_identity(root):
    names = ['predict.py', 's160_loc.py', 'localizer.json', 'loc_nn.npz', 'S160-NOTICE.txt']
    return {name: hashlib.sha256((root / 'model/stage2/s144' / name).read_bytes()).hexdigest()
            for name in names if (root / 'model/stage2/s144' / name).is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['base', 'candidate'], default='candidate')
    ap.add_argument('--panel', choices=['public', 'long'], default='long')
    ap.add_argument('--fault', choices=['none', 'load', 'runtime', 'invalid'], default='none')
    ap.add_argument('--integration', action='store_true')
    a = ap.parse_args()
    if a.integration:
        for name in ('final_E10b_base_k10.joblib', 'final_E10b_ctx_contact_k20.joblib',
                     'final_E10b_embed_ctx_contact_k10.joblib'):
            assert (DATA / 's144_rr3' / name).is_file(), name
    label = 'integration' if a.integration else f'{a.variant}_{a.panel}_{a.fault}'
    output = WORK / 'qa'
    output.mkdir(exist_ok=True)
    peak = [0]
    def monitor():
        while True:
            rss = host_memory()['rss_bytes']
            peak[0] = max(peak[0], rss)
            if rss > 3 * 1024**3:
                (output / (label + '_resource_error.json')).write_text(json.dumps({'rss_bytes': rss}))
                print('RAM cap exceeded', rss, flush=True)
                os._exit(86)
            time.sleep(.5)
    threading.Thread(target=monitor, daemon=True).start()
    print('framework setup', label, flush=True)
    environment = framework_setup(DATA / 'stage3_s107/vjepa/deps/site', gpu=True)
    import cv2
    import numpy as np
    import torch
    torch.cuda.set_per_process_memory_fraction(1.65 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    # No loader subprocesses in this resource-constrained preparation check.
    original_loader = torch.utils.data.DataLoader
    def loader(*args, **kwargs):
        kwargs['num_workers'] = 0
        return original_loader(*args, **kwargs)
    torch.utils.data.DataLoader = loader
    network = []
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            network.append(event)
            raise RuntimeError('offline QA: ' + event)
    sys.addaudithook(audit)
    if a.integration:
        # Existing integration independently recomputes detector, embedding, HGB,
        # torch network and decoder results; run its actual exported entrypoint.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import integration_s160
        sys.argv = ['integration_s160.py', '--name', 'E10_base', '--model-name', 'E10b',
                    '--stage', 'S160_loc', '--cascade', 'cascade_k2_final', '--loc']
        integration_s160.main()
        report = json.loads((WORK / 'integration_E10_base.json').read_text())
        report.update(environment=environment, network_attempts=network,
                      host_memory=host_memory(), monitored_peak_rss=peak[0],
                      package_identity=package_identity(DATA / 'stage2_s118/S160_loc/candidate'))
        assert not network
        assert report['host_memory']['high_water_bytes'] <= 3 * 1024**3
        (output / 'integration.json').write_text(json.dumps(report, indent=2))
        return
    stage = 'S156_casc' if a.variant == 'base' else 'S160_loc'
    root = DATA / 'stage2_s118' / stage / 'candidate'
    identity_before = package_identity(root)
    spec = importlib.util.spec_from_file_location('_s160_actual_qa', root / 'inference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = vars(mod)
    ns['DataLoader'] = loader
    inputs = REPO / 'Baseline/sample_evaluation_data/stage2' if a.panel == 'public' else DATA / 'stage2_s118/S127_nexar3'
    if a.fault != 'none':
        selected = sorted((inputs / 'images').iterdir())[0]
        inputs = WORK / 'qa_one_public_clip'
        (inputs / 'images').mkdir(parents=True, exist_ok=True)
        link = inputs / 'images' / selected.name
        if not link.exists():
            link.symlink_to(selected, target_is_directory=True)
    adapter = ns['_s118_adapter']()
    pkg = adapter.load_package(root, 'model/stage2/s144')
    timings = []
    row_timings = []
    original_init = pkg.Reranker.__init__
    original_numpy_load = np.load
    def fail_load(path, *args, **kwargs):
        if Path(path).name == 'loc_nn.npz':
            raise OSError('S160 QA injected missing/corrupt localizer weights')
        return original_numpy_load(path, *args, **kwargs)
    def init(self, *args, **kwargs):
        if a.fault == 'load':
            with patch.object(np, 'load', fail_load):
                original_init(self, *args, **kwargs)
            assert self.loc is None and self.loc_error
        else:
            original_init(self, *args, **kwargs)
        if getattr(self, 'loc', None) is not None:
            actual_rows = self.L.rows
            def rows(*args, **kwargs):
                t = time.perf_counter()
                try:
                    return actual_rows(*args, **kwargs)
                finally:
                    row_timings.append(time.perf_counter() - t)
            self.L.rows = rows
            actual = self.L.localize
            def localize(*args, **kwargs):
                t = time.perf_counter()
                try:
                    if a.fault == 'runtime':
                        raise RuntimeError('S160 QA injected localizer failure')
                    if a.fault == 'invalid':
                        return -1, np.asarray([np.nan])
                    return actual(*args, **kwargs)
                finally:
                    timings.append(time.perf_counter() - t)
            self.L.localize = localize
    pkg.Reranker.__init__ = init
    s109 = []
    original_base = ns['_S118_BASE_PREDICT_STAGE2']
    def base(*args, **kwargs):
        frame = original_base(*args, **kwargs)
        s109.append(frame.copy())
        gc.collect()
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        return frame
    ns['_S118_BASE_PREDICT_STAGE2'] = base
    emb = ns['_S2_COLLISION_NAMESPACE']
    hooks = (emb['locate_collision'], emb['predict_folder'])
    torch.cuda.reset_peak_memory_stats()
    print('inference start', label, flush=True)
    start = time.perf_counter()
    out = ns['predict_stage2'](inputs, root / 'model/stage2')
    elapsed = time.perf_counter() - start
    diag = ns['_S118_LAST_DIAGNOSTICS']
    assert hooks == (emb['locate_collision'], emb['predict_folder'])
    assert len(s109) == 1 and diag['single_base_call']
    untouched = ['ID', 'entry_side', 'evasion_space']
    assert out[untouched].equals(s109[0][untouched])
    assert out.entry_frame.tolist() == [min(int(e), int(c)) for e,c in zip(s109[0].entry_frame, out.collision_frame)]
    assert not any(diag.get(k) for k in ('rule_error','rerank_error','s142_fallbacks','s142_load_error','s142_capture_errors')), diag
    clips = diag.get('s142_clips', {})
    for row in clips.values():
        assert not row.get('refine_error'), row
        if a.fault == 'load':
            assert row.get('loc_load_error') and 'loc_to' not in row
        elif a.fault != 'none':
            assert row.get('loc_error') and 'loc_to' not in row
        elif a.variant == 'candidate':
            assert not row.get('loc_load_error') and not row.get('loc_error') and 'loc_to' in row, row
    if a.panel == 'long':
        assert len(clips) == len(out)
    csv = out.to_csv(index=False, lineterminator='\n').encode()
    (output / (label + '.csv')).write_bytes(csv)
    s109[0].to_csv(output / (label + '_s109.csv'), index=False)
    report = dict(validation_ok=True, label=label, seconds=elapsed, rows=len(out), diagnostics=diag,
                  localizer_seconds=timings, network_attempts=network, environment=environment,
                  localizer_rows_seconds=row_timings,
                  localizer_timing_scope='rows feature construction and localize scoring/decode measured separately; excludes package loading',
                  host_memory=host_memory(), monitored_peak_rss=peak[0],
                  cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                  csv_sha256=hashlib.sha256(csv).hexdigest(), loader_workers=0,
                  package_identity=identity_before)
    assert package_identity(root) == identity_before
    assert not network
    assert report['host_memory']['high_water_bytes'] <= 3 * 1024**3
    assert report['cuda_peak_reserved'] <= 2 * 1024**3
    (output / (label + '.json')).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str), flush=True)


if __name__ == '__main__':
    main()
