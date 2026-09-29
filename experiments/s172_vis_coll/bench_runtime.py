"""Exact exported visual arm timing after successful integration QA.

Public S160 OOF priors stand in for deployed priors for timing only. This is
not accuracy validation; all visual computation and frame I/O are unchanged.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from revision_path import revision_work

REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
from candidates.s167_stack.monitor_phase3 import monitor_resources,atomic_json
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
os.environ.update(THREAD_ENV)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def gpu():
    result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=name,memory.used,utilization.gpu,utilization.memory',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    return dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    DATA=Path('$DATA_DIR'); WORK=revision_work(DATA/'s172_vis_coll')
    qa=json.loads((WORK/'qa/long.json').read_text()); assert qa['passed']
    build_sha=hashlib.sha256((WORK/'build.json').read_bytes()).hexdigest()
    assert qa['build_sha256']==build_sha
    resources,stop,thread=monitor_resources(WORK/'qa','benchmark')
    signal.signal(signal.SIGALRM,lambda *a:os._exit(124)); signal.alarm(280)
    environment=framework_setup(DATA/'stage3_s107/vjepa/deps/site',gpu=True)
    import ctypes
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    request=Path('$GPU_REQUEST_PATH')
    if request.exists(): raise SystemExit(75)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    root=WORK/'candidate'; module=load(root/'inference.py','_s172_benchmark'); ns=vars(module)
    base=pd.read_csv(DATA/'s167_stack/phase3/qa/stage2_long.csv',dtype={'ID':str})
    priors={}
    for ident in base.ID:
        with np.load(DATA/'s172_vis_coll/baseline'/f'{ident}.npz') as z:
            priors[ident]={'s172_prior':dict(frames=z['frames'].tolist(),probs=z['probs'].tolist())}
    ns['_S118_LAST_DIAGNOSTICS']={'s142_clips':priors}
    runtime=ns['_s172_runtime']()
    original_position=runtime.VisualCollision.position
    completed=[]
    def checkpoint_position(self,paths,prior):
        if request.exists():
            print(json.dumps(dict(event='benchmark_checkpoint_pause',completed=completed)),flush=True)
            raise SystemExit(75)
        value=original_position(self,paths,prior)
        completed.append(Path(paths[0]).parent.name)
        print(json.dumps(dict(event='benchmark_clip_complete',id=completed[-1])),flush=True)
        return value
    runtime.VisualCollision.position=checkpoint_position
    gc.collect(); torch.cuda.empty_cache(); ctypes.CDLL('libc.so.6').malloc_trim(0)
    before=gpu(); attempts=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            attempts.append(event); raise RuntimeError('Offline benchmark: '+event)
    sys.addaudithook(audit)
    torch.cuda.reset_peak_memory_stats()
    if request.exists(): raise SystemExit(75)
    start=time.perf_counter()
    actual=runtime.apply(ns,base,DATA/'stage2_s118/S127_nexar3',root/'model/stage2/s172')
    elapsed=time.perf_counter()-start
    diag=ns['_S172_DIAGNOSTICS']; assert not diag['load_error'] and not diag.get('close_error'),diag
    assert len(diag['clips'])==3 and all(r.get('coarse_frames',0)>0 and not r.get('error') for r in diag['clips'].values()),diag
    assert diag['budget_triggered'] is None and diag['covered_clips']==3,diag
    assert actual[['ID','evasion_space','entry_side']].equals(base[['ID','evasion_space','entry_side']])
    assert (actual.entry_frame==np.minimum(base.entry_frame,actual.collision_frame)).all()
    assert not attempts
    after=gpu(); stop.set(); thread.join(timeout=2)
    times=[r['seconds'] for r in diag['clips'].values()]
    overhead=elapsed-sum(times)
    projected=137*(sum(times)/len(times)+qa['prior_capture_seconds']/len(times))+overhead
    report=dict(passed=True,purpose='Runtime only; cached public OOF priors, not accuracy validation',
        seconds=elapsed,diagnostics=diag,fixed_overhead_seconds=overhead,
        local_long_mean_seconds=sum(times)/len(times),local_long_max_seconds=max(times),
        conditional_137_clip_seconds=projected,within_480_seconds=projected<=480,
        gpu_before=before,gpu_after=after,resources=resources,
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        offline_attempts=attempts,environment=environment,server_l40s_measured=False,
        build_sha256=build_sha,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        integration_qa_added_seconds=qa['added_seconds'])
    assert atomic_json(WORK/'qa/benchmark.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
