"""Real offline WSL CUDA exported Stage2 plus fault recovery and field isolation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',S164_WORKERS='2')
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
from candidates.s167_stack.monitor_phase3 import monitor_resources,atomic_json
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
os.environ.update(THREAD_ENV)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',choices=['public','long'],required=True)
    args=ap.parse_args()
    DATA=Path('$DATA_DIR'); WORK=DATA/'s172_vis_coll'
    root=WORK/'candidate'; output=WORK/'qa'; output.mkdir(exist_ok=True)
    resources,stop,thread=monitor_resources(output,args.panel)
    signal.signal(signal.SIGALRM,lambda *args:os._exit(124)); signal.alarm(1400)
    environment=framework_setup(DATA/'stage3_s107/vjepa/deps/site',gpu=True)
    import cv2
    import torch
    import pandas as pd
    import multiprocessing
    multiprocessing.set_start_method('fork',force=True)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    module=load(root/'inference.py','_s172_actual'); ns=vars(module)
    original_loader=torch.utils.data.DataLoader
    def loader(*a,**kw):
        kw['num_workers']=0; return original_loader(*a,**kw)
    ns['DataLoader']=loader
    original=ns['_S172_BASE_STAGE2']; captured={}
    def capture(data,model):
        t=time.perf_counter(); base=original(data,model)
        captured.update(base=base.copy(),seconds=time.perf_counter()-t)
        return base
    ns['_S172_BASE_STAGE2']=capture
    attempts=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            attempts.append(event); raise RuntimeError('Offline QA: '+event)
    sys.addaudithook(audit)
    inputs=(REPO/'Baseline/sample_evaluation_data/stage2' if args.panel=='public'
        else DATA/'stage2_s118/S127_nexar3')
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); actual=module.predict_stage2(inputs,root/'model/stage2'); elapsed=time.perf_counter()-start
    base=captured['base']; diag=json.loads(json.dumps(ns['_S172_DIAGNOSTICS']))
    capture_seconds=sum(r.get('s172_capture_seconds',0) for r in ns.get('_S118_LAST_DIAGNOSTICS',{}).get('s142_clips',{}).values())
    expected=pd.read_csv(DATA/f's167_stack/phase3/qa/stage2_{args.panel}.csv',dtype={'ID':str})
    assert base.reset_index(drop=True).equals(expected.reset_index(drop=True)),'Fresh base differs from exact S171 QA'
    protected=[c for c in base if c not in ('collision_frame','entry_frame')]
    assert actual[protected].equals(base[protected])
    assert (actual.entry_frame==np_minimum(base.entry_frame,actual.collision_frame)).all()
    assert not diag['load_error'],diag
    assert not diag.get('close_error'),diag
    assert all(not r.get('error') for r in diag['clips'].values()),diag
    if args.panel=='long':
        assert all(r.get('coarse_frames',0)>0 for r in diag['clips'].values()),'No visual inference'
    actual.to_csv(output/f'{args.panel}.csv',index=False,lineterminator='\n')
    faults=[]; stage3=None
    if args.panel=='long':
        # Keep the observed exact-S171 base rows and run the actual exported
        # wrapper for each injected fault; no duplicated expensive S171 work.
        ns['_S172_BASE_STAGE2']=lambda *a:base.copy()
        package=ns['_s172_runtime']()
        ns['_s172_runtime']=lambda:package
        original_init=package.VisualCollision.__init__
        original_position=package.VisualCollision.position
        for fault in ('load','runtime','invalid'):
            injected=[]
            def fail_init(self,*a):
                injected.append('load'); raise RuntimeError('S172 injected load failure')
            def fail_position(self,*a):
                injected.append(fault)
                if fault=='runtime': raise RuntimeError('S172 injected runtime failure')
                return float('nan'),{}
            if fault=='load': package.VisualCollision.__init__=fail_init
            else: package.VisualCollision.position=fail_position
            try: failed=module.predict_stage2(inputs,root/'model/stage2')
            finally:
                package.VisualCollision.__init__=original_init
                package.VisualCollision.position=original_position
            assert injected and failed.equals(base),fault
            faults.append(dict(fault=fault,injected=len(injected),exact_s171=True))
            gc.collect(); torch.cuda.empty_cache()
        recovered=module.predict_stage2(inputs,root/'model/stage2')
        assert recovered.equals(actual),'Same-process recovery mismatch'
        # Check the next real stage after visual model construction and cleanup.
        import ctypes
        gc.collect(); torch.cuda.empty_cache(); ctypes.CDLL('libc.so.6').malloc_trim(0)
        t3=time.perf_counter()
        third=module.predict_stage3(DATA/'validation/public_stage3_10hz',root/'model/stage3')
        expected_third=pd.read_csv(DATA/'s167_stack/phase3/qa/stage3_public.csv',dtype={'ID':str})
        assert third.reset_index(drop=True).equals(expected_third.reset_index(drop=True)), 'Stage3 changed after S172'
        stage3=dict(rows=len(third),seconds=time.perf_counter()-t3,exact_s171=True)
    assert not attempts
    stop.set(); thread.join(timeout=2)
    report=dict(passed=True,panel=args.panel,clips=len(actual),seconds=elapsed,base_seconds=captured['seconds'],
        added_seconds=elapsed-captured['seconds']+capture_seconds,prior_capture_seconds=capture_seconds,
        diagnostics=diag,faults=faults,offline_attempts=attempts,
        protected_fields=protected,entry_coupling='min(exact S171 entry, S172 collision)',
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,resources=resources,
        changed_collisions=int((actual.collision_frame!=base.collision_frame).sum()),stage3_after_stage2=stage3,
        build_sha256=hashlib.sha256((WORK/'build.json').read_bytes()).hexdigest(),
        qa_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=environment)
    atomic_json(output/f'{args.panel}.json',report)
    print(json.dumps(report,default=str),flush=True)

def np_minimum(a,b):
    import numpy as np
    return np.minimum(a,b)
if __name__=='__main__': main()
