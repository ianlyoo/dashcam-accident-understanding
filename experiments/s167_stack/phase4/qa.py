"""Fresh offline WSL CUDA acceptance for S175, with fair GPU-request yields."""
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

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',S164_WORKERS='2')
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
REPO=H.parents[2]
sys.path.insert(0,str(REPO))
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
from candidates.s167_stack.monitor_phase3 import monitor_resources,atomic_json
os.environ.update(THREAD_ENV)
D=Path('$DATA_DIR')
W=D/'s167_stack/phase4'
REQUEST=Path('$GPU_REQUEST_PATH')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(label):
    if threading.current_thread() is threading.main_thread() and REQUEST.exists():
        print('GPU_YIELD',label,flush=True)
        raise SystemExit(75)


def trim(torch):
    gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['stage2','independent','fault','stage1','stage3'])
    ap.add_argument('--panel',required=True,choices=['public','long','cascade','entry','extras'])
    ap.add_argument('--tree',choices=['s172','s175'],default='s175')
    a=ap.parse_args()
    checkpoint('before framework')
    label=f'{a.mode}_{a.tree}_{a.panel}'
    output=W/'qa';output.mkdir(exist_ok=True,parents=True)
    resources,stop,thread=monitor_resources(output,label)
    signal.signal(signal.SIGALRM,lambda *unused:os._exit(124));signal.alarm(1500)
    environment=framework_setup(D/'stage3_s107/vjepa/deps/site',gpu=True)
    import cv2
    import torch
    import pandas as pd
    import multiprocessing
    multiprocessing.set_start_method('fork',force=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    root=(D/'s172_vis_coll/revision3/candidate' if a.tree=='s172' else W/'candidate')
    artifact=load(root/'inference.py','_s175_exported_'+a.tree)
    ns=vars(artifact)
    original_loader=torch.utils.data.DataLoader
    def loader(*args,**kwargs):
        kwargs['num_workers']=0
        return original_loader(*args,**kwargs)
    ns['DataLoader']=loader
    attempts=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            attempts.append(event)
            raise RuntimeError('offline QA blocked '+event)
    sys.addaudithook(audit)
    inputs=(REPO/'Baseline/sample_evaluation_data/stage2' if a.panel=='public'
            else D/'stage2_s118/S127_nexar3' if a.panel=='long'
            else D/'s144_rr3/refiner/integ_casc' if a.panel=='cascade'
            else D/'s161_entry_cross/integration' if a.panel=='entry'
            else D/'s163_s1_cue/qa/extras')
    if a.mode=='stage1':
        inputs=REPO/'Baseline/sample_evaluation_data/stage1' if a.panel=='public' else D/'s163_s1_cue/qa/extras'
    if a.mode=='stage3':inputs=D/'validation/public_stage3_10hz'
    # The entry diagnostic produced no changed entry on S172 anchors. Use the
    # same tested cascade panel for fault recovery: two observable S175 changes.
    fault_source='cascade' if a.mode=='fault' and a.panel=='entry' else a.panel
    if a.mode=='fault' and fault_source=='cascade':
        inputs=D/'s144_rr3/refiner/integ_casc'
    reference_csv=(W/'qa'/f'stage2_s172_{fault_source}.csv')
    if a.mode in ('stage2','independent','fault') and a.tree=='s175':
        baseline=pd.read_csv(reference_csv,dtype={'ID':str})
    else:
        baseline=None
    if a.mode in ('stage2','fault','stage3','stage1'):
        # Exact allocator prep from S172's successful exported QA.
        if a.mode in ('stage2','fault'):
            adapter=ns['_s118_adapter']()
            prior_pkg=adapter.load_package(root,'model/stage2/s144')
            old_init=prior_pkg.Reranker.__init__
            def init(self,*args,**kw):
                trim(torch);return old_init(self,*args,**kw)
            prior_pkg.Reranker.__init__=init
            visual=ns['_s172_runtime']()
            old_visual_init=visual.VisualCollision.__init__
            def visual_init(self,*args,**kw):
                trim(torch);return old_visual_init(self,*args,**kw)
            visual.VisualCollision.__init__=visual_init
            ns['_s172_runtime']=lambda:visual
            old_position=visual.VisualCollision.position
            def position(self,*args,**kw):
                checkpoint('before visual clip')
                value=old_position(self,*args,**kw)
                checkpoint('after visual clip')
                return value
            visual.VisualCollision.position=position
            pkg=adapter.load_package(root,'model/stage2/s161')
            old_analyze=pkg.Tracker.analyze
            def analyze(self,*args,**kw):
                checkpoint('before entry clip')
                value=old_analyze(self,*args,**kw)
                checkpoint('after entry clip')
                return value
            pkg.Tracker.analyze=analyze
            if hasattr(pkg,'_s174_analyze'):pkg._s174_analyze=analyze
        print('inference start',label,flush=True)
        checkpoint('before prediction')
        start=time.perf_counter()
        if a.mode=='stage2':
            frame=artifact.predict_stage2(inputs,root/'model/stage2')
        elif a.mode=='stage1':
            frame=artifact.predict_stage1(inputs,root/'model/stage1')
        elif a.mode=='stage3':
            frame=artifact.predict_stage3(inputs,root/'model/stage3')
        else:
            expected=pd.read_csv(W/'qa'/f'stage2_s175_{fault_source}.csv',dtype={'ID':str})
            assert len(expected)==len(baseline)
            original_base=ns['_S175_BASE_STAGE2']
            ns['_S175_BASE_STAGE2']=lambda *unused:baseline.copy()
            failure=[]
            def fail(self,*args,**kw):
                failure.append('injected')
                raise RuntimeError('S175 injected tracker failure')
            pkg.Tracker.analyze=fail;pkg._s174_analyze=fail
            try:
                failed=artifact.predict_stage2(inputs,root/'model/stage2')
                assert failure and failed.equals(baseline),'Tracker failure did not preserve exact S172 rows'
                assert all('S175 injected' in r.get('error','') for r in ns['_S175_DIAGNOSTICS']['clips'].values())
            finally:
                pkg.Tracker.analyze=analyze;pkg._s174_analyze=analyze
            trim(torch)
            frame=artifact.predict_stage2(inputs,root/'model/stage2')
            assert frame.equals(expected),'Same-process S174 recovery differs from healthy S175'
            assert ns['_S175_BASE_STAGE2'] is not original_base
            ns['_S175_BASE_STAGE2']=original_base
        seconds=time.perf_counter()-start
        if a.mode=='stage2':
            assert len(frame)==len(frame.ID.unique())
            if a.tree=='s175':
                protected=[c for c in baseline if c!='entry_frame']
                assert frame[protected].reset_index(drop=True).equals(baseline[protected]),'S172 protected fields changed'
                assert all(frame.entry_frame<=frame.collision_frame)
                assert not ns['_S175_DIAGNOSTICS'].get('load_error'),ns['_S175_DIAGNOSTICS']
                assert not any(r.get('error') for r in ns['_S175_DIAGNOSTICS']['clips'].values())
                assert all(r['s172_collision']==int(row.collision_frame)
                           for row,r in zip(baseline.itertuples(index=False),ns['_S175_DIAGNOSTICS']['clips'].values()))
                assert not ns['_S172_DIAGNOSTICS'].get('load_error'),ns['_S172_DIAGNOSTICS']
            if a.tree=='s172' and a.panel in ('public','long'):
                historical=D/'s172_vis_coll/revision3/qa'/f'{a.panel}.csv'
                assert frame.to_csv(index=False,lineterminator='\n').encode()==historical.read_bytes()
        if a.mode=='stage1':
            expected=D/'s167_stack/phase3/qa'/f'stage1_{a.panel}.csv'
            assert frame.to_csv(index=False,lineterminator='\n').encode()==expected.read_bytes()
        if a.mode=='stage3':
            expected=D/'s167_stack/phase3/qa/stage3_public.csv'
            assert len(frame)==2998 and frame.to_csv(index=False,lineterminator='\n').encode()==expected.read_bytes()
        if a.mode=='fault':
            assert frame.equals(expected) and failed.equals(baseline)
            assert any(frame.entry_frame!=baseline.entry_frame),'Fault panel lacks observable entry difference'
            failed.to_csv(output/f'{label}_injected.csv',index=False,lineterminator='\n')
        csv=output/f'{label}.csv'
        frame.to_csv(csv,index=False,lineterminator='\n')
        detail=dict(seconds=seconds,rows=len(frame),csv_sha256=sha(csv),
                    fault_source_panel=fault_source if a.mode=='fault' else None,
                    changed_entry_rows=None if baseline is None else int((frame.entry_frame!=baseline.entry_frame).sum()),
                    s172=ns.get('_S172_DIAGNOSTICS'),s175=ns.get('_S175_DIAGNOSTICS'))
    else:
        # Separate-process reference: exact pinned S174 module, fresh tracker,
        # S172 rows as collision anchors. No candidate dispatch is called.
        assert a.mode=='independent' and a.tree=='s175'
        ref=load(H/'source/S174_cov/model/stage2/s161/predict.py','_s175_pinned_s174')
        assert ref.Tracker.analyze is ref._s174_analyze
        tracker=ref.Tracker(root/'model/stage2/s161')
        candidate=pd.read_csv(output/f'stage2_s175_{a.panel}.csv',dtype={'ID':str})
        rows=[];start=time.perf_counter()
        try:
            for row in baseline.itertuples(index=False):
                checkpoint('before independent entry clip')
                paths=list(ns['_s008_frame_paths'](inputs/'images'/str(row.ID)))
                numbers=[int(ns['_s008_frame_number'](path)) for path in paths]
                anchor=ns['_s118_adapter']().position_of(numbers,int(row.collision_frame))
                result=tracker.analyze(paths,numbers,anchor)
                pick,_=ref.decide(result,'s109')
                entry=int(row.entry_frame) if pick is None else int(numbers[max(0,min(int(pick),len(numbers)-1))])
                entry=min(entry,int(row.collision_frame))
                rows.append(dict(ID=row.ID,collision=int(row.collision_frame),s172_entry=int(row.entry_frame),
                                 anchor=anchor,reason=result.get('reason'),accepted=pick is not None,expected_entry=entry))
                checkpoint('after independent entry clip')
        finally:tracker.close()
        seconds=time.perf_counter()-start
        assert candidate.entry_frame.tolist()==[r['expected_entry'] for r in rows], 'Independent S174-on-S172 replay mismatch'
        detail=dict(seconds=seconds,rows=rows,pinned_s174_sha256=sha(H/'source/S174_cov/model/stage2/s161/predict.py'),
                    candidate_csv_sha256=sha(output/f'stage2_s175_{a.panel}.csv'),reference_csv_sha256=sha(reference_csv))
    stop.set();thread.join(timeout=5);signal.alarm(0)
    assert not thread.is_alive() and not attempts
    assert resources['peak_aggregate_rss_bytes']<=4*2**30 and resources['peak_processes']<=2
    assert resources['minimum_host_free_commit_gib']>=12
    assert torch.cuda.max_memory_reserved()<=2*2**30
    report=dict(passed=True,mode=a.mode,tree=a.tree,panel=a.panel,
                detail=detail,resources=resources,cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                network_attempts=attempts,environment=environment,build_sha256=sha(W/'build.json'),
                qa_source_sha256=sha(__file__),monitor_sha256=sha(REPO/'candidates/s167_stack/monitor_phase3.py'),
                artifact_sha256=sha(root/'inference.py'),s161_sha256=sha(root/'model/stage2/s161/predict.py'))
    atomic_json(output/f'{label}.json',report)
    print(json.dumps(dict(passed=True,label=label,seconds=detail['seconds'],peak_rss=resources['peak_aggregate_rss_bytes'],
                          min_free_commit=resources['minimum_host_free_commit_gib'],
                          changed_entry_rows=detail.get('changed_entry_rows'))),flush=True)


if __name__=='__main__':main()
