"""Actual S171 CUDA outputs, independent S161 entry replay, fail-closed recovery."""
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
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',S164_WORKERS='2')
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from candidates.s167_stack.monitor_phase3 import monitor_resources, atomic_json
from candidates.s108_export.harness_common import framework_setup,host_memory,THREAD_ENV
os.environ.update(THREAD_ENV)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--stage',type=int,required=True)
    ap.add_argument('--panel',required=True,choices=['public','extras','long','cascade','entry','fault'])
    ap.add_argument('--baseline',action='store_true')
    a=ap.parse_args()
    root,output=Path(a.candidate),Path(a.output)
    data=Path('$DATA_DIR');work=output.parent
    output.mkdir(parents=True,exist_ok=True)
    label=f'stage{a.stage}_{a.panel}'
    resources,stop,thread=monitor_resources(output,label)
    # Bound the QA; stop only this process and the prefetch children it owns.
    def timeout_handler(*args):
        atomic_json(output/(label+'_timeout.json'), {'timeout_seconds':1500})
        for task in Path(f'/proc/{os.getpid()}/task').iterdir():
            for child in (task/'children').read_text().split():
                try: os.kill(int(child),signal.SIGTERM)
                except ProcessLookupError: pass
        os._exit(124)
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(1500)
    print('framework setup',flush=True)
    environment=framework_setup(data/'stage3_s107/vjepa/deps/site',gpu=True)
    import cv2
    import torch
    import pandas as pd
    import multiprocessing
    multiprocessing.set_start_method('fork',force=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    mod=load(root/'inference.py','_s171_actual')
    ns=vars(mod)
    original_loader=torch.utils.data.DataLoader
    def loader(*args,**kw):
        kw['num_workers']=0
        return original_loader(*args,**kw)
    ns['DataLoader']=loader
    print('framework ready; artifact loaded',flush=True)
    if a.stage==3:
        inputs=data/'validation/public_stage3_10hz'
    elif a.panel=='public':
        inputs=REPO/'Baseline/sample_evaluation_data'/f'stage{a.stage}'
    elif a.panel=='extras':
        inputs=data/'s163_s1_cue/qa/extras'
    elif a.panel in ('entry','fault'):
        inputs=data/'s161_entry_cross/integration'
    else:
        inputs=data/('stage2_s118/S127_nexar3' if a.panel=='long' else 's144_rr3/refiner/integ_casc')
    if a.stage==2:
        package=ns['_s118_adapter']().load_package(root,'model/stage2/s144')
        original_init=package.Reranker.__init__
        def trimmed_init(self,*args,**kw):
            gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
            return original_init(self,*args,**kw)
        package.Reranker.__init__=trimmed_init
    attempts=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            attempts.append(event);raise RuntimeError('Offline QA: '+event)
    sys.addaudithook(audit)
    torch.cuda.reset_peak_memory_stats()
    reference_path=(work/'qa_baseline/stage2_entry.csv' if a.panel in ('entry','fault')
                    else data/'s167_stack/phase2b/qa'/f'stage{a.stage}_{a.panel}.csv')
    baseline=None if a.baseline else pd.read_csv(reference_path,dtype={'ID':str})
    fault=None
    print('inference start',label,'baseline' if a.baseline else 'candidate',flush=True)
    t=time.perf_counter()
    if a.panel=='fault':
        adapter=ns['_s118_adapter']()
        entry_pkg=adapter.load_package(root,'model/stage2/s161')
        original_analyze=entry_pkg.Tracker.analyze
        emb=ns['_S2_COLLISION_NAMESPACE']
        hooks=(emb['extract_folder_features'],emb['locate_collision'],emb['predict_folder'],cv2.imread,ns['_S118_BASE_PREDICT_STAGE2'])
        injected=[]
        def fail(self,paths,numbers,collision_index):
            injected.append(dict(ID=paths[0].parent.name,collision=int(numbers[collision_index])))
            raise RuntimeError('S171 injected entry tracker failure')
        entry_pkg.Tracker.analyze=fail
        try:
            failed=ns['predict_stage2'](inputs,root/'model/stage2')
        finally:
            entry_pkg.Tracker.analyze=original_analyze
        assert failed.reset_index(drop=True).equals(baseline), 'Entry error must retain the complete S170 row'
        assert sorted(x['ID'] for x in injected)==sorted(baseline.ID)
        diag=json.loads(json.dumps(ns['_S118_LAST_DIAGNOSTICS']))
        assert all('S171 injected' in d.get('error','') for d in diag['clips'].values())
        fault_runtime=dict(ns['_S164_LAST_DIAGNOSTICS'])
        assert fault_runtime['prefetched']==len(baseline) and fault_runtime['unused_features']==0
        assert not fault_runtime.get('errors') and not fault_runtime.get('lazy_scene_error')
        assert hooks==(emb['extract_folder_features'],emb['locate_collision'],emb['predict_folder'],cv2.imread,ns['_S118_BASE_PREDICT_STAGE2'])
        failed.to_csv(output/'stage2_fault_injected.csv',index=False,lineterminator='\n')
        gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
        frame=ns['predict_stage2'](inputs,root/'model/stage2')
        expected=pd.read_csv(output/'stage2_entry.csv',dtype={'ID':str})
        assert frame.reset_index(drop=True).equals(expected)
        assert (expected.entry_frame!=baseline.entry_frame).any(), 'Need observable entry changes in fault panel'
        assert hooks==(emb['extract_folder_features'],emb['locate_collision'],emb['predict_folder'],cv2.imread,ns['_S118_BASE_PREDICT_STAGE2'])
        fault=dict(injected=injected,all_rows_equal_s170=True,next_call_recovers_s171=True,hooks_restored=True,
                   observable_entry_changes=int((expected.entry_frame!=baseline.entry_frame).sum()),s164=fault_runtime,
                   diagnostics=diag,csv_sha256=sha(output/'stage2_fault_injected.csv'))
    else:
        frame=ns[f'predict_stage{a.stage}'](inputs,root/f'model/stage{a.stage}')
    elapsed=time.perf_counter()-t
    diag=ns.get('_S118_LAST_DIAGNOSTICS',{})
    runtime=ns.get('_S164_LAST_DIAGNOSTICS')
    if runtime:
        assert not runtime.get('errors') and not runtime.get('lazy_scene_error') and runtime['unused_features']==0
    independent=None
    if a.stage==2:
        assert not any(diag.get(k) for k in ('rule_error','rerank_error','s142_fallbacks','s142_load_error','s142_capture_errors'))
        assert runtime['prefetched']==len(frame) and 'lazy_scene_skipped' in runtime
        assert all(not d.get('loc_error') and not d.get('loc_load_error') for d in diag.get('s142_clips',{}).values())
        if not a.baseline:
            protected=[c for c in baseline if c!='entry_frame']
            assert frame[protected].reset_index(drop=True).equals(baseline[protected])
            assert diag.get('track') and all(not d.get('error') for d in diag['clips'].values())
            assert all(d['s109'][0]==int(baseline.loc[baseline.ID==ident,'collision_frame'].iloc[0]) for ident,d in diag['clips'].items())
            if a.panel!='fault':
                gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
                ref=load(work/'reference_s161/model/stage2/s161/predict.py','_s171_pinned_reference')
                tracker=ref.Tracker(root/'model/stage2/s161')
                expected=[];records=[];ref_start=time.perf_counter()
                try:
                    for row in baseline.itertuples(index=False):
                        paths=list(ns['_s008_frame_paths'](inputs/'images'/str(row.ID)))
                        numbers=[int(ns['_s008_frame_number'](p)) for p in paths]
                        anchor=min(range(len(numbers)),key=lambda i:(abs(numbers[i]-int(row.collision_frame)),i))
                        result=tracker.analyze(paths,numbers,anchor)
                        pos,_=ref.decide(result,'s109')
                        source=int(row.entry_frame) if pos is None else int(numbers[pos])
                        entry=min(source,int(row.collision_frame));expected.append(entry)
                        records.append(dict(ID=row.ID,anchor=int(row.collision_frame),s170_entry=int(row.entry_frame),
                            accepted=pos is not None,reason=result.get('reason'),entry_source=source,expected_entry=entry))
                finally:
                    tracker.close()
                assert frame.entry_frame.tolist()==expected,'Independent S161 entry replay differs'
                independent=dict(rows=records,seconds=time.perf_counter()-ref_start,pinned_source_sha256=sha(work/'reference_s161/model/stage2/s161/predict.py'))
            else:
                assert frame.to_csv(index=False,lineterminator='\n')==(output/'stage2_entry.csv').read_text()
    else:
        assert frame.to_csv(index=False,lineterminator='\n').encode()==reference_path.read_bytes()
        if a.stage==1:
            assert ns['_S163_DIAGNOSTICS']['model_error'] is None
            assert all(d['error'] is None for d in ns['_S163_DIAGNOSTICS']['videos'].values())
        else:
            assert len(frame)==2998 and runtime['observer_replay_frames']==2998
            assert all(not d.get('fallback') for d in ns['_S108_LAST_DIAGNOSTICS'].values())
    csv=output/(label+'.csv');frame.to_csv(csv,index=False,lineterminator='\n')
    stop.set();thread.join();signal.alarm(0)
    assert resources['peak_aggregate_rss_bytes']<=4*2**30 and resources['peak_processes']<=2
    assert resources['minimum_host_free_commit_gib']>=12 and not attempts
    assert torch.cuda.max_memory_reserved()<=2*2**30
    report=dict(validation_ok=True,stage=a.stage,panel=a.panel,baseline_generation=a.baseline,
        seconds=elapsed,rows=len(frame),csv_sha256=sha(csv),reference_csv=None if a.baseline else str(reference_path),
        reference_sha256=None if a.baseline else sha(reference_path),resources=resources,host_memory=host_memory(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved(),network_attempts=attempts,environment=environment,
        s164=runtime,s118=diag,independent_entry=independent,fault=fault,
        changed_entry_rows=None if a.baseline or a.stage!=2 else int((frame.entry_frame.reset_index(drop=True)!=baseline.entry_frame).sum()),
        build_sha256=sha(work/'build.json'),qa_source_sha256=sha(__file__),
        resource_monitor_sha256=sha(REPO/'candidates/s167_stack/monitor_phase3.py'),
        source_sha256={p:sha(root/p) for p in ('inference.py','model/stage2/s118/adapter.py','model/stage2/s118/rule.json','model/stage2/s144/predict.py','model/stage3/s141/predict.py')})
    (output/(label+'.json')).write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(dict(validation_ok=True,label=label,seconds=elapsed,rows=len(frame),resources=resources,changed_entry_rows=report['changed_entry_rows'])),flush=True)

if __name__=='__main__':main()
