"""Reproduce phase6 S182 fault/order behavior with captured same-run S177 rows."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['S164_WORKERS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,ctypes,gc,hashlib,importlib.util,json,pathlib,subprocess,sys,threading,time,types,zipfile

HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(REPO))
D=pathlib.Path('$DATA_DIR');PHASE=REPO/'candidates/s167_stack/phase6'
W=D/'s181_fallback/diagnosis';W.mkdir(parents=True,exist_ok=True)
ROOT=D/'s167_stack/phase6/candidate'
INPUT=REPO/'Baseline/sample_evaluation_data/stage2'
REQUEST=pathlib.Path('$GPU_REQUEST_PATH')
parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('batch','single','full_single'),required=True)
parser.add_argument('--id',default='');args=parser.parse_args()
def checkpoint(label):
    if threading.current_thread() is threading.main_thread() and REQUEST.exists():
        (W/'yield.json').write_text(json.dumps(dict(label=label,epoch=time.time())))
        print('GPU_YIELD',label,flush=True);raise SystemExit(75)
def wrap(fn,label):
    def inner(*a,**kw):
        checkpoint(label+' before clip');out=fn(*a,**kw);checkpoint(label+' after clip');return out
    return inner
def trim():gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def records(df):return json.loads(df.to_json(orient='records'))
def keyed(df):return {str(r['ID']):r for r in records(df)}
def save(name,df):df.to_csv(W/(name+'.csv'),index=False,lineterminator='\n')
def diff(a,b):
    x,y=keyed(a),keyed(b)
    return {k:dict(left=x.get(k),right=y.get(k)) for k in sorted(set(x)|set(y)) if x.get(k)!=y.get(k)}

checkpoint('before setup')
from candidates.s173_wheel_entry.guard import Guard
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
guard=Guard(D/'s181_fallback','diagnose_'+args.mode+(('_'+args.id) if args.id else ''),2700)
os.environ.update(THREAD_ENV)
environment=framework_setup(D/'stage3_s107/vjepa/deps/site',gpu=True)
import torch,cv2,pandas as pd
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
artifact=load(ROOT/'inference.py','_s182_diagnosis_'+args.mode+(args.id or ''))
ns=vars(artifact);loader=torch.utils.data.DataLoader
def serial(*a,**kw):kw['num_workers']=0;return loader(*a,**kw)
ns['DataLoader']=serial
adapter=ns['_s118_adapter']()
prior=adapter.load_package(ROOT,'model/stage2/s144')
old_init=prior.Reranker.__init__
def rerank_init(self,*a,**kw):trim();return old_init(self,*a,**kw)
prior.Reranker.__init__=rerank_init
visual=ns['_s172_runtime']();old_visual_init=visual.VisualCollision.__init__
def visual_init(self,*a,**kw):trim();return old_visual_init(self,*a,**kw)
visual.VisualCollision.__init__=visual_init
ns['_s172_runtime']=lambda:visual
visual.VisualCollision.position=wrap(visual.VisualCollision.position,'visual')
pkg=adapter.load_package(ROOT,'model/stage2/s161')
pkg.Tracker.analyze=wrap(pkg.Tracker.analyze,'S177 tracker')
pkg._s174_analyze=pkg.Tracker.analyze
ns['_s008_frame_paths']=wrap(ns['_s008_frame_paths'],'enumeration')
attempts=[]
def audit(event,_args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        attempts.append(event);raise RuntimeError('offline '+event)
sys.addaudithook(audit)
saved=pd.read_csv(D/'s167_stack/phase5/qa/stage2_public.csv',dtype={'ID':str})
base_path=W/'captured_base.csv'
start=time.perf_counter()

if args.mode=='batch':
    source=ns['_S182_BASE_STAGE2'];capture=[]
    def capture_base(*a,**kw):
        out=source(*a,**kw);capture.append(out.copy());save('captured_base',out);return out
    ns['_S182_BASE_STAGE2']=capture_base
    normal=artifact.predict_stage2(INPUT,ROOT/'model/stage2')
    assert len(capture)==1
    live=capture[0];save('normal',normal)
    normal_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
    ns['_S182_BASE_STAGE2']=lambda *a,**kw:live.copy()
    frozen=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('frozen_same_order',frozen)
    frozen_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
    with zipfile.ZipFile(D/'releases/S181_fallback.zip') as archive:
        reference_source=archive.read('model/stage2/s161/predict.py')
    reference=types.ModuleType('_s182_diagnostic_replay')
    exec(compile(reference_source,'S181_fallback.zip/model/stage2/s161/predict.py','exec'),
         vars(reference))
    independent=reference.Tracker(ROOT/'model/stage2/s161');replay_reasons={}
    try:
        for row in live.itertuples(index=False):
            checkpoint('before independent replay clip')
            paths=list(ns['_s008_frame_paths'](INPUT/'images'/str(row.ID)))
            numbers=[int(ns['_s008_frame_number'](p)) for p in paths]
            anchor=adapter.position_of(numbers,int(row.collision_frame))
            replay_reasons[str(row.ID)]=independent.analyze(paths,numbers,anchor).get('reason')
            checkpoint('after independent replay clip')
    finally:independent.close()
    first=str(live.iloc[0]['ID']);old_paths=ns['_s008_frame_paths'];calls=[]
    def fail_first(path):
        if pathlib.Path(path).name==first:
            calls.append(first);raise RuntimeError('S182 injected frame enumeration fault')
        return old_paths(path)
    ns['_s008_frame_paths']=fail_first
    try:
        ns['_S182_BASE_STAGE2']=lambda *a,**kw:saved.copy()
        fault_saved=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('fault_saved_base',fault_saved)
        saved_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
        ns['_S182_BASE_STAGE2']=lambda *a,**kw:live.copy()
        fault_live=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('fault_live_base',fault_live)
        live_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
    finally:ns['_s008_frame_paths']=old_paths
    ns['_S182_BASE_STAGE2']=lambda *a,**kw:live.iloc[::-1].reset_index(drop=True)
    reversed_frame=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('reverse_order',reversed_frame)
    reverse_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
    ns['_S182_BASE_STAGE2']=lambda *a,**kw:live.copy()
    recovery=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('recovery',recovery)
    recovery_diag=json.loads(json.dumps(ns['_S182_DIAGNOSTICS']))
    unaffected=[str(v) for v in live.ID if str(v)!=first]
    summary=dict(mode='batch',seconds=time.perf_counter()-start,
                 saved_vs_live_base=diff(saved,live),normal_vs_frozen=diff(normal,frozen),
                 normal_vs_fault_saved_unaffected={k:v for k,v in diff(normal,fault_saved).items() if k in unaffected},
                 normal_vs_fault_live_unaffected={k:v for k,v in diff(normal,fault_live).items() if k in unaffected},
                 normal_vs_reverse=diff(normal,reversed_frame),
                 normal_vs_recovery=diff(normal,recovery),
                 first_id=first,first_fault_calls=len(calls),
                 first_fault_saved_equals_saved=keyed(fault_saved)[first]==keyed(saved)[first],
                 first_fault_live_equals_live=keyed(fault_live)[first]==keyed(live)[first],
                 replay_reasons=replay_reasons,
                 diagnostics=dict(normal=normal_diag,frozen=frozen_diag,
                                  fault_saved=saved_diag,fault_live=live_diag,
                                  reverse=reverse_diag,recovery=recovery_diag))
    (W/'batch.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('BATCH',json.dumps({k:v for k,v in summary.items() if k!='diagnostics'}),flush=True)
elif args.mode=='single':
    assert args.id and base_path.exists()
    live=pd.read_csv(base_path,dtype={'ID':str})
    assert args.id in live.ID.tolist()
    row=live[live.ID==args.id].copy().reset_index(drop=True)
    ns['_S182_BASE_STAGE2']=lambda *a,**kw:row.copy()
    frame=artifact.predict_stage2(INPUT,ROOT/'model/stage2');save('single_'+args.id,frame)
    assert frame.ID.tolist()==[args.id]
    summary=dict(mode='single',id=args.id,seconds=time.perf_counter()-start,
                 row=records(frame)[0],base=records(row)[0],
                 diagnostics=json.loads(json.dumps(ns['_S182_DIAGNOSTICS'])))
    (W/('single_'+args.id+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('SINGLE',args.id,summary['row'],flush=True)
else:
    assert args.id and args.id in saved.ID.tolist()
    one_root=W/'full_inputs'/args.id
    one_images=one_root/'images';one_images.mkdir(parents=True,exist_ok=True)
    source=(INPUT/'images'/args.id).resolve();link=one_images/args.id
    if not link.exists():link.symlink_to(source,target_is_directory=True)
    assert link.resolve()==source
    frame=artifact.predict_stage2(one_root,ROOT/'model/stage2')
    assert frame.ID.tolist()==[args.id]
    save('full_single_'+args.id,frame)
    summary=dict(mode='full_single',id=args.id,seconds=time.perf_counter()-start,
                 row=records(frame)[0],diagnostics=json.loads(json.dumps(ns['_S182_DIAGNOSTICS'])))
    (W/('full_single_'+args.id+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('FULL_SINGLE',args.id,summary['row'],flush=True)

resources=guard.close()
assert not attempts
assert resources['peak_rss']<=4*2**30 and resources['min_free_commit_gib']>=12
assert torch.cuda.max_memory_reserved()<=2*2**30
print('RESOURCES',json.dumps(dict(peak_rss=resources['peak_rss'],
                                  min_free_commit_gib=resources['min_free_commit_gib'],
                                  cuda_peak_reserved=torch.cuda.max_memory_reserved())),flush=True)
