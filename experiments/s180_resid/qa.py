"""Offline exported CUDA paired QA: exact S178 decision versus S180."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['S164_WORKERS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ctypes,gc,hashlib,importlib.util,json,pathlib,subprocess,sys,threading,time

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
D=pathlib.Path('$DATA_DIR');W=D/'s180_resid';ROOT=W/'candidate'
REQUEST=pathlib.Path('$GPU_REQUEST_PATH')

def checkpoint(point):
    if threading.current_thread() is threading.main_thread() and REQUEST.exists():
        (W/'qa_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
        print('GPU_YIELD',point,flush=True);raise SystemExit(75)
def wrap(fn,point):
    def inner(*a,**kw):
        checkpoint(point+' before clip');value=fn(*a,**kw);checkpoint(point+' after clip');return value
    return inner

checkpoint('before CUDA initialization')
from candidates.s173_wheel_entry.guard import Guard
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
guard=Guard(W,'qa',3600);os.environ.update(THREAD_ENV)
environment=framework_setup(D/'stage3_s107/vjepa/deps/site',gpu=True)
start=time.perf_counter();install_env=os.environ.copy()
if importlib.util.find_spec('pip') is None:
    import ensurepip
    wheel=list((pathlib.Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
    assert len(wheel)==1
    install_env['PYTHONPATH']=str(wheel[0])+os.pathsep+install_env.get('PYTHONPATH','')
install=subprocess.run([sys.executable,'-B','-m','pip','install','--no-index','--no-cache-dir',
                        '--disable-pip-version-check','-r',str(ROOT/'requirements.txt')],
                       env=install_env,capture_output=True,text=True,timeout=60)
install_seconds=time.perf_counter()-start
assert install.returncode==0,install.stdout+install.stderr
import torch,cv2,pandas as pd
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
spec=importlib.util.spec_from_file_location('_s180_export_qa',ROOT/'inference.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
loader=torch.utils.data.DataLoader
def serial(*a,**kw):kw['num_workers']=0;return loader(*a,**kw)
mod.DataLoader=serial
adapter=mod._s118_adapter();pkg=adapter.load_package(ROOT,'model/stage2/s161')
assert pkg.Tracker.analyze is pkg._s178_analyze and pkg.decide is not pkg._s180_previous_decide
pkg.Tracker.analyze=wrap(pkg.Tracker.analyze,'tracker')
mod._s008_frame_paths=wrap(mod._s008_frame_paths,'frame enumeration')
for name in ('predict_folder','frame_paths'):
    mod._S2_COLLISION_NAMESPACE[name]=wrap(mod._S2_COLLISION_NAMESPACE[name],'collision '+name)
attempts=[]
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        attempts.append(event);raise RuntimeError('offline '+event)
sys.addaudithook(audit)
def trim():gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)

# Repairs from the independent AI and human panels, and controls for every
# source that must retain S178 exactly. All are public CCD clips.
ids=['ccd_001115','ccd_001273','ccd_001427','ccd_000021','ccd_000091',
     'ccd_000176','ccd_000039','ccd_000131','ccd_000009']
input_dir=W/'qa_inputs';images=input_dir/'images';images.mkdir(parents=True,exist_ok=True)
for ident in ids:
    source=(D/'ccd/frames'/ident.split('_',1)[1]).resolve();assert source.is_dir()
    target=images/ident
    if not target.exists():target.symlink_to(source,target_is_directory=True)
    assert target.resolve()==source

def run(name):
    checkpoint('before '+name);trim();start=time.perf_counter()
    frame=mod.predict_stage2(input_dir,ROOT/'model/stage2').sort_values('ID').reset_index(drop=True)
    seconds=time.perf_counter()-start
    diag=json.loads(json.dumps(mod._S118_LAST_DIAGNOSTICS))
    assert sorted(frame.ID.tolist())==sorted(ids)
    assert not diag.get('rule_error') and not diag.get('rerank_error'),diag
    assert all(not v.get('error') for v in diag['clips'].values()),diag
    checkpoint('after '+name)
    print('QA_PASS',name,round(seconds,2),flush=True)
    return frame,diag,seconds

saved_decide=pkg.decide
pkg.decide=pkg._s180_previous_decide
try:base,base_diag,base_seconds=run('s178')
finally:pkg.decide=saved_decide
candidate,diag,candidate_seconds=run('s180')
protected=[c for c in base if c!='entry_frame']
assert candidate[protected].equals(base[protected]),'REAL CANDIDATE PROTECTED OUTPUT MISMATCH'
assert (candidate.entry_frame<=candidate.collision_frame).all(),'REAL CANDIDATE ENTRY CLAMP MISMATCH'
rows=[]
for before,after in zip(base.itertuples(index=False),candidate.itertuples(index=False)):
    assert before.ID==after.ID
    result=diag['clips'][before.ID]['result'];reason=result.get('reason')
    assert reason==base_diag['clips'][before.ID]['result'].get('reason')
    source='s174_crossing' if reason=='crossing' and result.get('side') is None else reason
    numbers=[int(mod._s008_frame_number(p)) for p in mod._s008_frame_paths(images/before.ID)]
    assert int(after.entry_frame) in numbers and int(before.entry_frame) in numbers
    expected=int(before.entry_frame)
    if source=='s174_crossing':
        index=numbers.index(int(before.entry_frame));collision=numbers.index(int(before.collision_frame))
        expected=numbers[min(collision,index+2)]
    assert int(after.entry_frame)==expected,(before.ID,source,before.entry_frame,after.entry_frame,expected)
    rows.append(dict(ID=before.ID,source=source,before=int(before.entry_frame),
                     after=int(after.entry_frame),collision=int(after.collision_frame)))
assert sum(r['after']!=r['before'] for r in rows)>=3,rows
assert all(r['source']=='s174_crossing' for r in rows if r['after']!=r['before'])

# Inject one clip failure into the unchanged tracker path. Both decisions
# must fall back to the same S109 row and recover in this process.
one='ccd_000021';single=W/'qa_one';one_images=single/'images';one_images.mkdir(parents=True,exist_ok=True)
link=one_images/one
if not link.exists():link.symlink_to((D/'ccd/frames/000021').resolve(),target_is_directory=True)
real=pkg.Tracker.analyze
def fail(*a,**kw):raise RuntimeError('intentional S180 per-clip fault')
pkg.Tracker.analyze=fail
try:failed=mod.predict_stage2(single,ROOT/'model/stage2').reset_index(drop=True)
finally:pkg.Tracker.analyze=real
expected_failure=base[base.ID==one].copy().reset_index(drop=True)
expected_failure.at[0,'entry_frame']=int(mod._S118_LAST_DIAGNOSTICS['clips'][one]['s109'][1])
expected_failure.at[0,'entry_side']=mod._S118_LAST_DIAGNOSTICS['clips'][one]['s109'][2]
assert failed.equals(expected_failure),'REAL CANDIDATE FAIL-CLOSED MISMATCH'
recovered=mod.predict_stage2(single,ROOT/'model/stage2').reset_index(drop=True)
normal=candidate[candidate.ID==one].reset_index(drop=True)
assert recovered.equals(normal),'REAL CANDIDATE SAME-PROCESS RECOVERY MISMATCH'
checkpoint('after fault/recovery')
resources=guard.close()
report=dict(passed=True,environment=environment,rows=rows,clips=len(rows),
            changed=sum(r['after']!=r['before'] for r in rows),only_entry_changed=True,
            entry_clamped=True,source_frame_ids=True,per_clip_failure_retains_s178=True,
            recovery_same_process=True,dependency_install_seconds=install_seconds,
            seconds=dict(base=base_seconds,candidate=candidate_seconds),
            resources=resources,cuda_peak_reserved=torch.cuda.max_memory_reserved(),
            network_attempts=attempts,build_sha256=hashlib.sha256((H/'build.json').read_bytes()).hexdigest())
(H/'qa.json').write_text(json.dumps(report,indent=2)+'\n')
print('QA_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('rows','environment','resources')},default=str),flush=True)
