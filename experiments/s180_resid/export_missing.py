"""Export exact S178 entry rows for the three uncovered public DKB clips."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['S164_WORKERS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ctypes,hashlib,importlib.util,json,pathlib,subprocess,sys,threading,time

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
D=pathlib.Path('$DATA_DIR');W=D/'s180_resid';ROOT=D/'s178_side/candidate'
REQUEST=pathlib.Path('$GPU_REQUEST_PATH')
def checkpoint(point):
    if threading.current_thread() is threading.main_thread() and REQUEST.exists():
        (W/'export_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
        print('GPU_YIELD',point,flush=True);raise SystemExit(75)
def wrap(fn,point):
    def inner(*args,**kwargs):
        checkpoint(point+' before clip')
        value=fn(*args,**kwargs)
        checkpoint(point+' after clip')
        return value
    return inner
checkpoint('before CUDA initialization')
from candidates.s173_wheel_entry.guard import Guard
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
guard=Guard(W,'export',1800);os.environ.update(THREAD_ENV)
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
assert install.returncode==0,install.stdout+install.stderr
install_seconds=time.perf_counter()-start
import torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
spec=importlib.util.spec_from_file_location('_s180_exact_s178',ROOT/'inference.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
loader=torch.utils.data.DataLoader
def serial(*a,**kw):kw['num_workers']=0;return loader(*a,**kw)
mod.DataLoader=serial
adapter=mod._s118_adapter();pkg=adapter.load_package(ROOT,'model/stage2/s161')
assert pkg.Tracker.analyze is pkg._s178_analyze
pkg.Tracker.analyze=wrap(pkg.Tracker.analyze,'tracker')
mod._s008_frame_paths=wrap(mod._s008_frame_paths,'enumeration')
for name in ('predict_folder','frame_paths'):
    mod._S2_COLLISION_NAMESPACE[name]=wrap(mod._S2_COLLISION_NAMESPACE[name],'collision '+name)
attempts=[]
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        attempts.append(event);raise RuntimeError('offline '+event)
sys.addaudithook(audit)
input_dir=W/'missing_inputs';images=input_dir/'images';images.mkdir(parents=True,exist_ok=True)
ids=('000016','000039','000044')
for ident in ids:
    src=(D/'ccd/frames'/ident).resolve();assert src.is_dir()
    dest=images/('ccd_'+ident)
    if not dest.exists():dest.symlink_to(src,target_is_directory=True)
    assert dest.resolve()==src
checkpoint('before export')
start=time.perf_counter();frame=mod.predict_stage2(input_dir,ROOT/'model/stage2')
seconds=time.perf_counter()-start
diag=json.loads(json.dumps(mod._S118_LAST_DIAGNOSTICS))
assert sorted(frame.ID.tolist())==['ccd_'+ident for ident in ids]
assert not diag.get('rule_error') and not diag.get('rerank_error')
rows=[]
for row in frame.itertuples(index=False):
    ident=row.ID;record=diag['clips'][ident];assert not record.get('error'),(ident,record)
    numbers=[int(mod._s008_frame_number(p)) for p in mod._s008_frame_paths(images/ident)]
    assert int(row.entry_frame) in numbers and int(row.collision_frame) in numbers
    result=record.get('result') or {}
    rows.append(dict(key=ident.replace('_','/',1),entry_frame=int(row.entry_frame),
                     collision_frame=int(row.collision_frame),reason=result.get('reason'),
                     side=result.get('side'),s109_entry=record['s109'][1],
                     s109_side=record['s109'][2],fps=10.))
assert all(r['entry_frame']<=r['collision_frame'] for r in rows)
(H/'export_missing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
resources=guard.close()
report=dict(passed=True,rows=len(rows),seconds=seconds,install_seconds=install_seconds,
            cuda_peak_reserved=torch.cuda.max_memory_reserved(),resources=resources,
            network_attempts=attempts,environment=environment,
            s178_release_sha256=None)
with (D/'releases/S178_side.zip').open('rb') as source:
    report['s178_release_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
assert report['s178_release_sha256']=='bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047'
assert not attempts and install_seconds<600 and resources['peak_rss']<=4*2**30
assert report['cuda_peak_reserved']<=2*2**30 and resources['min_free_commit_gib']>=12
(H/'export_missing.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(rows=rows,seconds=seconds,resources=resources,
                      cuda_peak_reserved=report['cuda_peak_reserved'])),flush=True)
