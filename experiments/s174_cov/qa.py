"""Actual offline exported S171/S174 Stage2 CUDA QA with process checkpoints."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['S164_WORKERS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ctypes,gc,hashlib,importlib.util,json,pathlib,subprocess,sys,threading,time,zipfile
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
D=pathlib.Path('$DATA_DIR');W=D/'s174_cov';ROOT=W/'candidate'
REQUEST=pathlib.Path('$GPU_REQUEST_PATH')
def checkpoint(point):
 if threading.current_thread() is threading.main_thread() and REQUEST.exists():
  (W/'qa_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
  print('GPU_YIELD',point,flush=True);raise SystemExit(75)
def wrap(fn,point):
 def inner(*a,**kw):
  checkpoint(point+' before clip');value=fn(*a,**kw);checkpoint(point+' after clip');return value
 return inner
checkpoint('before QA initialization')
from candidates.s173_wheel_entry.guard import Guard
guard=Guard(W,'qa',7200)
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
os.environ.update(THREAD_ENV)
environment=framework_setup(D/'stage3_s107/vjepa/deps/site',gpu=True)
start=time.perf_counter();install_env=os.environ.copy();installer_source='installed pip'
if importlib.util.find_spec('pip') is None:
 import ensurepip
 wheels=list((pathlib.Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
 assert len(wheels)==1,wheels
 installer_source=str(wheels[0]);install_env['PYTHONPATH']=str(wheels[0])+os.pathsep+install_env.get('PYTHONPATH','')
install=subprocess.run([sys.executable,'-B','-m','pip','install','--no-index','--no-cache-dir','--disable-pip-version-check','-r',str(ROOT/'requirements.txt')],
 env=install_env,capture_output=True,text=True,timeout=60)
install_seconds=time.perf_counter()-start
assert install.returncode==0,install.stdout+install.stderr
import torch,cv2,pandas as pd
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:original_source=z.read('model/stage2/s161/predict.py')
assert (ROOT/'model/stage2/s161/predict.py').read_bytes().startswith(original_source)
mod=load(ROOT/'inference.py','_s174_export_qa');assert '_s174_export_qa' not in sys.modules
loader=torch.utils.data.DataLoader
def serial(*a,**kw):kw['num_workers']=0;return loader(*a,**kw)
mod.DataLoader=serial
adapter=mod._s118_adapter();pkg=adapter.load_package(ROOT,'model/stage2/s161')
extended=pkg.Tracker.analyze;assert extended is pkg._s174_analyze
original=pkg._s174_original_analyze
mod._s008_frame_paths=wrap(mod._s008_frame_paths,'frame enumeration')
for name in ('predict_folder','frame_paths'):
 mod._S2_COLLISION_NAMESPACE[name]=wrap(mod._S2_COLLISION_NAMESPACE[name],'collision '+name)
adapter.load_package=wrap(adapter.load_package,'package transition')
attempts=[]
def audit(event,args):
 if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
  attempts.append(event);raise RuntimeError('offline '+event)
sys.addaudithook(audit)
def trim():gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)
panel=R/'Baseline/sample_evaluation_data/stage2/images'
sources={p.name:p for p in sorted(panel.iterdir())[:3]}
sources.update({p.name:p for p in sorted((D/'s161_entry_cross/integration/images').iterdir())[:3]})
evaluated=[json.loads(x) for x in (W/'evaluation_gpu.jsonl').read_text().splitlines() if x.strip()]
assert len(evaluated)==243
changed=sorted([r for r in evaluated if r['kind']=='ccd' and r['s171_entry']!=r['s174_entry']],key=lambda r:r['id'])
if len(changed)<3:
 extras=[r for r in evaluated if r['kind']=='ccd' and r not in changed]
 changed+=(sorted(extras,key=lambda r:r['id'])[:3-len(changed)])
for r in changed[:3]:sources['ccd_'+r['id']]=D/'ccd/frames'/r['id']
inputs=W/'qa_inputs';images=inputs/'images';images.mkdir(parents=True,exist_ok=True)
for name,src in sources.items():
 dest=images/name
 if not dest.exists():dest.symlink_to(src.resolve(),target_is_directory=True)
phase=W/'qa_phases';phase.mkdir(exist_ok=True)
identity=dict(build=hashlib.sha256((H/'build.json').read_bytes()).hexdigest(),harness=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
cached_resources=[];cached_cuda=[]
def run(folder,key=None):
 checkpoint('before '+str(key or 'live'))
 meta=phase/(str(key)+'.json');data=phase/(str(key)+'.pkl')
 if key and meta.exists() and data.exists():
  saved=json.loads(meta.read_text())
  if saved['identity']==identity:
   cached_resources.append(saved['resources']);cached_cuda.append(saved['cuda_peak_reserved'])
   print('QA_CACHED',key,flush=True)
   return pd.read_pickle(data),saved['seconds'],saved['diagnostics']
 print('QA_START',key or 'live',flush=True)
 trim();start=time.perf_counter();frame=mod.predict_stage2(folder,ROOT/'model/stage2')
 seconds=time.perf_counter()-start;diag=json.loads(json.dumps(mod._S118_LAST_DIAGNOSTICS))
 if key:
  frame.to_pickle(data)
  meta.write_text(json.dumps(dict(identity=identity,seconds=seconds,diagnostics=diag,
                                  resources=dict(guard.record),cuda_peak_reserved=torch.cuda.max_memory_reserved())))
 print('QA_DONE',key or 'live','rows',len(frame),'seconds',round(seconds,3),flush=True)
 checkpoint('completed '+str(key or 'live'))
 if key=='candidate':
  print('QA_RESOURCE_CHECKPOINT paired exports complete; restart before fault/recovery',flush=True)
  raise SystemExit(75)
 return frame,seconds,diag
pkg.Tracker.analyze=original
try:base,base_seconds,base_diag=run(inputs,'baseline')
finally:pkg.Tracker.analyze=extended
assert not base_diag.get('rule_error') and not base_diag.get('rerank_error'),base_diag
actual,candidate_seconds,diag=run(inputs,'candidate')
protected=[c for c in base if c!='entry_frame']
assert actual[protected].equals(base[protected]),'REAL CANDIDATE PROTECTED OUTPUT MISMATCH'
assert (actual.entry_frame<=actual.collision_frame).all(), 'REAL CANDIDATE ENTRY CLAMP MISMATCH'
for row in actual.itertuples(index=False):
 paths=mod._s008_frame_paths(images/str(row.ID));numbers=[int(mod._s008_frame_number(p)) for p in paths]
 assert int(row.entry_frame) in numbers,'REAL CANDIDATE ORIGINAL FRAME ID MISMATCH'
selected=[name for name in sources if name.startswith('ccd_') and
          int(actual.loc[actual.ID==name,'entry_frame'].iloc[0])!=int(base.loc[base.ID==name,'entry_frame'].iloc[0])]
single_id=selected[0] if selected else next(name for name in sources if name.startswith('ccd_'))
single=W/'qa_single';(single/'images').mkdir(parents=True,exist_ok=True)
target=single/'images'/single_id
if not target.exists():target.symlink_to((images/single_id).resolve(),target_is_directory=True)
expected=actual[actual.ID==single_id].reset_index(drop=True)
expected_base=base[base.ID==single_id].reset_index(drop=True)
repeat,repeat_seconds,_=run(single)
assert repeat.reset_index(drop=True).equals(expected),'REAL CANDIDATE SINGLE/BATCH OUTPUT MISMATCH'
original_fallback=pkg._s174_fallback;fault_calls=[0]
def fail(*a,**kw):
 fault_calls[0]+=1;raise RuntimeError('intentional S174 per-clip QA fault')
pkg._s174_fallback=fail
try:failed,_,_=run(single)
finally:pkg._s174_fallback=original_fallback
assert fault_calls[0]>0,'QA fault did not exercise extension'
assert failed.reset_index(drop=True).equals(expected_base),'REAL CANDIDATE FAIL-CLOSED MISMATCH'
recovered,_,_=run(single)
assert recovered.reset_index(drop=True).equals(expected),'REAL CANDIDATE SAME-PROCESS RECOVERY MISMATCH'
forced_calls=[0]
def forced(self,paths,numbers,c):
 forced_calls[0]+=1
 return pkg.Result(reason='crossing',entry_index=min(1,c),bracket=[0,min(1,c)],fps=10.,
                   inside_samples=2,outside_samples=2,track_len=4,collision_index=c)
pkg._s174_fallback=forced
try:forced_out,_,_=run(single)
finally:pkg._s174_fallback=original_fallback
assert forced_calls[0]>0,'QA forced crossing did not exercise extension'
numbers=[int(mod._s008_frame_number(p)) for p in mod._s008_frame_paths(single/'images'/single_id)]
assert int(forced_out.entry_frame.iloc[0])==min(numbers[1],int(expected_base.collision_frame.iloc[0])), 'REAL CANDIDATE FRAME MAPPING MISMATCH'
assert forced_out[protected].reset_index(drop=True).equals(expected_base[protected]), 'REAL CANDIDATE FORCED PROTECTED MISMATCH'
resources=guard.close()
for prior in cached_resources:
 resources['peak_rss']=max(resources['peak_rss'],prior['peak_rss'])
 resources['peak_processes']=max(resources['peak_processes'],prior['peak_processes'])
 if prior['min_free_commit_gib'] is not None:
  resources['min_free_commit_gib']=prior['min_free_commit_gib'] if resources['min_free_commit_gib'] is None else min(resources['min_free_commit_gib'],prior['min_free_commit_gib'])
report=dict(passed=True,environment=environment,rows=len(actual),changed=int((actual.entry_frame!=base.entry_frame).sum()),
 seconds=dict(s171=base_seconds,s174=candidate_seconds,delta=candidate_seconds-base_seconds,single=repeat_seconds),
 protected_columns=protected,only_entry_changed=True,source_frame_ids=True,entry_clamped=True,
 single_case_id=single_id,single_batch_equal=True,per_clip_failure_retains_s171=True,
 recovery_same_process=True,injected_crossing_maps_original_frame_id=True,
 injected_crossing_execution_only=True,dependency_install_seconds=install_seconds,
 dependency_installer_source=installer_source,network_attempts=attempts,
 resources=resources,cuda_peak_reserved=max([torch.cuda.max_memory_reserved(),*cached_cuda]),
 cached_phase_count=len(cached_resources),request_yield_protocol=True,
 local_harness=dict(s164_workers=2,dataloader_workers=0),
 build_sha256=identity['build'],baseline='Exact original S161 method on otherwise identical S171 staged tree',
 natural_errors={k:v.get('error') for k,v in diag['clips'].items() if v.get('error')})
assert not attempts and report['dependency_install_seconds']<600
assert report['resources']['peak_rss']<=4*2**30 and report['cuda_peak_reserved']<=2*2**30
assert report['resources']['min_free_commit_gib'] is not None and report['resources']['min_free_commit_gib']>=12
(H/'qa.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
