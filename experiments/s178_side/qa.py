"""Offline exported CUDA QA and full available human side-label evaluation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['S164_WORKERS']='2';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import csv,ctypes,gc,hashlib,importlib.util,json,pathlib,subprocess,sys,threading,time,zipfile

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
D=pathlib.Path('$DATA_DIR');W=D/'s178_side';ROOT=W/'candidate'
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
from candidates.s108_export.harness_common import framework_setup,THREAD_ENV
guard=Guard(W,'qa',7200)
os.environ.update(THREAD_ENV)
environment=framework_setup(D/'stage3_s107/vjepa/deps/site',gpu=True)
start=time.perf_counter();install_env=os.environ.copy();installer_source='installed pip'
if importlib.util.find_spec('pip') is None:
    import ensurepip
    wheels=list((pathlib.Path(ensurepip.__file__).parent/'_bundled').glob('pip-*.whl'))
    assert len(wheels)==1,wheels
    installer_source=str(wheels[0]);install_env['PYTHONPATH']=str(wheels[0])+os.pathsep+install_env.get('PYTHONPATH','')
install=subprocess.run([sys.executable,'-B','-m','pip','install','--no-index','--no-cache-dir',
                        '--disable-pip-version-check','-r',str(ROOT/'requirements.txt')],
                       env=install_env,capture_output=True,text=True,timeout=60)
install_seconds=time.perf_counter()-start
assert install.returncode==0,install.stdout+install.stderr
import torch,cv2,pandas as pd
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
mod=load(ROOT/'inference.py','_s178_export_qa')
loader=torch.utils.data.DataLoader
def serial(*a,**kw):kw['num_workers']=0;return loader(*a,**kw)
mod.DataLoader=serial
adapter=mod._s118_adapter();pkg=adapter.load_package(ROOT,'model/stage2/s161')
assert pkg.Tracker.analyze is pkg._s178_analyze
assert adapter._predict_track is adapter.__dict__['_predict_track']
mod._s008_frame_paths=wrap(mod._s008_frame_paths,'frame enumeration')
for name in ('predict_folder','frame_paths'):
    mod._S2_COLLISION_NAMESPACE[name]=wrap(mod._S2_COLLISION_NAMESPACE[name],'collision '+name)
attempts=[]
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        attempts.append(event);raise RuntimeError('offline '+event)
sys.addaudithook(audit)
def trim():gc.collect();torch.cuda.empty_cache();ctypes.CDLL('libc.so.6').malloc_trim(0)

dkb=D/'external_meta/repos/DKB-2000_CrashIntent-AI/data/stage2/labels_manual.csv'
jung=D/'external_meta/repos/Nonmaju_dashcam-accident-analysis_personal_repo/Baseline/external/jungmin_labels/legacy_labels.csv'
with dkb.open(newline='') as f:dkb_ids={r['ID'] for r in csv.DictReader(f) if r['entry_side'] in ('LEFT','RIGHT')}
with jung.open(newline='') as f:jung_ids={r['video_id'] for r in csv.DictReader(f) if r['review_status']=='DONE' and r['entry_side'] in ('LEFT','RIGHT')}
with (D/'s2eval/s109_full_240.csv').open(newline='') as f:known={r['ID'] for r in csv.DictReader(f)}
extra=sorted((dkb_ids|jung_ids)-known);assert len(extra)==76
selected=['ccd_000176','ccd_000484','ccd_000080','ccd_000296',
          'ccd_000019','nexar_00208','nexar_00060','ccd_000009']
ids=sorted(set('ccd_'+ident for ident in extra)|set(selected))
assert len(ids)>=76
audit_rows={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
sources={}
for ident in ids:
    kind,raw=ident.split('_',1)
    if kind=='ccd':src=D/'ccd/frames'/raw
    else:src=pathlib.Path(audit_rows[kind+'/'+raw]['folder'].replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'))
    assert src.is_dir() and list(src.glob('*.jpg')),ident
    sources[ident]=src
inputs=W/'qa_inputs';inputs.mkdir(parents=True,exist_ok=True)
def folder_for(name,group):
    folder=inputs/name/'images';folder.mkdir(parents=True,exist_ok=True)
    for ident in group:
        dest=folder/ident
        if not dest.exists():dest.symlink_to(sources[ident].resolve(),target_is_directory=True)
        assert dest.resolve()==sources[ident].resolve()
    return folder.parent
phase=W/'qa_phases';phase.mkdir(exist_ok=True)
identity=dict(build=hashlib.sha256((H/'build.json').read_bytes()).hexdigest(),
              harness=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
cached_resources=[];cached_cuda=[]
def run(name,group):
    checkpoint('before '+name)
    meta=phase/(name+'.json');data=phase/(name+'.pkl')
    if meta.exists() and data.exists():
        saved=json.loads(meta.read_text())
        if saved['identity']==identity and saved['ids']==group:
            cached_resources.append(saved['resources']);cached_cuda.append(saved['cuda_peak_reserved'])
            print('QA_CACHED',name,flush=True)
            return pd.read_pickle(data),saved['diagnostics'],saved['seconds']
    input_dir=folder_for(name,group)
    print('QA_START',name,len(group),flush=True)
    trim();start=time.perf_counter()
    actual=mod.predict_stage2(input_dir,ROOT/'model/stage2')
    seconds=time.perf_counter()-start
    diag=json.loads(json.dumps(mod._S118_LAST_DIAGNOSTICS))
    assert sorted(actual.ID.astype(str))==sorted(group),(name,actual.ID.tolist(),group)
    actual.to_pickle(data)
    meta.write_text(json.dumps(dict(identity=identity,ids=group,seconds=seconds,
                                    diagnostics=diag,resources=dict(guard.record),
                                    cuda_peak_reserved=torch.cuda.max_memory_reserved())))
    print('QA_DONE',name,'seconds',round(seconds,2),flush=True)
    checkpoint('completed '+name)
    return actual,diag,seconds

candidate_frames=[];candidate_diags={};candidate_seconds=0
for offset in range(0,len(ids),12):
    group=ids[offset:offset+12];name='candidate_%02d'%(offset//12)
    frame,diag,seconds=run(name,group)
    assert not diag.get('rule_error') and not diag.get('rerank_error'),diag
    candidate_frames.append(frame);candidate_diags.update(diag['clips']);candidate_seconds+=seconds
candidate=pd.concat(candidate_frames,ignore_index=True).sort_values('ID').reset_index(drop=True)
assert len(candidate)==len(ids) and len(candidate_diags)==len(ids)

saved_track=pkg.Tracker.analyze;saved_adapter=adapter._predict_track
pkg.Tracker.analyze=pkg._s178_previous_analyze
adapter._predict_track=adapter._s178_previous_predict_track
try:baseline,base_diag,baseline_seconds=run('baseline_selected',sorted(selected))
finally:
    pkg.Tracker.analyze=saved_track;adapter._predict_track=saved_adapter
assert not base_diag.get('rule_error') and not base_diag.get('rerank_error'),base_diag
actual_selected=candidate[candidate.ID.isin(selected)].sort_values('ID').reset_index(drop=True)
baseline=baseline.sort_values('ID').reset_index(drop=True)
protected=[c for c in baseline if c!='entry_side']
assert actual_selected[protected].equals(baseline[protected]),'REAL CANDIDATE PROTECTED OUTPUT MISMATCH'
assert (candidate.entry_frame<=candidate.collision_frame).all(),'REAL CANDIDATE ENTRY CLAMP MISMATCH'
for row in candidate.itertuples(index=False):
    numbers=[int(mod._s008_frame_number(p)) for p in mod._s008_frame_paths(sources[row.ID])]
    assert int(row.entry_frame) in numbers and int(row.collision_frame) in numbers,row.ID
rows=[]
for row in candidate.itertuples(index=False):
    record=candidate_diags[row.ID]
    assert 's109' in record and not record.get('error'),(row.ID,record)
    reason=(record.get('result') or {}).get('reason')
    before=record['s109'][2]
    after=row.entry_side
    assert before in ('LEFT','RIGHT') and after in ('LEFT','RIGHT')
    rows.append(dict(ID=row.ID,base_side=before,candidate_side=after,reason=reason,
                     collision_frame=int(row.collision_frame),entry_frame=int(row.entry_frame),
                     evasion_space=int(row.evasion_space)))
assert any(r['ID']=='ccd_000176' and r['base_side']!=r['candidate_side'] for r in rows)
assert any(r['ID']=='ccd_000080' and r['base_side']!=r['candidate_side'] for r in rows)
assert next(r for r in rows if r['ID']=='ccd_000019')['base_side']==next(r for r in rows if r['ID']=='ccd_000019')['candidate_side']
(H/'qa_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

# Single-file intentional tracker failure: both S176 and S178 fall back to
# the same base row, and the next call recovers in the same process.
one='ccd_000176';single=folder_for('single_fault',[one]);normal=candidate[candidate.ID==one].reset_index(drop=True)
real=pkg.Tracker.analyze
def fail(*a,**kw):raise RuntimeError('intentional S178 per-clip fault')
pkg.Tracker.analyze=fail
try:failed=mod.predict_stage2(single,ROOT/'model/stage2').reset_index(drop=True)
finally:pkg.Tracker.analyze=real
pkg.Tracker.analyze=fail
adapter._predict_track=adapter._s178_previous_predict_track
try:base_failed=mod.predict_stage2(single,ROOT/'model/stage2').reset_index(drop=True)
finally:
    pkg.Tracker.analyze=real;adapter._predict_track=saved_adapter
assert failed.equals(base_failed),'REAL CANDIDATE FAIL-CLOSED MISMATCH'
recovered=mod.predict_stage2(single,ROOT/'model/stage2').reset_index(drop=True)
assert recovered.equals(normal),'REAL CANDIDATE SAME-PROCESS RECOVERY MISMATCH'

resources=guard.close()
for prior in cached_resources:
    resources['peak_rss']=max(resources['peak_rss'],prior['peak_rss'])
    resources['peak_processes']=max(resources['peak_processes'],prior['peak_processes'])
    if prior['min_free_commit_gib'] is not None:
        resources['min_free_commit_gib']=prior['min_free_commit_gib'] if resources['min_free_commit_gib'] is None else min(resources['min_free_commit_gib'],prior['min_free_commit_gib'])
report=dict(passed=True,environment=environment,rows=len(rows),human_extra=len(extra),
            changed=sum(r['base_side']!=r['candidate_side'] for r in rows),
            seconds=dict(candidate=candidate_seconds,baseline_selected=baseline_seconds),
            only_side_changed=True,protected_columns=protected,source_frame_ids=True,entry_clamped=True,
            per_clip_failure_retains_s176=True,recovery_same_process=True,
            s174_added_crossing_preserves_side=True,
            dependency_install_seconds=install_seconds,dependency_installer_source=installer_source,
            network_attempts=attempts,resources=resources,
            cuda_peak_reserved=max([torch.cuda.max_memory_reserved(),*cached_cuda]),
            build_sha256=identity['build'],request_yield_protocol=True,
            local_harness=dict(s164_workers=2,dataloader_workers=0))
assert not attempts and install_seconds<600
assert resources['peak_rss']<=4*2**30 and report['cuda_peak_reserved']<=2*2**30
assert resources['min_free_commit_gib'] is not None and resources['min_free_commit_gib']>=12
(H/'qa.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
