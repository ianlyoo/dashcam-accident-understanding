"""Paired S174 fallback replay against exact exported S171 rows on 243 labels."""
import argparse,collections,hashlib,json,os,pathlib,pickle,sys,time,types
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
P=argparse.ArgumentParser();P.add_argument('--limit',type=int,default=0);P.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=P.parse_args()
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR') if os.name=='nt' else pathlib.Path('$DATA_DIR')
W=D/'s174_cov'
if os.name!='nt':sys.path.insert(0,str(R))
if a.device=='cuda':
 from candidates.s173_wheel_entry.guard import Guard
 guard=Guard(W,'eval',7200)
 request=pathlib.Path('$GPU_REQUEST_PATH')
else:request=None
def request_checkpoint(point):
 if request is not None and request.exists():
  (W/'eval_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
  print('GPU_YIELD',point,flush=True);raise SystemExit(75)
import torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
source=(D/'s174_cov/candidate/model/stage2/s161/predict.py').read_bytes()
pred=types.ModuleType('_s174_candidate_replay');exec(compile(source,'S174:model/stage2/s161/predict.py','exec'),pred.__dict__)
params=pred.Params(**json.loads((D/'s174_cov/candidate/model/stage2/s161/params.json').read_text())['params'])
raw=[json.loads(x) for x in (H/'audit_labeled.jsonl').read_text().splitlines() if x.strip()]
if not a.limit:assert len(raw)==243,len(raw)
rows=raw[:a.limit or None]
caches={}
for kind in ('ccd','nexar'):
 with (D/'s132_s2track'/('cache_'+kind+'_frcnn_640_0.3.pkl')).open('rb') as f:rawcache=pickle.load(f)
 caches[kind]=rawcache if os.name=='nt' else {k.replace('$DATA_DIR\\','$DATA_DIR/').replace('\\','/'):v for k,v in rawcache.items()}
class Lazy:
 def __init__(self):self.real=None;self.misses=0
 def ready(self):
  if self.real is None:self.real=pred.Detector('frcnn',D/'s174_cov/candidate/model/stage2/s144/detector.pth',device=a.device)
  return self.real
 def load(self,p):self.misses+=1;return self.ready().load(p)
 def __call__(self,images):return self.ready()(images)
det=Lazy();proxy=types.SimpleNamespace(params=params,detector=det)
out=(H/'eval_smoke.jsonl' if a.limit else W/'evaluation_gpu.jsonl' if a.device=='cuda' else H/'evaluation.jsonl')
done={r['key'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
t0=time.perf_counter()
with out.open('a',encoding='utf-8') as f:
 for row in rows:
  if row['key'] in done:continue
  request_checkpoint('before '+row['key'])
  entry=row['s171_entry'];result=dict(reason='s161_original_accepted');elapsed=0;error=None
  if not row['accepted']:
   folder=pathlib.Path(row['folder'] if os.name=='nt' else row['folder'].replace('$DATA_DIR\\','$DATA_DIR/').replace('\\','/'))
   paths=sorted(folder.glob('*.jpg'))
   assert paths,row['key']
   numbers=[int(p.stem.split('_')[-1]) for p in paths]
   c=min(range(len(numbers)),key=lambda i:(abs(numbers[i]-row['s171_collision']),i))
   start=time.perf_counter()
   try:
    result=pred._s174_fallback(proxy,paths,numbers,c)
    pos,_=pred.decide(result,'s109')
    if pos is not None:entry=min(int(numbers[pos]),int(row['s171_collision']))
   except torch.cuda.OutOfMemoryError as exc:
    (W/'eval_oom.json').write_text(json.dumps(dict(key=row['key'],error=str(exc),checkpoint_rows=len(done))))
    print('OOM',row['key'],flush=True);raise SystemExit(76)
   except Exception as exc:
    error=type(exc).__name__+': '+str(exc);result=dict(reason='s174_error')
   elapsed=time.perf_counter()-start
  record=dict(key=row['key'],kind=row['kind'],id=row['id'],fps=row['fps'],labels=row['labels'],
   s171_collision=row['s171_collision'],s171_entry=row['s171_entry'],s174_entry=entry,
   original_s161_accepted=row['accepted'],original_s161_reason=row['reason'],
   extension_accepted=result.get('reason')=='crossing',extension_result={k:result.get(k) for k in
    ('reason','entry_index','track_len','detected_frames','bracket','horizon','horizon_n','lane_k','anchor','s174_actor_candidates')},
   seconds=elapsed,error=error)
  assert entry<=row['s171_collision']
  f.write(json.dumps(record,default=float)+'\n');f.flush();done.add(row['key'])
  print(row['key'],row['reason'],record['extension_result']['reason'],row['s171_entry'],entry,round(elapsed,3),flush=True)
  request_checkpoint('after '+row['key'])
records=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
scores={}
for r in records:
 for lab in r['labels']:
  k=lab['source'];s=scores.setdefault(k,dict(n=0,s171_hits=0,s174_hits=0,repairs=[],breaks=[],changed=0))
  lo,hi=lab['interval'];b=lo-.30000001<=r['s171_entry']/r['fps']<=hi+.30000001
  v=lo-.30000001<=r['s174_entry']/r['fps']<=hi+.30000001
  s['n']+=1;s['s171_hits']+=b;s['s174_hits']+=v;s['changed']+=r['s171_entry']!=r['s174_entry']
  if v and not b:s['repairs'].append(r['key'])
  if b and not v:s['breaks'].append(r['key'])
long=[r['seconds'] for r in records if r['kind']=='nexar']
summary=dict(n=len(records),activation=sum(r['extension_accepted'] for r in records),changed=sum(r['s171_entry']!=r['s174_entry'] for r in records),
 original_s161_accepted=sum(r['original_s161_accepted'] for r in records),
 by_kind={k:dict(n=sum(r['kind']==k for r in records),activated=sum(r['extension_accepted'] for r in records if r['kind']==k),changed=sum(r['s171_entry']!=r['s174_entry'] for r in records if r['kind']==k)) for k in ('ccd','nexar')},
 reasons=dict(collections.Counter(r['extension_result']['reason'] for r in records)),sources=scores,
 long_n=len(long),long_mean_added_seconds=sum(long)/len(long) if long else None,
 long_137_seconds=137*sum(long)/len(long) if long else None,
 errors=[r['key'] for r in records if r['error']],
 baseline='exact exported S171 entry/collision on same 243 labeled JPEG folders',
 caveat='Candidate S174 fallback is cached-detector direct replay, not full exported Stage2; one S161 cached accepted result disagrees with exported S171 and is forced to retain the exact S171 row.',
 build_sha256=hashlib.sha256((H/'build.json').read_bytes()).hexdigest(),seconds=time.perf_counter()-t0)
if not a.limit:assert len(records)==243,len(records)
if a.device=='cuda':
 summary.update(resources=guard.close(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                detector_cache_misses=det.misses,device='cuda')
(H/('eval_smoke.json' if a.limit else 'metrics.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='sources'},default=float),flush=True)
