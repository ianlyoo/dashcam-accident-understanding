"""S161 activation on all public labeled clips using exact exported S171 anchors."""
import argparse,collections,hashlib,importlib.util,json,os,pathlib,pickle,sys,time,types,zipfile
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR') if os.name=='nt' else pathlib.Path('$DATA_DIR')
W=D/'s174_cov'
P=argparse.ArgumentParser();P.add_argument('--limit',type=int,default=0);P.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=P.parse_args()
if os.name!='nt':sys.path.insert(0,str(R))
if a.device=='cuda':
 from candidates.s173_wheel_entry.guard import Guard
 guard=Guard(W,'labeled',3600)
 request=pathlib.Path('$GPU_REQUEST_PATH')
else:request=None
def checkpoint(point):
 if request is not None and request.exists():
  (W/'labeled_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
  print('GPU_YIELD',point,flush=True);raise SystemExit(75)
import torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:
 source=z.read('model/stage2/s161/predict.py')
 cfg=json.loads(z.read('model/stage2/s161/params.json'))
pred=types.ModuleType('_s174_exact_s171_s161');exec(compile(source,'S171:model/stage2/s161/predict.py','exec'),pred.__dict__)
params=pred.Params(**cfg['params'])
raw=[json.loads(x) for x in (D/'s173_wheel_entry/evaluation.jsonl').read_text().splitlines() if x.strip()]
assert len(raw)==243,len(raw)
rows=sorted(raw,key=lambda r:(r['kind'],r['id']))[:a.limit or None]
caches={}
for kind in ('ccd','nexar'):
 with (D/'s132_s2track'/('cache_'+kind+'_frcnn_640_0.3.pkl')).open('rb') as f:rawcache=pickle.load(f)
 caches[kind]=rawcache if os.name=='nt' else {k.replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'):v for k,v in rawcache.items()}
class Lazy:
 def __init__(self):self.real=None;self.misses=0
 def ready(self):
  if self.real is None:
   self.real=pred.Detector('frcnn',D/'s173_wheel_entry/candidate/model/stage2/s144/detector.pth',device=a.device)
  return self.real
 def load(self,p):self.misses+=1;return self.ready().load(p)
 def __call__(self,images):return self.ready()(images)
det=Lazy();out=H/('audit_labeled_smoke.jsonl' if a.limit else 'audit_labeled.jsonl')
done={r['key'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
t0=time.perf_counter()
with out.open('a',encoding='utf-8') as f:
 for row in rows:
  key=row['kind']+'/'+row['id']
  if key in done:continue
  checkpoint('before '+key)
  folder=pathlib.Path(row['folder'].replace('/mnt/d/','$DATA_DIR/') if os.name=='nt' else row['folder'])
  paths=sorted(folder.glob('*.jpg'));assert paths,key
  numbers=[int(p.stem.split('_')[-1]) for p in paths]
  c=min(range(len(numbers)),key=lambda i:(abs(numbers[i]-row['s171_collision']),i))
  before=det.misses;start=time.perf_counter()
  try:
   result=pred.analyze(paths,numbers,c,det,params,cache=caches[row['kind']])
   pos,_=pred.decide(result,'s109')
   reason=result['reason'];error=None
  except pred.FrameBudgetExceeded:
   result=dict(reason='frame_budget',entry_index=None);pos=None;reason='frame_budget';error=None
  except torch.cuda.OutOfMemoryError as exc:
   (W/'labeled_oom.json').write_text(json.dumps(dict(key=key,error=str(exc),checkpoint_rows=len(done))))
   print('OOM',key,flush=True);raise SystemExit(76)
  except Exception as exc:
   result=dict(reason='error',entry_index=None);pos=None;reason='error';error=type(exc).__name__+': '+str(exc)
  predicted=min(numbers[pos],row['s171_collision']) if pos is not None else None
  record=dict(key=key,kind=row['kind'],id=row['id'],folder=str(folder),n=len(numbers),
   s171_collision=row['s171_collision'],s171_entry=row['s171_entry'],fps=row['fps'],labels=row['labels'],
   reason=reason,accepted=pos is not None,predicted=predicted,
   matches_s171=(predicted==row['s171_entry']) if pos is not None else None,
   result={k:result.get(k) for k in ('reason','entry_index','track_len','detected_frames','bracket','horizon','horizon_n','anchor','anchor_box','inside_samples','outside_samples','trace')},
   error=error,detector_cache_misses=det.misses-before,seconds=time.perf_counter()-start)
  f.write(json.dumps(record)+'\n');f.flush();done.add(key)
  print(key,reason,int(pos is not None),'misses',record['detector_cache_misses'],'s',round(record['seconds'],2),flush=True)
  checkpoint('after '+key)
allrows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
summary=dict(n=len(allrows),accepted=sum(r['accepted'] for r in allrows),
 reasons=dict(collections.Counter(r['reason'] for r in allrows)),
 by_kind={k:dict(n=sum(r['kind']==k for r in allrows),accepted=sum(r['accepted'] for r in allrows if r['kind']==k)) for k in ('ccd','nexar')},
 accepted_s171_mismatches=[r['key'] for r in allrows if r['accepted'] and not r['matches_s171']],
 cache_misses=sum(r['detector_cache_misses'] for r in allrows),seconds=time.perf_counter()-t0,
 s161_source_sha256=hashlib.sha256(source).hexdigest(),
 anchor='exact exported S171 final collision in the paired S173 evaluation records; S173 entry output ignored')
if not a.limit:assert len(allrows)==243,len(allrows)
if a.device=='cuda':summary.update(resources=guard.close(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),device='cuda')
(H/('audit_labeled_smoke.json' if a.limit else 'audit_labeled_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
