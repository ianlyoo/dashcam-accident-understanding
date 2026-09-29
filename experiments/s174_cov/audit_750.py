"""S161 activation on all 750 public Nexar clips, final-fit S160 anchor view."""
import collections,hashlib,json,os,pathlib,sys,time,types,zipfile
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];sys.path.insert(0,str(R))
D=pathlib.Path('$DATA_DIR');W=D/'s174_cov'
REQUEST=pathlib.Path('$GPU_REQUEST_PATH')
from candidates.s173_wheel_entry.guard import Guard
guard=Guard(W,'audit750',12600)
import numpy as np,torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:
 source=z.read('model/stage2/s161/predict.py')
 cfg=json.loads(z.read('model/stage2/s161/params.json'))
pred=types.ModuleType('_s174_exact_s171_s161');exec(compile(source,'S171:model/stage2/s161/predict.py','exec'),pred.__dict__)
params=pred.Params(**cfg['params'])
anchors={r['id']:r for r in map(json.loads,(H/'anchors_final_s160.jsonl').read_text().splitlines())}
assert len(anchors)==750
out=W/'audit_750.jsonl';done={r['id'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
def release_request(point):
 if REQUEST.exists():
  (W/'audit_750_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
  print('GPU_YIELD',point,flush=True);raise SystemExit(75)
shared_detector=pred.Detector('frcnn',D/'s174_cov/candidate/model/stage2/s144/detector.pth',device='cuda',max_side=640)
class VideoDetector:
 def __init__(self,vid,cap):self.vid=vid;self.cap=cap;self.misses=0
 def load(self,path):
  self.misses+=1
  i=int(path.stem.split('_')[-1]);self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
  ok,image=self.cap.read()
  if not ok:return None
  h,w=image.shape[:2];scale=640/float(max(h,w))
  if scale<1:image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
  return image
 def __call__(self,images):return shared_detector(images)
 def close(self):self.cap.release()
t0=time.perf_counter()
with out.open('a',encoding='utf-8') as f:
 for count,(vid,anchor) in enumerate(sorted(anchors.items()),1):
  if vid in done:continue
  release_request('before '+vid)
  source_json=json.loads((D/'s142_rr2/cache'/('nexar_'+vid+'.json')).read_text())
  n=int(source_json['n']);c=int(anchor['index']);assert n==anchor['n'] and 0<=c<n,(vid,n,c)
  cap=cv2.VideoCapture(str(D/'nexar_collision/train/positive'/(vid+'.mp4')))
  assert cap.isOpened(),vid
  width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  aspect=height/width if width and height else 0.5625
  det=VideoDetector(vid,cap)
  paths=[pathlib.Path('$DATA_DIR/s174_cov/virtual')/vid/('frame_%06d.jpg'%i) for i in range(n)]
  cache={}
  for key,value in source_json['detections'].items():
   i=int(key)
   if i<0 or i>=n:continue
   boxes=np.asarray(value['boxes'],np.float32).reshape(-1,6)
   hist=np.asarray(value['hist'],np.float32).reshape(-1,64)
   if len(boxes)!=len(hist):continue
   cache[str(paths[i])]=(boxes,hist,aspect)
  start=time.perf_counter()
  try:
   result=pred.analyze(paths,list(range(n)),c,det,params,fps_guess=float(source_json['fps']),cache=cache)
   pos,_=pred.decide(result,'s109')
   error=None
  except pred.FrameBudgetExceeded:
   result=dict(reason='frame_budget',entry_index=None);pos=None;error=None
  except torch.cuda.OutOfMemoryError as exc:
   (W/'audit_750_oom.json').write_text(json.dumps(dict(id=vid,error=str(exc),checkpoint_rows=len(done))))
   print('OOM',vid,flush=True);det.close();raise SystemExit(76)
  except Exception as exc:
   result=dict(reason='error',entry_index=None);pos=None;error=type(exc).__name__+': '+str(exc)
  record=dict(id=vid,n=n,fps=float(source_json['fps']),collision=c,
   anchor_source='S160 final-fit public localizer on retained feature rows',
   reason=result['reason'],accepted=pos is not None,entry_index=int(pos) if pos is not None else None,
   result={k:result.get(k) for k in ('reason','entry_index','track_len','detected_frames','bracket','horizon','horizon_n','anchor','anchor_box','inside_samples','outside_samples')},
   cached_detections=len(cache),detector_cache_misses=det.misses,error=error,
   seconds=time.perf_counter()-start)
  f.write(json.dumps(record,default=float)+'\n');f.flush();done.add(vid)
  det.close()
  if count%25==0:print('AUDIT750',count,'accepted',sum(1 for x in done),'seconds',round(time.perf_counter()-t0,1),flush=True)
  release_request('after '+vid)
rows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
assert len(rows)==750,len(rows)
shared_detector.model=None;torch.cuda.empty_cache()
resources=guard.close()
summary=dict(n=750,accepted=sum(r['accepted'] for r in rows),reasons=dict(collections.Counter(r['reason'] for r in rows)),
 detector_cache_misses=sum(r['detector_cache_misses'] for r in rows),
 errors=[r['id'] for r in rows if r['error']],
 median_cached_detections=float(np.median([r['cached_detections'] for r in rows])),
 mean_seconds=sum(r['seconds'] for r in rows)/750,
 source='S161 exact S171 code and params; public MP4s plus retained FRCNN detection caches; final-fit S160 anchors',
 limitations=['750 final-fit S160 anchors are in-sample; not a fully exported S171 stage2 replay.','Public MP4 decoding rather than exported JPEG panel may alter detector output on cache misses.'],
 s161_source_sha256=hashlib.sha256(source).hexdigest(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),resources=resources)
(H/'audit_750_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
