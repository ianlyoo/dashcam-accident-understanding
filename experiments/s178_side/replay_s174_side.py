"""Recover S174 accepted crossing sides from public cached detector observations."""
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time
import types

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR') if os.name!='nt' else pathlib.Path('$DATA_DIR')
W=D/'s178_side';W.mkdir(exist_ok=True)
sys.path.insert(0,str(R))
from candidates.s173_wheel_entry.guard import Guard
guard=Guard(W,'replay',3600)
request=pathlib.Path('$GPU_REQUEST_PATH')
def checkpoint(point):
    if request.exists():
        (W/'replay_yield.json').write_text(json.dumps(dict(point=point,epoch=time.time())))
        print('GPU_YIELD',point,flush=True)
        raise SystemExit(75)
checkpoint('before CUDA initialization')
import torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
source=(D/'s176_bucket/candidate/model/stage2/s161/predict.py').read_bytes()
pred=types.ModuleType('_s178_replay');exec(compile(source,'S176:model/stage2/s161/predict.py','exec'),pred.__dict__)
params=pred.Params(**json.loads((D/'s176_bucket/candidate/model/stage2/s161/params.json').read_text())['params'])
public={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
accepted=[r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines()) if r['extension_accepted']]
assert len(accepted)==24
caches={}
for kind in ('ccd','nexar'):
    with (D/'s132_s2track'/('cache_'+kind+'_frcnn_640_0.3.pkl')).open('rb') as f:raw=pickle.load(f)
    caches[kind]={k.replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'):v for k,v in raw.items()}
class Lazy:
    def __init__(self):self.real=None;self.misses=0
    def ready(self):
        if self.real is None:self.real=pred.Detector('frcnn',D/'s176_bucket/candidate/model/stage2/s144/detector.pth',device='cuda')
        return self.real
    def load(self,p):self.misses+=1;return self.ready().load(p)
    def __call__(self,images):return self.ready()(images)
det=Lazy();proxy=types.SimpleNamespace(params=params,detector=det)
out=H/'s174_accepted_side.jsonl'
done={r['key'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
with out.open('a',encoding='utf-8') as f:
    for row in accepted:
        key=row['key']
        if key in done:continue
        checkpoint('before '+key)
        folder=pathlib.Path(public[key]['folder'].replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'))
        paths=sorted(folder.glob('*.jpg'));assert paths,key
        numbers=[int(p.stem.split('_')[-1]) for p in paths]
        c=min(range(len(numbers)),key=lambda i:(abs(numbers[i]-row['s171_collision']),i))
        start=time.perf_counter()
        try:
            result=pred._s174_fallback(proxy,paths,numbers,c)
        except torch.cuda.OutOfMemoryError as exc:
            (W/'replay_oom.json').write_text(json.dumps(dict(key=key,error=str(exc),completed=len(done))))
            torch.cuda.empty_cache()
            raise SystemExit(76)
        assert result['reason']=='crossing' and result['entry_index']==row['extension_result']['entry_index'],(key,result,row)
        assert result['side'] in ('LEFT','RIGHT')
        record=dict(key=key,side=result['side'],entry_index=result['entry_index'],
                    anchor_box=result.get('anchor_box'),seconds=time.perf_counter()-start)
        f.write(json.dumps(record)+'\n');f.flush();done.add(key)
        print(key,result['side'],round(record['seconds'],3),flush=True)
        checkpoint('after '+key)
assert len(done)==24
report=dict(n=len(done),source_sha256=hashlib.sha256(source).hexdigest(),
            resources=guard.close(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),detector_cache_misses=det.misses)
(H/'s174_replay_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
