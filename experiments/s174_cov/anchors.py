"""Reconstruct exact final-fit S160 public Nexar picks from its retained rows."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');S=R/'candidates/s160_coll_loc/s160_pkg'
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args()
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=importlib.util.spec_from_file_location('_s174_s160',S/'s160_loc.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
cfg=json.loads((S/'localizer.json').read_text())
z=np.load(S/cfg['nn_file']);nets=[]
for key in cfg['nn_keys']:
 pre=key+'/'
 nets.append(dict(med=z[pre+'med'],scale=z[pre+'scale'],
                  params={k[len(pre):]:z[k] for k in z.files if k.startswith(pre) and k[len(pre):] not in ('med','scale')}))
sources=sorted((D/'s160_coll_loc/rows').glob('*.npz'))
assert len(sources)==750,len(sources)
out=H/'anchors_final_s160.jsonl' if not a.limit else H/'anchors_smoke.jsonl'
done={r['id'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
counts=0;hits=0;t0=time.perf_counter()
with out.open('a',encoding='utf-8') as f:
 for path in sources[:a.limit or None]:
  if path.stem in done:continue
  with np.load(path) as row:
   X=np.zeros((10,91,212),np.float32);M=np.zeros((10,91),bool);FR=np.full((10,91),-1,np.int64)
   for rank in range(10):
    sel=row['rank']==rank;n=int(sel.sum());X[rank,:n]=row['x'][sel];M[rank,:n]=True;FR[rank,:n]=row['frames'][sel]
   pick,_=P.localize(X,M,FR,cfg,nets)
   item=dict(id=path.stem,index=int(pick),n=int(row['n']),fps=float(row['fps']),
             in_sample_hit=abs(pick/float(row['fps'])-float(row['toe']))<=.3+1e-9)
  f.write(json.dumps(item)+'\n');f.flush();counts+=1;hits+=item['in_sample_hit']
  if counts%25==0:print('ANCHORS',counts,'seconds',round(time.perf_counter()-t0,1),flush=True)
rows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
if not a.limit:
 assert len(rows)==750,len(rows)
 assert sum(r['in_sample_hit'] for r in rows)==659,'S160 public final-parity aggregate mismatch'
report=dict(n=len(rows),in_sample_hits=sum(r['in_sample_hit'] for r in rows),
            source='public S160 final-fit exported localizer on retained per-clip feature rows',
            s160_config_sha256=sha(S/'localizer.json'),s160_net_sha256=sha(S/cfg['nn_file']),
            anchor_file_sha256=sha(out),seconds=time.perf_counter()-t0,
            limitation='S160 final-fit anchors are in-sample on Nexar, unlike labeled-243 exact exported S171 anchors.')
(H/('anchors_smoke.json' if a.limit else 'anchors_receipt.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
