"""Reproduce the fixed 458/750 S160 proxy without changing its selection."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np

DATA=Path('$DATA_DIR'); SRC=DATA/'s160_coll_loc'; OUT=DATA/'s172_vis_coll'
def logsm(z,m):
    z=np.where(m,z,-1e9); z=z-z.max((1,2),keepdims=True)
    return z-np.log(np.exp(z).sum((1,2),keepdims=True))
def main():
    ids=np.load(SRC/'probs_nn_soft1.npz')['ids'].tolist()
    def load(tag,kind):
        with np.load(SRC/f'{kind}_{tag}.npz') as z:
            if 'ids' in z: assert z['ids'].tolist()==ids
            return z['probs' if kind=='probs' else 'marg'].astype(np.float64)
    conv=(3*load('nn_soft1','probs')+2*load('nn_soft1b','probs'))/5
    pn=(conv+load('nn_soft1_mlp','probs'))/2
    mg=(load('hgb10_big','marg')+load('hgb10_big_s1','marg'))/2
    M=np.zeros(mg.shape,bool); FR=np.full(M.shape,-1,np.int32)
    meta=[]
    for c,ident in enumerate(ids):
        with np.load(SRC/'rows'/f'{ident}.npz') as z:
            for rank in range(10):
                fr=z['frames'][z['rank']==rank]
                M[c,rank,:len(fr)]=True; FR[c,rank,:len(fr)]=fr
            meta.append(dict(id=ident,fold=int(z['fold']),fps=float(z['fps']),toe=float(z['toe']),n=int(z['n'])))
    probs=np.exp(logsm(logsm(mg,M)+logsm(np.log(np.maximum(pn,1e-12)),M),M))
    picks=[]; target=OUT/'baseline'; target.mkdir(parents=True,exist_ok=True)
    for c,row in enumerate(meta):
        fr,inv=np.unique(FR[c][M[c]],return_inverse=True)
        p=np.bincount(inv,weights=probs[c][M[c]])
        cs=np.r_[0,np.cumsum(p)]
        mass=cs[np.searchsorted(fr,fr+9,'right')]-cs[np.searchsorted(fr,fr-9,'left')]
        pick=int(fr[np.argmax(mass)])
        np.savez(target/f'{row["id"]}.npz',frames=fr,probs=p,**row)
        picks.append(dict(**row,frame=pick,error_s=abs(pick/row['fps']-row['toe'])))
    hit=sum(r['error_s']<=.3+1e-9 for r in picks)
    assert hit==458,(hit,'proxy mismatch')
    result=dict(hits=hit,clips=750,fold_hits={f:sum(r['error_s']<=.3+1e-9 for r in picks if r['fold']==f) for f in range(5)},picks=picks)
    (OUT/'baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='picks'}),flush=True)
if __name__=='__main__': main()
