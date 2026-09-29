"""Evaluate lower exact-cache frame rates on the original five OOF folds."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from head import TemporalHead,dense_fusion,decode
from train import infer
from extract import resources

HERE=Path(__file__).resolve().parent
WORK=Path('$DATA_DIR/s172_vis_coll')
PATTERNS={'2':[0], '2.5':[0,2,3,5,6], '3':[0,1,2]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rate',choices=PATTERNS,required=True)
    args=ap.parse_args(); torch.set_num_threads(2)
    pattern=PATTERNS[args.rate]; cycle={'2':2,'2.5':8,'3':4}[args.rate]
    meta=json.loads((HERE/'inputs.json').read_text()); rows=[]; base=[]
    for row in meta:
        with np.load(WORK/'coarse'/f'{row["id"]}.npz') as z:
            indices=np.array([i for i in range(len(z['frames'])) if i%cycle in pattern],np.int32)
            frames=z['frames'][indices].copy(); features=z['features'][indices].copy()
            rows.append(dict(**row,frames=frames,features=features))
        with np.load(WORK/'baseline'/f"{row['id']}.npz") as z:
            base.append((z['frames'].copy(),z['probs'].copy()))
        if len(rows)%50==0: resources()
    assert len(rows)==750 and [sum(r['fold']==f for r in rows) for f in range(5)]==[150]*5
    predictions={}
    for fold in range(5):
        test=[i for i,row in enumerate(rows) if row['fold']==fold]
        pair=[]
        for seed in (0,1):
            checkpoint=torch.load(WORK/'heads'/f'v2mass_f{fold}_s{seed}_ep30.pt',map_location='cpu',weights_only=False)
            assert checkpoint['fold']==fold and checkpoint['seed']==seed and checkpoint['epoch']==30
            assert set(checkpoint['held_out_ids'])=={rows[i]['id'] for i in test}
            model=TemporalHead(checkpoint['hidden'],drop=0).eval()
            model.load_state_dict(checkpoint['state'],strict=True)
            pair.append(infer(model,rows,test,'cpu'))
            resources()
        for i in test:
            ident=rows[i]['id']
            predictions[ident]=np.mean([pair[s][ident] for s in (0,1)],axis=0)
        print(json.dumps(dict(event='fold',rate=args.rate,fold=fold)),flush=True)
    picks=[]
    for row,(bf,bp) in zip(rows,base):
        p=dense_fusion(predictions[row['id']],row['frames'],bf,bp,row['n'],'product',.5)
        frame=decode(p,9)
        picks.append(dict(id=row['id'],fold=row['fold'],frame=frame,hit=bool(abs(frame/row['fps']-row['toe'])<=.3+1e-9)))
    hit=sum(r['hit'] for r in picks)
    previous={r['id']:r['error_s']<=.3+1e-9 for r in json.loads((WORK/'baseline.json').read_text())['picks']}
    result=dict(rate_hz=args.rate,hits=hit,baseline=458,original_selected_oof=485,
        folds=[sum(r['hit'] for r in picks if r['fold']==f) for f in range(5)],
        repaired=sum(r['hit'] and not previous[r['id']] for r in picks),
        regressed=sum(not r['hit'] and previous[r['id']] for r in picks),
        cached_features_exact=True,grid='4-Hz DINO source frames subsampled by '+str(pattern)+' modulo '+str(cycle),
        setting='existing 30-epoch seeds 0/1, product weight .5, half-width 9; no decoder reselection',
        selection_bias='Downsampling rate selected on reused 750 five-fold OOF predictions.',
        picks=picks)
    path=HERE/f'rate_{args.rate.replace(".","p")}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='picks'}),flush=True)
if __name__=='__main__': main()
