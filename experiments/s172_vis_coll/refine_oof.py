"""Actual DINO 10 Hz OOF refinement; no labels in the peak schedule."""
import argparse
import json
from pathlib import Path
import os
import time
import numpy as np
import torch
from extract import DATA,WORK,SCR,load_model,extract,resources
from head import TemporalHead,dense_fusion,decode
from refinement import leading_frames,refine_logits,merge_logits

HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',required=True)
    ap.add_argument('--tag',default='v1')
    ap.add_argument('--epoch',type=int,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    args=ap.parse_args()
    selected=json.loads(Path(args.selection).read_text())['best']
    rows=json.loads((HERE/'inputs.json').read_text())
    coarse=np.load(WORK/f'logits_{selected["tag"]}.npz')
    dest=WORK/f'refine_{selected["tag"]}_{selected["kind"]}_{selected["weight"]}'; dest.mkdir(exist_ok=True)
    backbone=load_model(); models_by_fold={}; picks=[]
    for row in rows:
        target=dest/f'{row["id"]}.npz'
        with np.load(WORK/'coarse'/f'{row["id"]}.npz') as z:
            fr,feat=z['frames'],z['features']
        with np.load(WORK/'baseline'/f'{row["id"]}.npz') as z:
            bf,bp=z['frames'],z['probs']
        prior=dense_fusion(coarse[row['id']],fr,bf,bp,row['n'],selected['kind'],selected['weight'])
        wanted=leading_frames(prior,row['fps'])
        if not target.exists():
            if (SCR/'gpu.request').exists() or (SCR/'gpu.lock').exists(): return 75
            t=time.perf_counter()
            features,timing=extract(backbone,DATA/'nexar_collision/train/positive'/f'{row["id"]}.mp4',wanted)
            temp=target.with_suffix('.tmp.npz'); np.savez(temp,frames=wanted,features=features)
            os.replace(temp,target)
            print(json.dumps(dict(event='fine',id=row['id'],**timing)),flush=True)
        with np.load(target) as z:
            assert np.array_equal(z['frames'],wanted),'Refinement schedule changed'
            fine=z['features']
        if row['fold'] not in models_by_fold:
            models=[]
            for seed in args.seeds:
                record=torch.load(WORK/'heads'/f'{args.tag}_f{row["fold"]}_s{seed}_ep{args.epoch}.pt',map_location='cpu',weights_only=False)
                net=TemporalHead(record['hidden'],drop=0)
                net.load_state_dict(record['state']); models.append(net.eval().to('cuda'))
            models_by_fold[row['fold']]=models
        models=models_by_fold[row['fold']]
        refined=refine_logits(models,fr,feat,wanted,fine,row['fps'],'cuda')
        merged,values=merge_logits(fr,coarse[row['id']],wanted,refined)
        p=dense_fusion(values,merged,bf,bp,row['n'],selected['kind'],selected['weight'])
        frame=decode(p,selected['half'])
        picks.append(dict(id=row['id'],fold=row['fold'],frame=frame,error_s=abs(frame/row['fps']-row['toe'])))
        resources()
    hits=sum(r['error_s']<=.3+1e-9 for r in picks)
    result=dict(baseline=458,coarse=selected,grid=1,best=dict(selected,hits=hits,refine=True,
        fold_hits=[sum(r['error_s']<=.3+1e-9 for r in picks if r['fold']==f) for f in range(5)]),picks=picks,
        selection_bias='Refinement evaluated after selecting coarse head/fusion on the same 750 clips; no independent confirmation.')
    (WORK/f'refined_{selected["tag"]}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='refined_oof',coarse_hits=selected['hits'],hits=hits,delta=hits-458)),flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
