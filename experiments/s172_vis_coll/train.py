"""Five original folds; frozen visual features; every tried OOF decoder recorded."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from head import TemporalHead, dense_fusion, decode
from extract import resources

HERE=Path(__file__).resolve().parent
WORK=Path('$DATA_DIR/s172_vis_coll')

def batch(rows,indices,device):
    length=max(len(rows[i]['frames']) for i in indices)
    x=np.zeros((len(indices),length,2048),np.float32)
    mask=np.zeros((len(indices),length),bool)
    target=np.zeros((len(indices),length),np.float32)
    positive=np.zeros((len(indices),length),bool)
    for j,i in enumerate(indices):
        row=rows[i]; count=len(row['frames'])
        x[j,:count]=row['features']; mask[j,:count]=True
        error=(row['frames']/row['fps']-row['toe'])
        positive[j,:count]=np.abs(error)<=.3+1e-9
        target[j,:count]=np.exp(-.5*(error/.16)**2)
        target[j]/=target[j].sum()
    return [torch.from_numpy(v).to(device) for v in (x,mask,target,positive)]

def infer(net,rows,indices,device):
    output={}
    net.eval()
    with torch.inference_mode():
        for j in range(0,len(indices),16):
            ix=indices[j:j+16]; x,m,_,_=batch(rows,ix,device)
            y=net(x,m).float().cpu().numpy()
            for k,i in enumerate(ix): output[rows[i]['id']]=y[k,:len(rows[i]['frames'])]
    return output

def evaluate(rows,logits,tag):
    variants=[('visual',1.)]+[('product',w) for w in (.125,.25,.5,1.,2.,4.)]+[('mix',w) for w in (.1,.25,.5,.75)]
    summary=[]; all_picks=[]
    base=[]
    for row in rows:
        with np.load(WORK/'baseline'/f'{row["id"]}.npz') as z:
            base.append((z['frames'],z['probs']))
    for kind,weight in variants:
        for half in (0,6,9):
            picks=[]
            for row,(bf,bp) in zip(rows,base):
                p=dense_fusion(logits[row['id']],row['frames'],bf,bp,row['n'],kind,weight)
                frame=decode(p,half)
                picks.append(dict(id=row['id'],fold=row['fold'],frame=frame,error_s=abs(frame/row['fps']-row['toe'])))
            record=dict(tag=tag,kind=kind,weight=weight,half=half,hits=sum(r['error_s']<=.3+1e-9 for r in picks),
                fold_hits=[sum(r['error_s']<=.3+1e-9 for r in picks if r['fold']==f) for f in range(5)])
            summary.append(record)
            all_picks.append(picks)
    best=max(summary,key=lambda r:r['hits'])
    picked=all_picks[summary.index(best)]
    previous={r['id']:r['error_s']<=.3+1e-9 for r in json.loads((WORK/'baseline.json').read_text())['picks']}
    repaired=sum(r['error_s']<=.3+1e-9 and not previous[r['id']] for r in picked)
    broken=sum(r['error_s']>.3+1e-9 and previous[r['id']] for r in picked)
    (WORK/f'summary_{tag}.json').write_text(json.dumps(dict(baseline=458,grid=len(summary),best=best,all=summary,
        repaired=repaired,broken=broken,picks=picked,
        selection_bias='All decoder/hyperparameter comparisons reuse the same 750 OOF clips; not fresh confirmation.'),indent=2)+'\n')
    print(json.dumps(dict(event='oof',**best)),flush=True)
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--epochs',type=int,nargs='+',default=[15,30])
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--tag',default='v1')
    ap.add_argument('--loss',choices=['gaussian','mass'],default='gaussian')
    ap.add_argument('--full',action='store_true')
    args=ap.parse_args(); torch.set_num_threads(2)
    cutoff=datetime.datetime(2026,9,28,10 if args.full else 6,tzinfo=datetime.timezone.utc).timestamp()
    binding={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('train.py','head.py','inputs.json')}
    (WORK/f'fit_{args.tag}_{"full" if args.full else "oof"}.json').write_text(
        json.dumps(dict(args=vars(args),source_binding=binding),indent=2)+'\n')
    meta=json.loads((HERE/'inputs.json').read_text())
    ready_deadline=time.monotonic()+60
    while any(not (WORK/'coarse'/f'{row["id"]}.npz').is_file() for row in meta):
        if time.monotonic()>ready_deadline: raise TimeoutError('Incomplete atomic feature cache')
        time.sleep(1)
    rows=[]
    for row in meta:
        with np.load(WORK/'coarse'/f'{row["id"]}.npz') as z:
            rows.append(dict(**row,frames=z['frames'],features=z['features']))
    assert len(rows)==750
    output=WORK/'heads'; output.mkdir(exist_ok=True)
    all_logits={ep:{seed:{} for seed in args.seeds} for ep in args.epochs}
    start=time.perf_counter()
    for fold in ([-1] if args.full else range(5)):
        train=[i for i,row in enumerate(rows) if row['fold']!=fold]
        test=[i for i,row in enumerate(rows) if row['fold']==fold]
        for seed in args.seeds:
            files={ep:output/f'{args.tag}_f{fold}_s{seed}_ep{ep}.pt' for ep in args.epochs}
            if all(p.exists() for p in files.values()):
                for ep in args.epochs:
                    checkpoint=torch.load(files[ep],map_location='cpu',weights_only=False)
                    assert checkpoint['source_binding']==binding,'Training source changed across resume'
                    assert checkpoint['hidden']==args.hidden and checkpoint['loss']==args.loss
                    if not args.full: all_logits[ep][seed].update(checkpoint['oof'])
                continue
            torch.manual_seed(seed); rng=np.random.default_rng(seed)
            model=TemporalHead(args.hidden).to(args.device)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for epoch in range(1,max(args.epochs)+1):
                model.train(); losses=[]
                for offset in range(0,len(train),16):
                    if offset==0: perm=rng.permutation(train)
                    ix=perm[offset:offset+16]
                    x,mask,target,positive=batch(rows,ix,args.device)
                    x=x+torch.randn_like(x)*.01
                    logits=model(x,mask)
                    lp=F.log_softmax(logits,dim=1)
                    loss=-(target*lp).sum(1).mean()
                    if args.loss=='mass':
                        loss=loss-torch.logsumexp(lp.masked_fill(~positive,-1e4),dim=1).mean()
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                    losses.append(float(loss.detach()))
                    resources()
                    if time.time()>cutoff:
                        passed=any(json.loads(p.read_text())['best']['hits']>468 for p in WORK.glob('summary_*.json'))
                        final_cutoff=datetime.datetime(2026,9,28,10,tzinfo=datetime.timezone.utc).timestamp()
                        if args.full or not passed or time.time()>final_cutoff:
                            raise TimeoutError('S172 task decision/release deadline reached')
                        cutoff=final_cutoff
                if epoch in args.epochs:
                    oof={} if args.full else infer(model,rows,test,args.device)
                    checkpoint=dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},
                        hidden=args.hidden,fold=fold,seed=seed,epoch=epoch,oof=oof,source_binding=binding,loss=args.loss,
                        training_ids=[rows[i]['id'] for i in train],held_out_ids=[rows[i]['id'] for i in test])
                    torch.save(checkpoint,files[epoch])
                    if not args.full: all_logits[epoch][seed].update(oof)
                if epoch%5==0:
                    print(json.dumps(dict(event='train',fold=fold,seed=seed,epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-start)),flush=True)
            del model,opt
    if not args.full:
        for ep in args.epochs:
            averaged={r['id']:np.mean([all_logits[ep][s][r['id']] for s in args.seeds],0) for r in rows}
            tag=f'{args.tag}_ep{ep}_s'+''.join(map(str,args.seeds))
            np.savez(WORK/f'logits_{tag}.npz',**averaged)
            evaluate(rows,averaged,tag)
if __name__=='__main__': main()
