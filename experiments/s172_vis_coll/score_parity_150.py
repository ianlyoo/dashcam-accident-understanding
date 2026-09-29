"""Measure actual batch16/autocast feature and OOF effect on fixed fold0."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import torch
from head import TemporalHead,dense_fusion,decode
from train import infer
from extract import resources

HERE=Path(__file__).resolve().parent; WORK=Path('$DATA_DIR/s172_vis_coll')
def main():
    torch.set_num_threads(2)
    meta=[r for r in json.loads((HERE/'inputs.json').read_text()) if r['fold']==0]
    assert len(meta)==150
    rows=[]; base=[]; max_diff=0.; changed=0; total=0
    for row in meta:
        with np.load(WORK/'coarse'/f"{row['id']}.npz") as original, np.load(
            WORK/'revision3/parity_150'/f"{row['id']}.npz") as actual:
            ix=[i for i in range(len(original['frames'])) if i%8 in (0,2,3,5,6)]
            assert np.array_equal(original['frames'][ix],actual['frames'])
            old=original['features'][ix].astype(np.float32); new=actual['features'].astype(np.float32)
            delta=np.abs(old-new)
            max_diff=max(max_diff,float(delta.max()))
            changed+=int(np.count_nonzero(delta)); total+=delta.size
            rows.append(dict(**row,frames=actual['frames'].copy(),features=actual['features'].copy()))
        with np.load(WORK/'baseline'/f"{row['id']}.npz") as z:
            base.append((z['frames'].copy(),z['probs'].copy()))
        if len(rows)%25==0: resources()
    pair=[]
    for seed in (0,1):
        checkpoint=torch.load(WORK/'heads'/f'v2mass_f0_s{seed}_ep30.pt',map_location='cpu',weights_only=False)
        assert set(checkpoint['held_out_ids'])=={r['id'] for r in rows}
        model=TemporalHead(checkpoint['hidden'],drop=0).eval()
        model.load_state_dict(checkpoint['state'],strict=True)
        pair.append(infer(model,rows,list(range(len(rows))),'cpu'))
    hits=0; picks=[]
    for row,(bf,bp) in zip(rows,base):
        logits=np.mean([pair[s][row['id']] for s in (0,1)],axis=0)
        frame=decode(dense_fusion(logits,row['frames'],bf,bp,row['n'],'product',.5),9)
        hit=bool(abs(frame/row['fps']-row['toe'])<=.3+1e-9)
        hits+=hit; picks.append(dict(id=row['id'],frame=frame,hit=hit))
    original={r['id']:r for r in json.loads((HERE/'rate_2p5.json').read_text())['picks'] if r['fold']==0}
    assert len(original)==150
    result=dict(clips=150,original_4hz_hits=102,cached_2p5hz_hits=103,
        actual_batch16_autocast_2p5hz_hits=hits,descriptor_max_abs_diff=max_diff,
        descriptor_changed_values=changed,descriptor_total_values=total,
        picks_changed=sum(r['frame']!=original[r['id']]['frame'] for r in picks),
        fixed_subset='Original fold 0, all 150 public positive clips',
        caveat='Reused selected OOF subset; original 4 Hz cache represents an FP16, batch4 pass, not FP32.',
        picks=picks)
    (HERE/'parity_150.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='picks'}),flush=True)
if __name__=='__main__': main()
