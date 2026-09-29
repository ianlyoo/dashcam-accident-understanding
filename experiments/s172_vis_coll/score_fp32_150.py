"""Score fixed public fold-0 FP32 control against exported FP16 batch16."""
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
    rows=[]; base=[]; maxdiff=0.; sumdiff=0.; values=0; cache_sumdiff=0.; cache_maxdiff=0.
    for row in meta:
        with np.load(WORK/'revision3/fp32_150'/f"{row['id']}.npz") as fp32, np.load(
            WORK/'revision3/parity_150'/f"{row['id']}.npz") as fp16:
            assert np.array_equal(fp32['frames'],fp16['frames'])
            features=fp32['features'].copy()
            delta=np.abs(features.astype(np.float32)-fp16['features'].astype(np.float32))
            maxdiff=max(maxdiff,float(delta.max()))
            sumdiff+=float(delta.sum(dtype=np.float64)); values+=delta.size
            with np.load(WORK/'coarse'/f"{row['id']}.npz") as cached:
                ix=[i for i in range(len(cached['frames'])) if i%8 in (0,2,3,5,6)]
                assert np.array_equal(cached['frames'][ix],fp16['frames'])
                cache_delta=np.abs(cached['features'][ix].astype(np.float32)-fp16['features'].astype(np.float32))
                cache_sumdiff+=float(cache_delta.sum(dtype=np.float64))
                cache_maxdiff=max(cache_maxdiff,float(cache_delta.max()))
            rows.append(dict(**row,frames=fp32['frames'].copy(),features=features))
        with np.load(WORK/'baseline'/f"{row['id']}.npz") as z:
            base.append((z['frames'].copy(),z['probs'].copy()))
        if len(rows)%25==0: resources()
    outputs=[]
    for seed in (0,1):
        checkpoint=torch.load(WORK/'heads'/f'v2mass_f0_s{seed}_ep30.pt',map_location='cpu',weights_only=False)
        assert set(checkpoint['held_out_ids'])=={r['id'] for r in rows}
        model=TemporalHead(checkpoint['hidden'],drop=0).eval()
        model.load_state_dict(checkpoint['state'],strict=True)
        outputs.append(infer(model,rows,list(range(150)),'cpu'))
    fp16={r['id']:r for r in json.loads((HERE/'parity_150.json').read_text())['picks']}
    picks=[]
    for row,(bf,bp) in zip(rows,base):
        logits=np.mean([outputs[s][row['id']] for s in (0,1)],axis=0)
        frame=decode(dense_fusion(logits,row['frames'],bf,bp,row['n'],'product',.5),9)
        picks.append(dict(id=row['id'],frame=frame,
                          hit=bool(abs(frame/row['fps']-row['toe'])<=.3+1e-9)))
    result=dict(clips=150,fp32_2p5hz_hits=sum(r['hit'] for r in picks),
        fp16_batch16_2p5hz_hits=103,fp16_cached_2p5hz_hits=103,
        fp16_cached_4hz_hits=102,changed_picks_vs_fp16_batch16=sum(
            r['frame']!=fp16[r['id']]['frame'] for r in picks),
        fp32_vs_fp16_descriptor_max_abs_diff=maxdiff,
        fp32_vs_fp16_descriptor_mean_abs_diff=sumdiff/values,
        cached_fp16_vs_batch16_descriptor_mean_abs_diff=cache_sumdiff/values,
        cached_fp16_vs_batch16_descriptor_max_abs_diff=cache_maxdiff,
        fixed_subset='Entire original fold 0, 150 public clips',
        interpretation='FP32 is an extra control; prior production 4 Hz extraction used FP16.',
        picks=picks)
    (HERE/'fp32_150.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='picks'}),flush=True)
if __name__=='__main__': main()
