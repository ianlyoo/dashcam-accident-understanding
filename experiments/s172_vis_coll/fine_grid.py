"""Calibrate cached fine predictions; the coarse peak schedule stays fixed."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import torch
from head import TemporalHead,dense_fusion,decode
from refinement import leading_frames,refine_logits,merge_logits
from extract import resources,WORK,SCR
from train import evaluate

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    source=json.loads((WORK/'summary_v2mass_ep30_s01.json').read_text())['best']
    raw=json.loads((WORK/'refined_v2mass_ep30_s01.json').read_text())
    expected={r['id']:r['frame'] for r in raw['picks']}
    meta=json.loads((HERE/'inputs.json').read_text())
    coarse=np.load(WORK/'logits_v2mass_ep30_s01.npz')
    cache=WORK/'refine_v2mass_ep30_s01_product_0.5'
    models={}; rows=[]; logits={}; parity=0
    for row in meta:
        if (SCR/'gpu.request').exists() or (SCR/'gpu.lock').exists(): raise SystemExit(75)
        fold=row['fold']
        if fold not in models:
            models[fold]=[]
            for seed in (0,1):
                state=torch.load(WORK/'heads'/f'v2mass_f{fold}_s{seed}_ep30.pt',map_location='cpu',weights_only=False)
                model=TemporalHead(state['hidden'],drop=0)
                model.load_state_dict(state['state']); models[fold].append(model.eval().to('cuda'))
        with np.load(WORK/'coarse'/f'{row["id"]}.npz') as z: fr,feat=z['frames'],z['features']
        with np.load(WORK/'baseline'/f'{row["id"]}.npz') as z: bf,bp=z['frames'],z['probs']
        prior=dense_fusion(coarse[row['id']],fr,bf,bp,row['n'],source['kind'],source['weight'])
        wanted=leading_frames(prior,row['fps'])
        with np.load(cache/f'{row["id"]}.npz') as z:
            assert np.array_equal(z['frames'],wanted)
            extra=z['features']
        values=refine_logits(models[fold],fr,feat,wanted,extra,row['fps'],'cuda')
        merged,values=merge_logits(fr,coarse[row['id']],wanted,values)
        p=dense_fusion(values,merged,bf,bp,row['n'],source['kind'],source['weight'])
        parity+=decode(p,source['half'])==expected[row['id']]
        rows.append(dict(row,frames=merged)); logits[row['id']]=values
        resources()
    assert parity==750,('Fine replay changed',parity)
    tag='fine_v2mass_ep30_s01'
    np.savez(WORK/f'logits_{tag}.npz',**logits)
    evaluate(rows,logits,tag)
    path=WORK/f'summary_{tag}.json'; result=json.loads(path.read_text())
    result.update(coarse_schedule=source,exact_fine_replay=parity,logits_tag=tag,
        selection_bias='132 coarse and 33 fine decoder comparisons on reused OOF; fine sampling fixed from the selected coarse model.')
    result['best'].update(tag=source['tag'],refine=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='fine_grid',best=result['best'],repaired=result['repaired'],broken=result['broken'],parity=parity)),flush=True)
