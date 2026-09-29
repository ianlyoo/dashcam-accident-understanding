"""Independently check persisted fold membership and OOF ensemble reconstruction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; WORK=Path('$DATA_DIR/s172_vis_coll')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
    rows=json.loads((HERE/'inputs.json').read_text()); ids={r['id'] for r in rows}
    assert len(rows)==len(ids)==750
    checkpoints={}; summaries={}
    for tag,loss in [('v1','gaussian'),('v2mass','mass')]:
        for ep in (15,30):
            by_seed={0:{},1:{}}
            for fold in range(5):
                held={r['id'] for r in rows if r['fold']==fold}
                assert len(held)==150
                for seed in (0,1):
                    path=WORK/'heads'/f'{tag}_f{fold}_s{seed}_ep{ep}.pt'
                    c=torch.load(path,map_location='cpu',weights_only=False)
                    assert c['fold']==fold and c['epoch']==ep and c['seed']==seed and c['loss']==loss
                    assert set(c['held_out_ids'])==held and set(c['training_ids'])==ids-held
                    assert set(c['oof'])==held
                    assert all(np.isfinite(v).all() for v in c['oof'].values())
                    for name,digest in c['source_binding'].items(): assert sha(HERE/name)==digest
                    by_seed[seed].update(c['oof'])
                    checkpoints[path.name]=sha(path)
            name=f'{tag}_ep{ep}_s01'
            with np.load(WORK/f'logits_{name}.npz') as predictions:
                assert set(predictions.files)==ids
                for ident in ids:
                    assert np.array_equal(predictions[ident],np.mean([by_seed[s][ident] for s in (0,1)],0))
            result=json.loads((WORK/f'summary_{name}.json').read_text())
            assert len(result['picks'])==750 and len(result['all'])==result['grid']==33
            assert sum(r['error_s']<=.3+1e-9 for r in result['picks'])==result['best']['hits']
            summaries[name]=result['best']
    report=dict(passed=True,clips=750,fold_sizes=[150]*5,checkpoints=len(checkpoints),
        exact_fold_membership=True,train_test_disjoint=True,ensemble_reconstruction_exact=True,
        head_sha256=sha(HERE/'head.py'),input_sha256=sha(HERE/'inputs.json'),
        checkpoint_sha256=checkpoints,coarse_decoder_comparisons=132,best_by_fit=summaries)
    (HERE/'oof_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoint_sha256'}),flush=True)
