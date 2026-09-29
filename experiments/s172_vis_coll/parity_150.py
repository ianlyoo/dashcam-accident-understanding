"""Checkpointed exact-fold0 descriptors from exported batch16/autocast DINO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from transformers import Dinov2Model
from extract import extract,resources

HERE=Path(__file__).resolve().parent
DATA=Path('$DATA_DIR'); WORK=DATA/'s172_vis_coll'
REQUEST=Path('$USER_HOME/Documents/code/shared_project/scratchpad/gpu.request')

class AutocastBackbone:
    def __init__(self,model): self.model=model
    def __call__(self,**kw):
        if REQUEST.exists(): raise SystemExit(75)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            return self.model(**kw)

def main():
    rows=[r for r in json.loads((HERE/'inputs.json').read_text()) if r['fold']==0]
    assert len(rows)==150
    target=WORK/'revision3/parity_150'; target.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    if REQUEST.exists(): raise SystemExit(75)
    model,info=Dinov2Model.from_pretrained(str(WORK/'revision3/candidate/model/stage2/s172/backbone'),
        local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,
        attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=AutocastBackbone(model.eval().requires_grad_(False).to('cuda'))
    resources(); done=0
    for row in rows:
        path=target/(row['id']+'.npz')
        if path.exists():
            with np.load(path) as z:
                assert len(z['frames'])==len(z['features']) and z['features'].shape[1]==2048
            done+=1; continue
        if REQUEST.exists(): raise SystemExit(75)
        with np.load(WORK/'coarse'/f"{row['id']}.npz") as z:
            indices=[i for i in range(len(z['frames'])) if i%8 in (0,2,3,5,6)]
            frames=z['frames'][indices].copy()
        features,timing=extract(model,DATA/'nexar_collision/train/positive'/f"{row['id']}.mp4",frames,batch=16)
        with tempfile.NamedTemporaryFile(dir=target,suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            np.savez(stream,frames=frames,features=features,seconds=timing['seconds'])
        os.replace(temp,path)
        done+=1
        if done%10==0: print(json.dumps(dict(event='parity_checkpoint',done=done)),flush=True)
        resources()
    print(json.dumps(dict(event='parity_complete',clips=done,cuda_peak_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
if __name__=='__main__': main()
