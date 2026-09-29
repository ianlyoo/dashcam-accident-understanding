"""Behavioral checks for posterior decoding, temporal spacing and fail-closed rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from head import TemporalHead,decode
from refinement import refine_logits
import runtime

WORK=Path('$DATA_DIR/s172_vis_coll')
def main():
    import ast
    sources=list(Path(__file__).parent.glob('*.py'))
    for path in sources: ast.parse(path.read_text(),filename=str(path))
    torch.set_num_threads(2)
    reference=json.loads((WORK/'baseline.json').read_text())
    for row in reference['picks']:
        with np.load(WORK/'baseline'/f'{row["id"]}.npz') as z:
            p=np.zeros(row['n']); p[z['frames']]=z['probs']
            assert decode(p,9)==row['frame']
    torch.manual_seed(9)
    model=TemporalHead(drop=0).eval()
    features=np.random.default_rng(10).normal(size=(35,2048)).astype(np.float32)
    frames=np.arange(35)*8
    with torch.inference_mode(): expected=model(torch.from_numpy(features)[None])[0].numpy()
    actual=refine_logits([model],frames,features,frames,features,32.,'cpu')
    assert np.max(abs(expected-actual))<1e-5,float(np.max(abs(expected-actual)))
    base=pd.DataFrame([dict(ID='a',collision_frame=30,entry_frame=25,entry_side='LEFT',evasion_space=1),
                       dict(ID='b',collision_frame=50,entry_frame=20,entry_side='RIGHT',evasion_space=0)])
    paths=[Path(str(i)) for i in range(60)]
    ns=dict(_s008_frame_paths=lambda p:paths,_s008_frame_number=lambda p:int(p.name),
        _S118_LAST_DIAGNOSTICS={'s142_clips':{i:{'s172_prior':{'frames':[1],'probs':[1.]}} for i in ('a','b')}})
    class Fake:
        mode='good'
        def __init__(self,root):
            torch.rand(1)
            if self.mode=='load': raise RuntimeError('injected load')
        def position(self,paths,prior):
            if self.mode=='runtime': raise RuntimeError('injected runtime')
            return (float('nan') if self.mode=='invalid' else 10),{}
        def close(self): pass
    original=runtime.VisualCollision; runtime.VisualCollision=Fake
    try:
        for mode in ('load','runtime','invalid'):
            Fake.mode=mode
            rng=torch.random.get_rng_state().clone()
            assert runtime.apply(ns,base,Path('.'),Path('.')).equals(base),mode
            assert torch.equal(torch.random.get_rng_state(),rng),'Incumbent RNG changed'
        Fake.mode='good'; actual=runtime.apply(ns,base,Path('.'),Path('.'))
        assert actual.collision_frame.tolist()==[10,10]
        assert actual.entry_frame.tolist()==[10,10]
        assert actual[['ID','entry_side','evasion_space']].equals(base[['ID','entry_side','evasion_space']])
        assert base.collision_frame.tolist()==[30,50]
    finally: runtime.VisualCollision=original
    import hashlib
    result=dict(parsed_sources=len(sources),baseline_decode_parity=750,temporal_spacing_parity=True,
        load_runtime_invalid_fallback=True,entry_clamp_and_protected_fields=True,cpu_rng_preserved=True,
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('head.py','refinement.py','runtime.py','check_cpu.py')})
    (Path(__file__).parent/'cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()
