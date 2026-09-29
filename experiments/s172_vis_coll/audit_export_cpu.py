"""Read the real ZIP: check unregistered head loading and full-fit serialization."""
import ast
import io
import json
from pathlib import Path
import types
import zipfile
import numpy as np
import torch
from head import TemporalHead

HERE=Path(__file__).resolve().parent; WORK=Path('$DATA_DIR/s172_vis_coll')
if __name__=='__main__':
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    archive_record=json.loads((WORK/'pending_archive.json').read_text())
    with zipfile.ZipFile(WORK/'S172_vis.zip') as archive,zipfile.ZipFile('$DATA_DIR/releases/S171_ent.zip') as base:
        original=base.read('inference.py'); current=archive.read('inference.py')
        assert current.startswith(original)
        extra=ast.parse(current[len(original):].decode())
        assert [n.name for n in extra.body if isinstance(n,ast.FunctionDef)]==['_s172_runtime','predict_stage2']
        assert len(extra.body)==3
        module=types.ModuleType('_s172_zip_head')
        exec(compile(archive.read('model/stage2/s172/head.py'),'ZIP/head.py','exec'),vars(module))
        cfg=json.loads(archive.read('model/stage2/s172/config.json'))
        assert cfg['refine'] is False and cfg['selected_oof_hits']==485
        count=0; max_diff=0.
        for seed in (0,1):
            record=torch.load(io.BytesIO(archive.read(f'model/stage2/s172/head_{seed}.pt')),map_location='cpu',weights_only=True)
            fitted=torch.load(WORK/'heads'/f'v2mass_f-1_s{seed}_ep30.pt',map_location='cpu',weights_only=False)
            assert set(record['state'])==set(fitted['state'])
            assert all(torch.equal(v,fitted['state'][k]) for k,v in record['state'].items())
            exported=module.TemporalHead(record['hidden'],drop=0).eval(); exported.load_state_dict(record['state'])
            research=TemporalHead(record['hidden'],drop=0).eval(); research.load_state_dict(fitted['state'])
            for ident in ('00000','00003','00004','00005','00006'):
                with np.load(WORK/'coarse'/f'{ident}.npz') as z:
                    x=torch.from_numpy(z['features'].astype(np.float32))[None]
                with torch.inference_mode():
                    a,b=exported(x),research(x)
                max_diff=max(max_diff,float((a-b).abs().max()))
                assert torch.equal(a,b)
                count+=1
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,archive_sha256=archive_record['sha256'],heads=2,clip_head_pairs=count,
        tensor_weights_exact=True,logits_max_abs_diff=max_diff,unregistered_module_loading=True,
        original_inference_prefix_identical=True,stage1_stage3_definitions_unmodified=True,
        cuda_initialized=False,scope='Serialization/execution parity on reused public cached features, not accuracy evidence')
    (HERE/'export_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
