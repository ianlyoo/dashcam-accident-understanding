"""Reduce checkpoint-loading RAM; preserve the exact FP16 inference parameters."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Dinov2Model
from build import sha
from extract import extract,resources

HERE=Path(__file__).resolve().parent; DATA=Path('$DATA_DIR'); WORK=DATA/'s172_vis_coll'
WEIGHT='model/stage2/s172/backbone/model.safetensors'
NOTICE='model/licenses/S172-NOTICE.txt'
if __name__=='__main__':
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    old=WORK/'candidate'; revision=WORK/'revision2'; root=revision/'candidate'
    assert not revision.exists(),'Preserve existing revision'
    root.mkdir(parents=True); (revision/'logs').mkdir(); (revision/'qa').mkdir()
    previous=json.loads((WORK/'build.json').read_text())
    assert sha(old/WEIGHT)=='399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23'
    for path in old.rglob('*'):
        if not path.is_file(): continue
        name=path.relative_to(old).as_posix()
        if name==WEIGHT: continue
        target=root/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    converted={}
    with safe_open(old/WEIGHT,framework='pt',device='cpu') as source:
        for name in source.keys():
            value=source.get_tensor(name)
            converted[name]=(value.to(torch.float16) if value.is_floating_point() else value.clone()).contiguous()
            assert torch.isfinite(converted[name]).all(),name
            resources()
    del value
    save_file(converted,str(root/WEIGHT),metadata={'format':'pt',
        's172_conversion':'2026-09-28: stored as FP16 to match the existing FP16 CUDA inference; no fitting or parameter update',
        'source_sha256':previous['members'][WEIGHT]})
    count=len(converted); del converted; gc.collect()
    with safe_open(old/WEIGHT,framework='pt',device='cpu') as source,safe_open(root/WEIGHT,framework='pt',device='cpu') as target:
        assert source.keys()==target.keys()
        for name in source.keys():
            a,b=source.get_tensor(name),target.get_tensor(name)
            assert torch.equal(a.to(b.dtype),b),name
    del a,b; gc.collect(); resources()
    print(json.dumps(dict(event='half_parameters_verified',tensors=count,bytes=(root/WEIGHT).stat().st_size)),flush=True)
    frames=np.array([0,600],np.int32); video=DATA/'nexar_collision/train/positive/00000.mp4'
    values=[]
    for path in (old/'model/stage2/s172/backbone',root/'model/stage2/s172/backbone'):
        model,info=Dinov2Model.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model=model.eval().requires_grad_(False).to('cuda')
        features,timing=extract(model,video,frames)
        values.append(features)
        del model; gc.collect(); torch.cuda.empty_cache(); resources()
    assert np.array_equal(values[0],values[1]),float(np.max(abs(values[0].astype(np.float32)-values[1])))
    conversion=dict(source_sha256=sha(old/WEIGHT),converted_sha256=sha(root/WEIGHT),tensors=count,
        exact_original_to_fp16_parameters=True,public_frames=frames.tolist(),features_exact=True,max_abs_feature_diff=0.,
        converted_bytes=(root/WEIGHT).stat().st_size,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        reason='Initial long QA guard stopped at 4.06 GiB RSS during backbone load; reduce transient checkpoint RAM')
    with (root/NOTICE).open('a') as stream:
        stream.write('\nS172 revision 2 changes the storage dtype of the DINO checkpoint from FP32\n'
            'to FP16, the dtype already used by this CUDA inference. All converted\n'
            'parameters equal original.to(float16); public-frame descriptors agree\n'
            'bit for bit. No training or parameter update. Conversion notice also\n'
            'appears in the safetensors header. Original source: '+conversion['source_sha256']+'\n')
    current={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    changed=[name for name in current if current[name]!=previous['members'][name]]
    assert set(changed)=={WEIGHT,NOTICE},changed
    record=dict(previous,members=current,precision_conversion=conversion,
        previous_build_sha256=sha(WORK/'build.json'),previous_artifact=json.loads((WORK/'pending_archive.json').read_text()),
        initial_qa_failure=json.loads((WORK/'qa/long_resource_failure.json').read_text()),
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    (revision/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    (HERE/'precision_conversion.json').write_text(json.dumps(conversion,indent=2)+'\n')
    print(json.dumps(dict(event='revision2_staged',conversion=conversion,changed_from_initial=changed)),flush=True)
