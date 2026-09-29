"""Frozen local DINOv2-L; public Nexar only; atomic per-clip resume."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch
from transformers import Dinov2Model

from inspect_inputs import Memory
import ctypes

DATA = Path('$DATA_DIR')
WORK = DATA/'s172_vis_coll'
SCR = Path('$USER_HOME/Documents/code/shared_project/scratchpad')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)

def resources():
    mem=Memory(); mem.length=ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    rss=psutil.Process().memory_info().rss
    if rss > 4*2**30 or mem.free_commit < 12*2**30:
        raise RuntimeError(f'Resource cap: rss={rss}, free_commit={mem.free_commit}')
    return rss, mem.free_commit

def load_model():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=Dinov2Model.from_pretrained(str(DATA/'pretrained/dinov2-large-47b73eef'),
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    return model

def extract(model, path, frames, batch=4):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {path}')
    wanted=set(map(int,frames)); last=max(wanted)
    images=[]; out=[]; decode_s=0.; encode_s=0.; t=time.perf_counter()
    def flush():
        nonlocal encode_s
        start=time.perf_counter()
        x=torch.from_numpy(np.stack(images)).to('cuda',dtype=torch.float16)
        with torch.inference_mode():
            h=model(pixel_values=x).last_hidden_state
            out.append(torch.cat([h[:,0], h[:,1:].mean(1)],1).cpu().numpy())
        images.clear()
        encode_s+=time.perf_counter()-start
        resources()
    try:
        for index in range(last+1):
            tick=time.perf_counter()
            if not cap.grab():
                raise RuntimeError(f'Missing frame {index}')
            if index in wanted:
                ok, frame=cap.retrieve()
                if not ok: raise RuntimeError(f'Retrieve failed {index}')
                frame=cv2.cvtColor(cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images.append(((frame.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))
            decode_s+=time.perf_counter()-tick
            if len(images)>=batch: flush()
        if images: flush()
    finally:
        cap.release()
    result=np.concatenate(out)
    assert result.shape==(len(frames),2048) and np.isfinite(result).all()
    return result,dict(seconds=time.perf_counter()-t,decode_seconds=decode_s,encode_seconds=encode_s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--batch',type=int,default=4)
    args=ap.parse_args()
    rows=json.loads((Path(__file__).parent/'inputs.json').read_text())
    dest=WORK/'coarse'; dest.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); resources(); model=load_model(); resources()
    print(json.dumps(dict(event='loaded',seconds=time.perf_counter()-start)),flush=True)
    done=0
    for row in rows:
        target=dest/(row['id']+'.npz')
        if target.exists():
            with np.load(target) as z:
                assert z['features'].shape[1]==2048 and int(z['fold'])==row['fold']
            continue
        if (SCR/'gpu.request').exists() or (SCR/'gpu.lock').exists():
            print(json.dumps(dict(event='pause_at_checkpoint',done=done)),flush=True)
            return 75
        frames=np.unique(np.minimum(np.rint(np.arange(0,row['n']/row['fps'],.25)*row['fps']).astype(np.int32),row['n']-1))
        features,timing=extract(model,DATA/'nexar_collision/train/positive'/(row['id']+'.mp4'),frames,args.batch)
        temp=target.with_suffix('.tmp.npz')
        np.savez(temp,frames=frames,features=features,fold=row['fold'],fps=row['fps'],toe=row['toe'],n=row['n'])
        os.replace(temp,target)
        rss,commit=resources()
        record=dict(id=row['id'],samples=len(frames),**timing,rss_gib=rss/2**30,
            free_commit_gib=commit/2**30,cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        print(json.dumps(record),flush=True)
        with (WORK/'timings.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        done+=1
        if args.limit and done>=args.limit: break
    return 0

if __name__=='__main__':
    raise SystemExit(main())
