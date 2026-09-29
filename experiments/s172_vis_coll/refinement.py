"""Label-blind 10 Hz refinement around two leading fused peaks."""
import numpy as np
import torch
from head import dense_fusion

def leading_frames(probability,fps):
    cs=np.r_[0.,np.cumsum(probability)]
    grid=np.arange(len(probability)); half=max(1,round(.3*fps))
    mass=cs[np.minimum(grid+half+1,len(grid))]-cs[np.maximum(grid-half,0)]
    centers=[]
    for _ in range(2):
        index=int(np.argmax(mass))
        if mass[index]<0: break
        centers.append(index)
        mass[max(0,index-round(1.5*fps)):min(len(mass),index+round(1.5*fps)+1)]=-1
    return np.unique([min(len(probability)-1,max(0,round(c+d*fps))) for c in centers for d in np.arange(-.5,.501,.1)]).astype(np.int32)

def refine_logits(models,coarse_frames,coarse_features,new_frames,new_features,fps,device):
    # Each target gets the same 15 samples / 3.5 seconds used by the trained
    # 4 Hz convolution. Do not silently apply a 4 Hz convolution at 10 Hz.
    all_frames=np.r_[coarse_frames,new_frames]
    all_features=np.concatenate([coarse_features,new_features])
    unique,first=np.unique(all_frames,return_index=True)
    features=all_features[first].astype(np.float32)
    sample_at=new_frames[:,None]+np.arange(-7,8)[None,:]*fps/4
    hi=np.clip(np.searchsorted(unique,sample_at),1,len(unique)-1)
    lo=hi-1
    a=np.clip((sample_at-unique[lo])/(unique[hi]-unique[lo]),0,1)
    x=features[lo]*(1-a[...,None])+features[hi]*a[...,None]
    # Match zero padding outside the real clip in the coarse network.
    mask=(sample_at>=0)&(sample_at<=unique[-1])
    out=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            xb=torch.from_numpy(x[start:start+16].astype(np.float32)).to(device)
            mb=torch.from_numpy(mask[start:start+16]).to(device)
            logits=torch.stack([model(xb,mb)[:,7] for model in models]).mean(0)
            out.extend(logits.cpu().numpy())
    return np.asarray(out)

def merge_logits(coarse_frames,coarse_logits,new_frames,new_logits):
    # Fine predictions override coincident coarse targets, preserving all
    # untouched coarse support outside the selected windows.
    values={int(f):float(v) for f,v in zip(coarse_frames,coarse_logits)}
    values.update({int(f):float(v) for f,v in zip(new_frames,new_logits)})
    frames=np.array(sorted(values),np.int32)
    return frames,np.array([values[int(f)] for f in frames])
