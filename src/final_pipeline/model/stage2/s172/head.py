"""Position-free temporal visual head; 15-sample receptive field at 4 Hz."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TemporalHead(nn.Module):
    def __init__(self,hidden=64,drop=.2):
        super().__init__()
        self.proj=nn.Linear(2048,hidden)
        self.layers=nn.ModuleList([nn.Conv1d(hidden,hidden,3,padding=d,dilation=d) for d in (1,2,4)])
        self.drop=nn.Dropout(drop)
        self.head=nn.Linear(hidden,1)
    def forward(self,x,mask=None):
        # Separate unit normalization of global CLS and patch mean; no fitted
        # preprocessing statistics and no position/clip length inputs.
        x=x.reshape(*x.shape[:-1],2,1024)
        x=F.normalize(x,dim=-1).flatten(-2)*32
        h=F.gelu(self.proj(x))
        if mask is not None: h=h*mask.unsqueeze(-1)
        h=h.transpose(1,2)
        for layer in self.layers:
            h=h+self.drop(F.gelu(layer(h)))
            if mask is not None: h=h*mask.unsqueeze(1)
        logits=self.head(h.transpose(1,2)).squeeze(-1)
        return logits if mask is None else logits.masked_fill(~mask,-1e4)

def probabilities(logits):
    z=np.asarray(logits,dtype=np.float64); z=z-z.max()
    p=np.exp(z)
    return p/p.sum()

def dense_fusion(logits,frames,base_frames,base_probs,n,kind,weight):
    grid=np.arange(n)
    visual=probabilities(np.interp(grid,frames,logits))
    baseline=np.zeros(n,dtype=np.float64)
    np.add.at(baseline,np.asarray(base_frames,dtype=np.int32),base_probs)
    baseline/=baseline.sum()
    if kind=='visual': return visual
    if kind=='mix': return (1-weight)*baseline+weight*visual
    if kind=='product':
        z=np.log(np.maximum(baseline,1e-30))+weight*np.log(np.maximum(visual,1e-30))
        p=probabilities(z); p[baseline==0]=0
        return p/p.sum()
    raise ValueError(kind)

def decode(p,half=9):
    grid=np.arange(len(p)); cs=np.r_[0.,np.cumsum(p)]
    mass=cs[np.minimum(grid+half+1,len(p))]-cs[np.maximum(grid-half,0)]
    # Match baseline support-only decoder; zero-probability holes are excluded.
    return int(np.argmax(np.where(p>0,mass,-1)))
