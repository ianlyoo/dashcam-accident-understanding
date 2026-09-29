"""CPU cost estimate only; synthetic inputs, no scientific result or saved fit."""
import os
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
import json
import time
import torch
from head import TemporalHead

if __name__=='__main__':
    torch.set_num_threads(2); torch.manual_seed(0)
    net=TemporalHead(); opt=torch.optim.AdamW(net.parameters(),lr=.001)
    x=torch.randn(16,180,2048); mask=torch.ones(16,180,dtype=torch.bool)
    target=torch.zeros(16,180); target[:,90]=1
    times=[]
    for i in range(8):
        t=time.perf_counter(); y=net(x,mask)
        loss=-(target*torch.log_softmax(y,1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.perf_counter()-t)
    seconds=sum(times[2:])/len(times[2:])
    print(json.dumps(dict(cpu_seconds_per_batch=seconds,
        conditional_5fold_2seed_30epoch_minutes=seconds*38*5*2*30/60)),flush=True)
