"""Lease-owned head fit, including aggregate RAM of the task's CPU sibling."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
import runpy
import sys
import psutil
import torch
import extract

torch.cuda.set_per_process_memory_fraction(1.65*2**30/torch.cuda.get_device_properties(0).total_memory)
original_resources=extract.resources
def combined_resources():
    rss,commit=original_resources()
    other=0
    pid=int(os.environ.get('S172_OTHER_CPU_PID','0'))
    if pid:
        try:
            parent=psutil.Process(pid)
            for process in [parent]+parent.children(recursive=True):
                try: other+=process.memory_info().rss
                except psutil.NoSuchProcess: pass
        except psutil.NoSuchProcess: pass
    if rss+other>3.95*2**30:
        raise RuntimeError(f'Combined S172 RAM cap: own={rss}, sibling={other}')
    if (extract.SCR/'gpu.request').exists() or (extract.SCR/'gpu.lock').exists():
        raise SystemExit(75)
    return rss,commit
extract.resources=combined_resources
if __name__=='__main__':
    sys.argv[0]=str(Path(__file__).parent/'train.py')
    runpy.run_path(sys.argv[0],run_name='__main__')
