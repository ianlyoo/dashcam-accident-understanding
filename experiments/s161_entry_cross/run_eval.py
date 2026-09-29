"""Acquire the shared 2 GiB GPU lease, check commit, and run one evaluation process."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import psutil

HERE=Path(__file__).resolve().parent
ROOT=Path('$DATA_DIR/s161_entry_cross')
LEASES=Path('$USER_HOME/Documents/code/shared_project/scratchpad/gpu_leases')
P=type('P',(ctypes.Structure,),{'_fields_':[('cb',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']]+[(k,wintypes.DWORD) for k in ['HandleCount','ProcessCount','ThreadCount']]})
p=P();p.cb=ctypes.sizeof(p)
assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p),p.cb)
free=(p.CommitLimit-p.CommitTotal)*p.PageSize/2**30
assert free>=12, ('free commit',free)
assert not (LEASES.parent/'gpu.lock').exists()
assert not (LEASES.parent/'gpu.request').exists()
live=[]
for path in LEASES.glob('*.json'):
    v=json.loads(path.read_text(encoding='utf-8-sig'))
    if psutil.pid_exists(int(v['pid'])): live.append(v)
assert sum(v['cap_gb'] for v in live)+2<=14,live
lease=LEASES/'video-s161-entry.json'
from datetime import datetime,timezone
with lease.open('x') as f:
    json.dump(dict(owner='Sol Worker S161',pid=os.getpid(),cap_gb=2,ram_gb=3,
                   start=datetime.now(timezone.utc).isoformat(),end='on process exit',checkpoint='per clip'),f)
try:
    ROOT.mkdir(exist_ok=True)
    print('free_commit_gb',free,'lease',str(lease),flush=True)
    runpy.run_path(str(HERE/'evaluate.py'),run_name='__main__')
finally:
    lease.unlink()
