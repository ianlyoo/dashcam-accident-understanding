"""One serialized WSL CUDA QA plus long-clip parity under shared locks."""
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path('$DATA_DIR/s161_entry_cross')
LEASES=Path('$USER_HOME/Documents/code/shared_project/scratchpad/gpu_leases')
P=type('P',(ctypes.Structure,),{'_fields_':[('cb',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']]+[(k,wintypes.DWORD) for k in ['HandleCount','ProcessCount','ThreadCount']]})
claim=dict(owner='Sol Worker S161 QA',pid=os.getpid(),cap_gb=2,ram_gb=3,
           start=datetime.now(timezone.utc).isoformat(),end='on exit',checkpoint='QA completion')
lock=ROOT.parent/'qa.lock'
lease=LEASES/'video-entry.json'
while True:
    p=P();p.cb=ctypes.sizeof(p)
    assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p),p.cb)
    free=(p.CommitLimit-p.CommitTotal)*p.PageSize/2**30
    caps=0
    for path in LEASES.glob('*.json'):
        try:v=json.loads(path.read_text(encoding='utf-8-sig'))
        except FileNotFoundError:continue
        if psutil.pid_exists(int(v['pid'])):caps+=v['cap_gb']
    if (free>=12 and caps+2<=14 and not (LEASES.parent/'gpu.lock').exists()
            and not (LEASES.parent/'gpu.request').exists()):
        try:
            with lock.open('x') as f:json.dump(claim,f)
            break
        except FileExistsError:pass
    print('Waiting for shared QA/GPU slot; caps=',caps,'free_commit=',round(free,2),flush=True)
    time.sleep(20)
try:
    with lease.open('x') as f:json.dump(claim,f)
    try:
        cmd=['wsl.exe','-d','Hermes-Ubuntu','--cd','.',
             '--','$DATA_DIR/s065_linux/.venv/bin/python','-B']
        with (ROOT/'logs/qa_v3.log').open('w') as log:
            subprocess.run(cmd+['-m','candidates.s161_entry_cross.qa','--stage',
                '$DATA_DIR/s161_entry_cross/stage','--output',
                '$DATA_DIR/s161_entry_cross/qa_v3'],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (ROOT/'logs/integration.log').open('w') as log:
            subprocess.run(cmd+['-m','candidates.s161_entry_cross.integration'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('QA and integration PASS',flush=True)
    finally:lease.unlink()
finally:lock.unlink()
