"""Windows lease owner; bounded subprocess; resumable public feature extraction."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import psutil
from inspect_inputs import Memory

HERE=Path(__file__).resolve().parent
WORK=Path('$DATA_DIR/s172_vis_coll')
SCR=Path('$USER_HOME/Documents/code/shared_project/scratchpad')
LEASE=SCR/'gpu_leases/video-s172.json'
PYTHON=Path('./.venv/Scripts/python.exe')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--deadline-seconds',type=int,default=30000)
    ap.add_argument('--script',default='extract.py',choices=['extract.py','refine_oof.py','train.py','gpu_train.py','fine_grid.py','half_revision.py','parity_150.py','parity_fp32_150.py'])
    ap.add_argument('--args-json',default='[]')
    ap.add_argument('--args-file')
    args=ap.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+args.deadline_seconds
    token=str(uuid.uuid4())
    while time.monotonic()<end:
        mem=Memory(); mem.length=ctypes.sizeof(mem)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total=0
        for path in (SCR/'gpu_leases').glob('*.json'):
            record=json.loads(path.read_text(encoding='utf-8-sig'))
            if psutil.pid_exists(int(record.get('pid',0))): total+=float(record.get('cap_gb',0))
        blocked=(SCR/'gpu.lock').exists() or (SCR/'gpu.request').exists() or LEASE.exists()
        if blocked or total+2>14 or mem.free_commit<12*2**30:
            print(json.dumps(dict(event='lease_wait',caps=total,blocked=blocked,free_commit_gib=mem.free_commit/2**30)),flush=True)
            time.sleep(30); continue
        with LEASE.open('x') as stream:
            json.dump(dict(owner='Sol Worker S172',pid=os.getpid(),project='video',cap_gb=2,ram_gb=4,
                start=datetime.datetime.now().isoformat(),end='per bounded extraction job',checkpoint='every clip',token=token),stream)
        try:
            command=[str(PYTHON),'-B',str(HERE/args.script)]
            command+=['--limit',str(args.limit)] if args.script=='extract.py' else json.loads(
                Path(args.args_file).read_text() if args.args_file else args.args_json)
            print(json.dumps(dict(event='start',command=command)),flush=True)
            result=subprocess.run(command,timeout=max(1,end-time.monotonic()),check=False)
        finally:
            if LEASE.exists() and json.loads(LEASE.read_text())['token']==token:
                LEASE.unlink()
        if result.returncode==75:
            time.sleep(30); continue
        print(json.dumps(dict(event='exit',code=result.returncode)),flush=True)
        return result.returncode
    raise TimeoutError('Extraction admission deadline')

if __name__=='__main__':
    raise SystemExit(main())
