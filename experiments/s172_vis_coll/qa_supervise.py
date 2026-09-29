"""Serial shared-lock WSL QA with Windows commit telemetry and bounded children."""
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
from revision_path import revision_work

WORK=revision_work(Path('$DATA_DIR/s172_vis_coll'))
SCR=Path('$USER_HOME/Documents/code/shared_project/scratchpad')
LEASE=SCR/'gpu_leases/video-s172.json'
LOCK=Path('$DATA_DIR/qa.lock')
def free_commit():
    mem=Memory(); mem.length=ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    return mem.free_commit/2**30
def write_commit(panel):
    temp=WORK/f'host_commit.{os.getpid()}.tmp'
    temp.write_text(json.dumps(dict(free_commit_gib=free_commit(),sampled_epoch=time.time(),job=panel)))
    os.replace(temp,WORK/'host_commit.json')
def main():
    for panel in ('public','long'):
        token=str(uuid.uuid4()); start=time.monotonic()
        while time.monotonic()-start<7200:
            total=0; valid=True
            for path in (SCR/'gpu_leases').glob('*.json'):
                try:
                    record=json.loads(path.read_text(encoding='utf-8-sig'))
                    if psutil.pid_exists(int(record.get('pid',0))): total+=float(record['cap_gb'])
                except (OSError,ValueError,KeyError): valid=False
            if valid and not (SCR/'gpu.lock').exists() and not (SCR/'gpu.request').exists() and not LEASE.exists() and not LOCK.exists() and total+2<=14 and free_commit()>=12:
                try:
                    with LOCK.open('x') as stream:
                        json.dump(dict(owner='Sol S172 QA',pid=os.getpid(),token=token,panel=panel),stream)
                    break
                except FileExistsError: pass
            print(json.dumps(dict(event='qa_wait',panel=panel,caps=total)),flush=True)
            time.sleep(30)
        else: raise TimeoutError('QA admission timeout')
        child=None
        try:
            with LEASE.open('x') as stream:
                json.dump(dict(owner='Sol Worker S172 QA',pid=os.getpid(),cap_gb=2,ram_gb=4,project='video',
                    token=token,start=datetime.datetime.now().isoformat(),end='bounded QA panel',checkpoint=panel),stream)
            write_commit(panel)
            command=['wsl','-d','Hermes-Ubuntu','--cd','.',
                '--','/usr/bin/timeout','--signal=TERM','--kill-after=10s','1450s','/usr/bin/env','MALLOC_ARENA_MAX=2',
                'S164_WORKERS=2','S164_DETECT_BATCH=1',f'S171_ADMISSION_FREE_COMMIT_GIB={free_commit()}',
                'S172_REVISION='+os.environ.get('S172_REVISION',''),
                '$DATA_DIR/s065_linux/.venv/bin/python','-B','candidates/s172_vis_coll/qa.py','--panel',panel]
            with (WORK/f'logs/qa_{panel}.log').open('w') as out,(WORK/f'logs/qa_{panel}.err').open('w') as err:
                child=subprocess.Popen(command,stdout=out,stderr=err)
                deadline=time.monotonic()+1480
                while child.poll() is None and time.monotonic()<deadline:
                    try: write_commit(panel)
                    except OSError as exc: print(f'Warning: commit telemetry skipped: {exc}',flush=True)
                    time.sleep(5)
                code=child.wait(timeout=10)
            print(json.dumps(dict(event='qa_exit',panel=panel,code=code)),flush=True)
            if code!=0: raise RuntimeError(f'{panel} QA failed: {code}')
        finally:
            # A still-live child owns the lease until its timeout; never release early.
            if child is not None and child.poll() is None: child.wait(timeout=30)
            for path in (LEASE,LOCK):
                if path.exists() and json.loads(path.read_text()).get('token')==token: path.unlink()
if __name__=='__main__': main()
