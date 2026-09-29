"""One serial runtime check outside gpu.request smoke windows; no artifact edits."""
import datetime
import json
import os
import subprocess
import time
import uuid
import psutil
from qa_supervise import WORK,SCR,LEASE,LOCK,free_commit,write_commit

def run_once():
    assert json.loads((WORK/'qa/long.json').read_text())['passed']
    token=str(uuid.uuid4()); deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        total=0
        for path in (SCR/'gpu_leases').glob('*.json'):
            record=json.loads(path.read_text(encoding='utf-8-sig'))
            if psutil.pid_exists(int(record.get('pid',0))): total+=float(record['cap_gb'])
        if not any((SCR/name).exists() for name in ('gpu.lock','gpu.request')) and not LEASE.exists() and not LOCK.exists() and total+2<=14 and free_commit()>=12:
            try:
                with LOCK.open('x') as stream: json.dump(dict(owner='Sol S172 benchmark',pid=os.getpid(),token=token),stream)
                break
            except FileExistsError: pass
        print(json.dumps(dict(event='benchmark_wait',caps=total)),flush=True); time.sleep(30)
    else: raise TimeoutError('Benchmark admission timeout')
    child=None
    try:
        with LEASE.open('x') as stream:
            json.dump(dict(owner='Sol S172 benchmark',pid=os.getpid(),cap_gb=2,ram_gb=4,project='video',
                token=token,start=datetime.datetime.now().isoformat(),end='bounded 300-second runtime check',checkpoint='per run'),stream)
        write_commit('benchmark')
        command=['wsl','-d','Hermes-Ubuntu','--cd','.',
            '--','/usr/bin/timeout','--signal=TERM','--kill-after=10s','300s','/usr/bin/env','MALLOC_ARENA_MAX=2',
            'S172_REVISION='+os.environ.get('S172_REVISION',''),f'S171_ADMISSION_FREE_COMMIT_GIB={free_commit()}',
            '$DATA_DIR/s065_linux/.venv/bin/python','-B','candidates/s172_vis_coll/bench_runtime.py']
        with (WORK/'logs/benchmark.log').open('a') as out,(WORK/'logs/benchmark.err').open('a') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err)
            deadline=time.monotonic()+325
            while child.poll() is None and time.monotonic()<deadline:
                try: write_commit('benchmark')
                except OSError as exc: print(f'Warning: commit sample skipped: {exc}',flush=True)
                time.sleep(5)
            code=child.wait(timeout=10)
        print(json.dumps(dict(event='benchmark_exit',code=code)),flush=True)
        if code not in (0,75): raise RuntimeError(f'Runtime check failed: {code}')
        return code
    finally:
        if child is not None and child.poll() is None: child.wait(timeout=30)
        for path in (LEASE,LOCK):
            if path.exists() and json.loads(path.read_text()).get('token')==token: path.unlink()

def main():
    while run_once()==75:
        print(json.dumps(dict(event='benchmark_paused_for_gpu_request')),flush=True)
        time.sleep(30)

if __name__=='__main__': main()
