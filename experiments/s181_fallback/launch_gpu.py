"""Bounded S181 WSL CUDA export/QA supervisor with a 2 GiB GPU lease."""

import argparse,ctypes,datetime,json,os,pathlib,subprocess,sys,time

from ctypes import wintypes



H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]

W=pathlib.Path('$DATA_DIR/s181_fallback');(W/'logs').mkdir(parents=True,exist_ok=True)

GPU=pathlib.Path('$USER_HOME/Documents/code/shared_project/scratchpad')

LEASE=GPU/'gpu_leases/video-s181.json';REQUEST=GPU/'gpu.request';LOCK=W.parent/'qa.lock'

p=argparse.ArgumentParser();p.add_argument('--supervise',action='store_true')

p.add_argument('--seconds',type=int,default=3600)

p.add_argument('--job',choices=('export_missing.py','qa.py','diagnose_phase6.py','diagnose_integrator.py','diagnose_side.py','integrator_public'),required=True)
p.add_argument('--mode',choices=('batch','single','full_single'))
p.add_argument('--id',default='')
a=p.parse_args()

if not a.supervise:

    prefix=W/'logs'/(a.job[:-3]+('_'+a.id if a.id else '')+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(str(prefix)+'.out.log','wb') as out,open(str(prefix)+'.err.log','wb') as err:

        proc=subprocess.Popen([sys.executable,'-B',str(H/'launch_gpu.py'),'--supervise',
                               '--seconds',str(a.seconds),'--job',a.job,*(['--mode',a.mode] if a.mode else []),
                               *(['--id',a.id] if a.id else [])],
                              cwd=R,stdout=out,stderr=err,

                              creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS)

    print(json.dumps(dict(pid=proc.pid,log=str(prefix))))

    sys.exit(0)

fields=['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',

        'KernelTotal','KernelPaged','KernelNonpaged','PageSize']

class Perf(ctypes.Structure):

    _fields_=[('cb',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in fields]+[(k,wintypes.DWORD) for k in ['HandleCount','ProcessCount','ThreadCount']]

def free_commit():

    v=Perf();v.cb=ctypes.sizeof(v)

    assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(v),v.cb)

    return (v.CommitLimit-v.CommitTotal)*v.PageSize/2**30

deadline=time.monotonic()+a.seconds;next_vram_check=0

while time.monotonic()<deadline:

    if REQUEST.exists():print('WAIT gpu.request',flush=True);time.sleep(30);continue

    if time.monotonic()<next_vram_check:time.sleep(30);continue

    live=sum(float(json.loads(f.read_text(encoding='utf-8-sig'))['cap_gb'])

             for f in (GPU/'gpu_leases').glob('*.json'))

    free=free_commit()

    smi=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total',

                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)

    used,total=[float(x.strip()) for x in smi.stdout.strip().split(',')]

    vram=(total-used)/1024

    if LEASE.exists() or LOCK.exists() or (GPU/'gpu.lock').exists() or live+2>16 or free<12 or vram<2.5:

        print('WAIT admission',live,round(free,2),round(vram,2),flush=True)

        if vram<2.5:next_vram_check=time.monotonic()+120

        time.sleep(30);continue

    owned=False;owned_lock=False

    try:

        try:

            with LEASE.open('x') as f:

                json.dump(dict(owner='Sol S181',pid=os.getpid(),cap_gb=2,ram_gb=4,

                               start=datetime.datetime.now().isoformat(),end='process exit or per-clip yield',

                               checkpoint='per clip',aggregate_cap_gb=16,actual_free_vram_gib=vram,

                               cap_rationale='team authorized DV <=12 GB plus video 4 GB reservation; two video 2 GB leases fit aggregate 16 GB.',

                               request_rule='No start while gpu.request exists; yield after current clip, free CUDA, poll 30s.'),f)

            owned=True

            with LOCK.open('x') as f:f.write('S181 pid='+str(os.getpid()))

            owned_lock=True

        except FileExistsError:continue

        if REQUEST.exists():continue

        cmd=['wsl','-d','Hermes-Ubuntu',
             '--cd','.','--',
             'timeout','--signal=TERM',str(max(1,int(deadline-time.monotonic())))]
        if a.job=='integrator_public':
            cmd.extend(['/usr/bin/env',f'S171_ADMISSION_FREE_COMMIT_GIB={free:.3f}',
                        'S164_WORKERS=2','S164_DETECT_BATCH=1','MALLOC_ARENA_MAX=2',
                        '$DATA_DIR/s065_linux/.venv/bin/python','-B',
                        'candidates/s167_stack/phase6/qa.py','--panel','public'])
        else:
            cmd.extend(['$DATA_DIR/s065_linux/.venv/bin/python','-B',
                        'candidates/s181_fallback/'+a.job])
        if a.mode:cmd.extend(['--mode',a.mode])
        if a.id:cmd.extend(['--id',a.id])
        if a.job=='diagnose_integrator.py':cmd.extend(['--panel','public'])
        print('RUN',cmd,flush=True)

        run=subprocess.Popen(cmd)

        while run.poll() is None:

            temp=W/('host_commit_'+str(os.getpid())+'.tmp')

            temp.write_text(json.dumps(dict(free_commit_gib=free_commit(),sampled_epoch=time.time())))

            try:os.replace(temp,W/'host_commit.json')
            except PermissionError:pass
            if a.job=='integrator_public':
                phase=W.parent/'s167_stack/phase6'
                other=phase/('host_commit_'+str(os.getpid())+'.tmp')
                other.write_text(json.dumps(dict(free_commit_gib=free_commit(),sampled_epoch=time.time())))
                try:os.replace(other,phase/'host_commit.json')
                except PermissionError:pass
            time.sleep(2)

        code=run.returncode;print('EXIT',code,flush=True)

    finally:

        if owned_lock:

            assert LOCK.read_text()=='S181 pid='+str(os.getpid());LOCK.unlink()

        if owned:

            data=json.loads(LEASE.read_text());assert data['pid']==os.getpid();LEASE.unlink()

    if code==75:

        print('CUDA released; checkpoint retained',flush=True)

        time.sleep(30);continue

    sys.exit(code)

raise TimeoutError('S181 GPU admission deadline')

