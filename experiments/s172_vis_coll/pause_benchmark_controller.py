"""Replace only the idle benchmark controller after team's request override."""
from pathlib import Path
import psutil

scratch=Path('$USER_HOME/Documents/code/shared_project/scratchpad')
assert (scratch/'gpu.request').exists()
assert not (scratch/'gpu_leases/video-s172.json').exists(), 'GPU work is active; checkpoint it instead'
parent=psutil.Process(22372)
children=parent.children(recursive=True)
assert all(process.name().lower() in ('python.exe','conhost.exe') for process in children)
targets=[process for process in children if process.name().lower()=='python.exe']+[parent]
for process in targets:
    command=' '.join(process.cmdline()).replace('\\','/')
    assert 's172_vis_coll/bench_supervise.py' in command,(process.pid,command)
assert (scratch/'gpu.request').exists() and not (scratch/'gpu_leases/video-s172.json').exists()
for process in targets: process.terminate()
gone,alive=psutil.wait_procs(targets,timeout=5)
assert not alive,[process.pid for process in alive]
print('Stopped idle controllers only:',[process.pid for process in gone])
