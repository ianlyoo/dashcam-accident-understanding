"""Stop only a verified, idle S174 evaluator admission supervisor."""
import pathlib,psutil
pid=42712
p=psutil.Process(pid)
cmd=' '.join(p.cmdline()).replace('\\','/').lower()
assert 'candidates/s174_cov/launch_gpu.py' in cmd and '--supervise' in cmd
assert '--job eval_candidate.py' in cmd,cmd
lease=pathlib.Path('$USER_HOME/Documents/code/shared_project/scratchpad/gpu_leases/video-s174.json')
if lease.exists():
 import json
 assert json.loads(lease.read_text())['pid']!=pid,'Do not stop an active GPU lease owner'
p.terminate();p.wait(timeout=10)
print('Stopped idle S174 evaluator supervisor',pid,'for labeled-audit prerequisite; active 750 GPU owner unaffected')
