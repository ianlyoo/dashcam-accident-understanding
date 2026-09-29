"""Optional local GPU allocation gate for historical experiment replay."""
import atexit
import datetime
import json
import os
import time
from pathlib import Path

SCR = Path(os.environ.get('GPU_RESOURCE_DIR', '.'))
LEASES = SCR / 'gpu_leases'


def _alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True


def _total():
    tot = 0.0
    for p in LEASES.glob('*.json'):
        try:
            d = json.loads(p.read_text())
            if _alive(d.get('pid', 0)):
                tot += float(d.get('cap_gb', 0))
        except Exception:
            pass
    return tot


def acquire(job, cap_gb=1.0, ram_gb=3.0):
    me = LEASES / ('video-%s.json' % job)
    while True:
        if not (SCR / 'gpu.lock').exists() and not (SCR / 'gpu.request').exists() and _total() + cap_gb <= 14.0:
            me.write_text(json.dumps({'owner': 'experiment worker', 'project': 'video', 'job': job, 'pid': os.getpid(),
                                      'cap_gb': cap_gb, 'ram_gb': ram_gb,
                                      'start': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'end': 'on exit',
                                      'checkpoint': 'per fold/seed'}))
            atexit.register(lambda: me.unlink(missing_ok=True))
            return me
        print(json.dumps({'event': 'lease_wait', 'total': _total()}), flush=True)
        time.sleep(60)


def wait_request():
    while (SCR / 'gpu.request').exists():
        print(json.dumps({'event': 'gpu_request_pause'}), flush=True)
        time.sleep(60)
