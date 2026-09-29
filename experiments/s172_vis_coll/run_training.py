"""Wait for complete resumable cache, then a bounded CPU OOF job."""
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
WORK=Path('$DATA_DIR/s172_vis_coll')
PYTHON='./.venv/Scripts/python.exe'
if __name__=='__main__':
    deadline=time.monotonic()+6*3600
    targets=[WORK/'coarse'/f'{row["id"]}.npz' for row in json.loads((HERE/'inputs.json').read_text())]
    while time.monotonic()<deadline:
        count=sum(path.is_file() for path in targets)
        if count==750: break
        print(json.dumps(dict(event='cache_wait',complete=count,total=750)),flush=True)
        time.sleep(30)
    else:
        raise TimeoutError('No complete extraction cache in six hours')
    result=subprocess.run([PYTHON,'-B',str(HERE/'train.py'),'--device','cpu'],timeout=5*3600,check=False)
    print(json.dumps(dict(event='train_exit',code=result.returncode)),flush=True)
    raise SystemExit(result.returncode)
