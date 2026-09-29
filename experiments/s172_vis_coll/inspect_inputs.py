import ctypes
import hashlib
import json
from pathlib import Path
import sys

DATA = Path('$DATA_DIR')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (k, ctypes.c_ulonglong) for k in ('physical', 'free_physical', 'commit', 'free_commit', 'virtual', 'free_virtual', 'extended')]

if __name__ == '__main__':
    import numpy as np
    import torch
    import transformers
    import psutil
    mem = Memory(); mem.length = ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.cuda.is_available(), free_commit_gib=mem.free_commit/2**30)), flush=True)
    ids = np.load(DATA/'s160_coll_loc/probs_nn_soft1.npz')['ids'].tolist()
    meta=[]
    for ident in ids:
        with np.load(DATA/'s160_coll_loc/rows'/f'{ident}.npz') as z:
            meta.append(dict(id=ident, fold=int(z['fold']), fps=float(z['fps']), n=int(z['n']), toe=float(z['toe'])))
    print(json.dumps(dict(clips=len(meta), first=meta[:2], folds={k:sum(r['fold']==k for r in meta) for k in range(5)},
        min_seconds=min(r['n']/r['fps'] for r in meta), max_seconds=max(r['n']/r['fps'] for r in meta),
        all_public_paths_exist=all((DATA/'nexar_collision/train/positive'/f'{r["id"]}.mp4').exists() for r in meta))), flush=True)
    for name in ['releases/S171_ent.zip', 'pretrained/dinov2-large-47b73eef/model.safetensors']:
        p=DATA/name
        print(json.dumps(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))), flush=True)
    (Path(__file__).parent/'inputs.json').write_text(json.dumps(meta, indent=2)+'\n')
