"""Bind the complete public feature cache and summarize measured extraction cost."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; WORK=Path('$DATA_DIR/s172_vis_coll')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''): h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    rows=json.loads((HERE/'inputs.json').read_text()); records=[]
    for row in rows:
        path=WORK/'coarse'/f'{row["id"]}.npz'
        with np.load(path) as z:
            expected=np.unique(np.minimum(np.rint(np.arange(0,row['n']/row['fps'],.25)*row['fps']).astype(np.int32),row['n']-1))
            assert np.array_equal(z['frames'],expected)
            assert z['features'].shape==(len(expected),2048) and np.isfinite(z['features']).all()
            for key in ('fold','fps','toe','n'): assert float(z[key])==row[key]
        records.append(dict(id=row['id'],samples=len(expected),sha256=sha(path),bytes=path.stat().st_size))
    timing=[json.loads(line) for line in (WORK/'timings.jsonl').read_text().splitlines()]
    assert len(timing)==750 and {r['id'] for r in timing}=={r['id'] for r in rows}
    seconds=np.array([r['seconds'] for r in timing])
    result=dict(clips=750,frames=sum(r['samples'] for r in records),records=records,
        extraction_source_sha256=sha(HERE/'extract.py'),inputs_sha256=sha(HERE/'inputs.json'),
        mean_seconds=float(seconds.mean()),median_seconds=float(np.median(seconds)),p95_seconds=float(np.quantile(seconds,.95)),
        max_seconds=float(seconds.max()),total_seconds=float(seconds.sum()),
        max_rss_gib=max(r['rss_gib'] for r in timing),min_free_commit_gib=min(r['free_commit_gib'] for r in timing),
        max_cuda_reserved_gib=max(r['cuda_reserved_gib'] for r in timing),
        caveat='Windows MP4 extraction, shared GPU load, no trained head or refinement; not exported folder runtime.')
    (HERE/'feature_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
