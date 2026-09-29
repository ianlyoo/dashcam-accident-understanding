"""Freeze paired metrics, annotation provenance and source identities in the repo."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate


def identity(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    records=evaluate.rows(evaluate.WORK/'evaluation.jsonl')
    report=evaluate.score(records)
    labels={(r['kind'],r['id']):r['labels'] for r in evaluate.dataset()}
    pairs={}
    for r in records:
        for lab in labels[(r['kind'],r['id'])]:
            lo,hi=lab['interval']
            hit=lambda e:int(lo-.3-1e-9 <= e/r['fps'] <= hi+.3+1e-9)
            pairs.setdefault(lab['source'],[]).append(hit(r['entry'])-hit(r['base_entry']))
    rng=np.random.default_rng(161)
    for name,delta in pairs.items():
        a=np.asarray(delta)
        sampled=rng.choice(a,size=(10000,len(a)),replace=True).mean(axis=1)
        report[name]['paired_bootstrap_95_delta_rate']=np.quantile(sampled,[.025,.975]).tolist()
    inputs=[evaluate.REPO/'candidates/s126_longclip/blind_labels_frozen.jsonl',
            evaluate.REPO/'tracking/s117_entry_annotations.json',
            evaluate.DATA/'s132_s2track/ccd_labels.json',
            evaluate.DATA/'s144_rr3/refiner/cascade_k2_oof.json',
            evaluate.REPO/'Baseline/data/stage2/labels.csv']
    inputs+=sorted((evaluate.DATA/'annot_s122').glob('labels_*.jsonl'))
    report['provenance']={str(p):identity(p) for p in inputs}
    report['coverage']=dict(predicted_unique_clips=len(records),available_unique_clips=len(list(evaluate.dataset())),
                            ccd=sum(r['kind']=='ccd' for r in records),nexar=sum(r['kind']=='nexar' for r in records))
    report['methodology']={
        'collision_anchors':'S156 cascade K=2 delta=.2 OOF on Nexar; unchanged S109 on short CCD',
        'annotations':'Reused AI proxy labels, not human/official ground truth; no fresh holdout claim',
        'parameters':'Fixed crossing confidence rule, no S161 entry fitting or parameter sweep',
        'public_official_entries':'Baseline stage2 labels have t_entry=-1 for all five rows; not scored',
        's117_unknowns':'00957,00769,00418 have no observable interval; excluded from timing accuracy',
        'overlap':'Annotator-specific CCD and overlapping S117/Nexar low-confidence rows reported separately; never pooled',
        'runtime':'Cold detection per clip, model-load excluded; 137 is a scenario, not known server workload'}
    if (evaluate.WORK/'metrics.json').exists():
        resource=json.load(open(evaluate.WORK/'metrics.json'))
        report['resources']={k:resource[k] for k in ['cuda_peak_bytes','memory_bytes'] if k in resource}
    report['source_sha256']={p.name:identity(p) for p in [HERE/'predict.py',HERE/'adapter.py',HERE/'params.json',HERE/'evaluate.py']}
    (HERE/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'paired_predictions.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['id','kind','fps','collision','base_entry','entry','seconds']} | {'reason':r['result']['reason'],'labels':labels[(r['kind'],r['id'])]})+'\n' for r in records))
    print(json.dumps(report['coverage']))


if __name__=='__main__':main()
