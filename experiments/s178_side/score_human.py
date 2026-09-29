"""Score full public human CCD side panels after exported S178 CUDA QA."""
import csv,json,pathlib,sys

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];D=pathlib.Path('$DATA_DIR')
sys.path.insert(0,str(R/'src'))
from videohackathon.evaluation import macro_f1
study={r['key']:r for r in map(json.loads,(H/'study_rows.jsonl').read_text().splitlines())}
qa={r['ID'].replace('_','/',1):r for r in map(json.loads,(H/'qa_rows.jsonl').read_text().splitlines())}
with (D/'external_meta/repos/DKB-2000_CrashIntent-AI/data/stage2/labels_manual.csv').open(newline='') as f:
    dkb={'ccd/'+r['ID']:r['entry_side'] for r in csv.DictReader(f) if r['entry_side'] in ('LEFT','RIGHT')}
with (D/'external_meta/repos/Nonmaju_dashcam-accident-analysis_personal_repo/Baseline/external/jungmin_labels/legacy_labels.csv').open(newline='') as f:
    jung={'ccd/'+r['video_id']:r['entry_side'] for r in csv.DictReader(f)
          if r['review_status']=='DONE' and r['entry_side'] in ('LEFT','RIGHT')}
assert len(dkb)==74 and len(jung)==36
def proxy(row):
    return row['s161_cross'] or row['center']['0.15'] or row['base']['side']
parity=[]
for key,row in qa.items():
    if key in study:
        old=study[key]
        if old['base']['side']!=row['base_side'] or proxy(old)!=row['candidate_side']:
            parity.append(dict(key=key,cached_before=old['base']['side'],exported_before=row['base_side'],
                               cached_candidate=proxy(old),exported_candidate=row['candidate_side']))
def prediction(key):
    if key in qa:
        row=qa[key];return row['base_side'],row['candidate_side'],'exported'
    row=study[key];return row['base']['side'],proxy(row),'cached_trace'
def score(labels):
    keys=sorted(labels)
    assert all(k in qa or k in study for k in keys)
    y=[labels[k] for k in keys]
    pairs=[prediction(k) for k in keys]
    before=[p[0] for p in pairs];after=[p[1] for p in pairs]
    changed=[k for k,b,a in zip(keys,before,after) if b!=a]
    repairs=[k for k,t,b,a in zip(keys,y,before,after) if b!=t and a==t]
    breaks=[k for k,t,b,a in zip(keys,y,before,after) if b==t and a!=t]
    classes={}
    for c in ('LEFT','RIGHT'):
        classes[c]=dict(n=sum(t==c for t in y),
                        incumbent_f1=round(float(macro_f1(y,before,(c,))),5),
                        candidate_f1=round(float(macro_f1(y,after,(c,))),5),
                        incumbent_correct=sum(t==c and b==c for t,b in zip(y,before)),
                        candidate_correct=sum(t==c and a==c for t,a in zip(y,after)))
    return dict(n=len(keys),exported=sum(p[2]=='exported' for p in pairs),
                incumbent_f1=round(float(macro_f1(y,before,('LEFT','RIGHT'))),5),
                candidate_f1=round(float(macro_f1(y,after,('LEFT','RIGHT'))),5),
                incumbent_correct=sum(t==b for t,b in zip(y,before)),
                candidate_correct=sum(t==a for t,a in zip(y,after)),
                changed=changed,repairs=repairs,breaks=breaks,per_class=classes)
report=dict(dkb=score(dkb),jungmin=score(jung),selected_export_parity_mismatches=parity,
            caveat='Human labels are from external public repos and overlap each other. '
                   'Most older CCD predictions are cached-detector replay; the 76 previously uncovered '
                   'clips and selected QA cases are actual exported CUDA outputs.')
(H/'human.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ('dkb','jungmin'):
    v=report[name]
    print(name,v['n'],v['exported'],v['incumbent_f1'],v['candidate_f1'],
          len(v['changed']),len(v['repairs']),len(v['breaks']))
print('selected QA parity mismatches',len(parity),parity)
