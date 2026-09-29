"""Public paired side/evasion trace study with fixed official class F1.

S172 changes collision only. Its side/evasion outputs are S109's, as retained
by S171/S174/S176. The 240-CCD S109 CSV and 31-Nexar public exported rows are
used as incumbent predictions. Inference rows for other human-labeled clips
were not available and are reported as uncovered, never scored as negatives.
"""
import collections
import csv
import json
import pathlib
import sys

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');sys.path.insert(0,str(R/'src'))
from videohackathon.evaluation import macro_f1

audit={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
old={r['kind']+'/'+r['id']:r for r in map(json.loads,(D/'s161_entry_cross/evaluation.jsonl').read_text().splitlines())}
s174={r['key']:r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
assert len(audit)==len(old)==len(s174)==243
s174_side={r['key']:r['side'] for r in map(json.loads,(H/'s174_accepted_side.jsonl').read_text().splitlines())}
assert len(s174_side)==24 and set(s174_side)=={k for k,r in s174.items() if r['extension_accepted']}
with (D/'s2eval/s109_full_240.csv').open(newline='') as f:
    incumbent={'ccd/'+r['ID']:dict(side=r['entry_side'],evasion=int(r['evasion_space'])) for r in csv.DictReader(f)}
for r in map(json.loads,(D/'s126_longclip/entry_base_rows.jsonl').read_text().splitlines()):
    incumbent['nexar/'+r['id']]=dict(side=r['s109']['entry_side'],evasion=int(r['s109']['evasion_space']))
assert len(incumbent)==271

def iou(a,b):
    if not a or not b:return 0.
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[2],b[2]);bottom=min(a[3],b[3])
    inter=max(0.,right-left)*max(0.,bottom-top)
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0.

def crossing_side(row):
    if not row or not row['accepted'] or row['reason']!='crossing':return None
    trace=row['result'].get('trace') or []
    bracket=row['result'].get('bracket') or []
    if len(bracket)!=2:return None
    outside=[t for t in trace if t[0]==bracket[0] and t[2] in ('LEFT','RIGHT')]
    return outside[-1][2] if outside else None

BUCKETS=('inside_from_start','inside_when_first_seen')
rows={}
for key,base in incumbent.items():
    a=audit.get(key);o=old.get(key);ex=s174.get(key)
    s161_cross=crossing_side(a)
    fallback_cross=s174_side.get(key) if ex and ex['extension_accepted'] and not s161_cross else None
    exact_cross=s161_cross or fallback_cross
    anchor_iou=iou(a['result'].get('anchor_box'),o['result'].get('anchor_box')) if a and o else 0.
    aligned=bool(a and o and a['reason']==o['result']['reason'] and anchor_iou>=.60)
    origin=(o['result'].get('side') if aligned and a['reason'] in BUCKETS and
            ex and not ex['extension_accepted'] else None)
    if origin not in ('LEFT','RIGHT'):origin=None
    anchor=(a['result'].get('anchor_box') if a else None)
    cx=(anchor[0]+anchor[2])/2 if anchor else None
    center={str(t):('LEFT' if cx<=.5-t else 'RIGHT' if cx>=.5+t else None)
            if cx is not None and a and a['reason'] in BUCKETS and ex and not ex['extension_accepted'] else None
            for t in (.1,.15,.2)}
    ev=(o['result'].get('evasion') if aligned and
        ((a['accepted'] and a['reason']=='crossing') or (a['reason'] in BUCKETS and ex and not ex['extension_accepted'])) else None)
    if ev not in (0,1):ev=None
    rows[key]=dict(base=base,trace=bool(a),reason=a['reason'] if a else None,
                   accepted=a['accepted'] if a else False,s174_accepted=ex['extension_accepted'] if ex else False,
                   cross=exact_cross,s161_cross=s161_cross,s174_cross=fallback_cross,
                   origin=origin,anchor_iou=round(anchor_iou,3),
                   anchor_center=round(cx,3) if cx is not None else None,
                   center=center,evasion=ev)

panels={}
ccd=json.loads((D/'s132_s2track/ccd_labels.json').read_text())
panels['ccd_consensus_side']={'ccd/'+r['id']:r['side'] for r in ccd if r['side'] in ('LEFT','RIGHT')}
panels['ccd_consensus_evasion']={'ccd/'+r['id']:int(r['evasion']) for r in ccd if r['evasion'] in (0,1)}
for n in (1,2,3):
    raw=[json.loads(s) for s in (D/f'annot_s122/labels_A{n}.jsonl').read_text().splitlines()]
    for conf in ('confident','low'):
        keep=('medium','high') if conf=='confident' else ('low',)
        panels[f'ccd_A{n}_{conf}_side']={'ccd/'+r['id']:r['entry_side'] for r in raw
                                         if r['side_conf'] in keep and r['entry_side'] in ('LEFT','RIGHT')}
        panels[f'ccd_A{n}_{conf}_evasion']={'ccd/'+r['id']:int(r['evasion_space']) for r in raw
                                            if r['evasion_conf'] in keep and r['evasion_space'] in (0,1)}
blind=[json.loads(s) for s in (R/'candidates/s126_longclip/blind_labels_frozen.jsonl').read_text().splitlines()]
for conf in ('confident','low'):
    keep=('medium','high') if conf=='confident' else ('low',)
    panels[f'nexar_{conf}_side']={'nexar/'+r['id']:r['side'] for r in blind
                                  if r['side_confidence'] in keep and r['side'] in ('LEFT','RIGHT')}
    panels[f'nexar_{conf}_evasion']={'nexar/'+r['id']:int(r['evasion']) for r in blind
                                     if r['evasion_confidence'] in keep and r['evasion'] in (0,1)}
with (D/'external_meta/repos/DKB-2000_CrashIntent-AI/data/stage2/labels_manual.csv').open(newline='') as f:
    raw=list(csv.DictReader(f))
panels['dkb_human_side']={'ccd/'+r['ID']:r['entry_side'] for r in raw if r['entry_side'] in ('LEFT','RIGHT')}
panels['dkb_human_evasion']={'ccd/'+r['ID']:int(r['evasion_space']) for r in raw if r['evasion_space'] in ('0','1')}
with (D/'external_meta/repos/Nonmaju_dashcam-accident-analysis_personal_repo/Baseline/external/jungmin_labels/legacy_labels.csv').open(newline='') as f:
    raw=list(csv.DictReader(f))
panels['jungmin_human_side']={'ccd/'+r['video_id']:r['entry_side'] for r in raw if r['review_status']=='DONE' and r['entry_side'] in ('LEFT','RIGHT')}
panels['jungmin_human_evasion']={'ccd/'+r['video_id']:int(r['evasion_space']) for r in raw if r['review_status']=='DONE' and r['evasion_space'] in ('0','1')}

def proposed(row,name):
    base=row['base']
    if name=='incumbent':return base['side']
    if name=='s161_only':return row['s161_cross'] or base['side']
    if name=='s161_center_0.15':return row['s161_cross'] or row['center']['0.15'] or base['side']
    if name=='cross_only':return row['cross'] or base['side']
    if name=='cross_origin':return row['cross'] or row['origin'] or base['side']
    if name.startswith('cross_center_'):
        threshold=name.rsplit('_',1)[-1]
        return row['cross'] or row['center'][threshold] or base['side']
    if name=='evasion':return row['evasion'] if row['evasion'] is not None else base['evasion']
    raise KeyError(name)

def measure(labels,name,kind):
    keys=sorted(set(labels)&set(rows))
    lab=[labels[k] for k in keys]
    before=[rows[k]['base'][kind] for k in keys]
    after=[proposed(rows[k],name) for k in keys]
    universe=(0,1) if kind=='evasion' else ('LEFT','RIGHT')
    changed=[k for k,a,b in zip(keys,before,after) if a!=b]
    repairs=[k for k,a,b,t in zip(keys,before,after,lab) if a!=t and b==t]
    breaks=[k for k,a,b,t in zip(keys,before,after,lab) if a==t and b!=t]
    perclass={str(c):dict(n=sum(t==c for t in lab),
                          incumbent_f1=round(float(macro_f1(lab,before,(c,))),5),
                          candidate_f1=round(float(macro_f1(lab,after,(c,))),5),
                          incumbent_correct=sum(t==c and p==c for t,p in zip(lab,before)),
                          candidate_correct=sum(t==c and p==c for t,p in zip(lab,after))) for c in universe}
    return dict(labeled=len(labels),measured=len(keys),trace_coverage=sum(rows[k]['trace'] for k in keys),
                incumbent_f1=round(float(macro_f1(lab,before,universe)),5),
                candidate_f1=round(float(macro_f1(lab,after,universe)),5),
                incumbent_correct=sum(a==t for a,t in zip(before,lab)),
                candidate_correct=sum(a==t for a,t in zip(after,lab)),
                changed=len(changed),repairs=repairs,breaks=breaks,per_class=perclass)

summary=dict(side={},evasion={},coverage=dict(incumbent=len(incumbent),audit=len(audit),
             crossing=sum(bool(r['cross']) for r in rows.values()),
             s161_crossing=sum(bool(r['s161_cross']) for r in rows.values()),
             s174_crossing=sum(bool(r['s174_cross']) for r in rows.values()),
             origin=sum(bool(r['origin']) for r in rows.values()),
             center={t:sum(bool(r['center'][t]) for r in rows.values()) for t in ('0.1','0.15','0.2')},
             evasion=sum(r['evasion'] is not None for r in rows.values())))
side_policies=('incumbent','s161_only','s161_center_0.15',
               'cross_only','cross_origin','cross_center_0.1','cross_center_0.15','cross_center_0.2')
for name,labels in panels.items():
    if name.endswith('_side'):
        summary['side'][name]={policy:measure(labels,policy,'side') for policy in side_policies}
    else:
        summary['evasion'][name]={policy:measure(labels,policy,'evasion') for policy in ('incumbent','evasion')}

(H/'study_rows.jsonl').write_text(''.join(json.dumps(dict(key=k,**v))+'\n' for k,v in rows.items()))
(H/'study.json').write_text(json.dumps(summary,indent=2)+'\n')
for name in ('ccd_consensus_side','ccd_A1_confident_side','ccd_A2_confident_side',
             'ccd_A3_confident_side','nexar_confident_side','dkb_human_side','jungmin_human_side'):
    print(name,[(p,x['measured'],x['incumbent_f1'],x['candidate_f1'],x['changed'],len(x['repairs']),len(x['breaks']))
                for p,x in summary['side'][name].items()])
print('coverage',summary['coverage'])
