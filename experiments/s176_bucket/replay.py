"""Pair fixed winning rules with exact S174 output rows across public labels."""
import collections
import hashlib
import json
import pathlib

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR')
current=[json.loads(s) for s in (D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines() if s.strip()]
measured={r['key']:r for r in map(json.loads,(H/'measure_rows.jsonl').read_text().splitlines())}
assert len(current)==243 and len(measured)==96
rules={'inside_from_start':'first_clip','inside_when_first_seen':'minus_0p2s'}
records=[]
for r in current:
    m=measured.get(r['key']);rule=rules.get(m['bucket']) if m else None
    if m:assert m['eligible']==(not r['extension_accepted'])
    new=int(m['candidates'][rule]) if rule and m['eligible'] else int(r['s174_entry'])
    assert new<=r['s171_collision']
    records.append(dict(key=r['key'],kind=r['kind'],s171_entry=r['s171_entry'],s174_entry=r['s174_entry'],
                        s176_entry=new,collision=r['s171_collision'],fps=r['fps'],labels=r['labels'],
                        bucket=m['bucket'] if m else None,rule=rule if m and m['eligible'] else None))

def hit(frame,fps,label):
    lo,hi=label['interval'];return lo-.30000001<=frame/fps<=hi+.30000001

sources={}
for r in records:
    for label in r['labels']:
        source=label['source'];s=sources.setdefault(source,dict(n=0,s171=0,s174=0,s176=0,
                                                      repairs=[],breaks=[],changed=0))
        a=hit(r['s174_entry'],r['fps'],label);b=hit(r['s176_entry'],r['fps'],label)
        s['n']+=1;s['s171']+=hit(r['s171_entry'],r['fps'],label);s['s174']+=a;s['s176']+=b
        s['changed']+=r['s174_entry']!=r['s176_entry']
        if b and not a:s['repairs'].append(r['key'])
        if a and not b:s['breaks'].append(r['key'])
result=dict(n=len(records),changed=sum(r['s174_entry']!=r['s176_entry'] for r in records),
            changed_by_bucket=dict(collections.Counter(r['bucket'] for r in records if r['s174_entry']!=r['s176_entry'])),
            sources=sources,rules=rules,
            base_sha256=json.loads((H/'build.json').read_text())['base_sha256'],
            build_sha256=hashlib.sha256((H/'build.json').read_bytes()).hexdigest(),
            caveat='Exact exported S171 entries plus cached-detector S174 replay; S176 rule outcomes inferred from the same S161 traces, not a full exported 243-clip candidate run.')
(H/'paired_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(H/'paired.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
for k in ('ccd_consensus','ccd_labels_A3_low','nexar_blind_confident','s117_interval'):
    print(k,sources[k]['n'],sources[k]['s174'],sources[k]['s176'],
          len(sources[k]['repairs']),len(sources[k]['breaks']))
