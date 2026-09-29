"""Check S181 on all other retained AI label views without pooling them."""
import collections,json,pathlib

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR')
paired=[json.loads(line) for line in (R/'candidates/s176_bucket/paired_rows.jsonl').read_text().splitlines()]
audit={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
s174={r['key']:r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
eligible=frozenset(('short_track','not_in_lane_at_collision','far_inside_when_first_seen'))
views=collections.defaultdict(lambda:dict(n=0,before=0,after=0,repairs=[],breaks=[]))
selected=0
for row in paired:
    key=row['key']
    if row['rule'] or audit[key]['accepted'] or s174[key]['extension_accepted']:
        continue
    if not row['labels']:continue
    selected+=1
    prior=max(0,row['collision']-int(round(.85*row['fps'])))
    after=prior if audit[key]['reason'] in eligible else row['s176_entry']
    for label in row['labels']:
        view=views[label['source']];view['n']+=1
        lo,hi=label['interval']
        old_hit=lo-.30000001<=row['s176_entry']/row['fps']<=hi+.30000001
        new_hit=lo-.30000001<=after/row['fps']<=hi+.30000001
        view['before']+=old_hit;view['after']+=new_hit
        if new_hit and not old_hit:view['repairs'].append(key)
        if old_hit and not new_hit:view['breaks'].append(key)
result=dict(selected_clips=selected,views=dict(sorted(views.items())))
(H/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
