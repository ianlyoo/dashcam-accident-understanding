"""Break S109 fallback misses down by the exported/cached tracker reason."""
import collections,json,pathlib

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR')
rows=[json.loads(line) for line in (H/'rows.jsonl').read_text().splitlines()]
audit={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
s174={r['key']:r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
diagnostics={}
for file in sorted((D/'s178_side/qa_phases').glob('candidate_*.json')):
    batch=json.loads(file.read_text())
    diagnostics.update({key.replace('_','/',1):item
                        for key,item in batch['diagnostics']['clips'].items()})
fallback=collections.defaultdict(lambda:dict(n=0,hits=0,misses=0,examples=[]))
for row in rows:
    if row['source']!='s109_fallback':continue
    key=row['key']
    reason=((diagnostics.get(key,{}).get('result') or {}).get('reason')
            or (s174.get(key,{}).get('extension_result') or {}).get('reason')
            or audit.get(key,{}).get('reason') or 'unknown')
    item=fallback[reason];item['n']+=1
    lo,hi=row['interval'];t=row['entry']/row['fps']
    if lo-.30000001<=t<=hi+.30000001:item['hits']+=1
    else:item['misses']+=1;item['examples'].append(key)
assert sum(x['n'] for x in fallback.values())==91
assert sum(x['misses'] for x in fallback.values())==68
result=dict(sorted(fallback.items(),key=lambda kv:-kv[1]['misses']))
(H/'fallback_buckets.json').write_text(json.dumps(result,indent=2)+'\n')
largest=collections.Counter()
for row in rows:
    if row['source']!='s109_fallback' or row['key'] not in audit:continue
    key=row['key'];lo,hi=row['interval'];t=row['entry']/row['fps']
    if lo-.30000001<=t<=hi+.30000001:continue
    reason=(s174.get(key,{}).get('extension_result') or {}).get('reason')
    if reason=='s174_no_supported_crossing':largest[audit[key]['reason']]+=1
result['_s174_no_supported_crossing_original_reason_misses']=dict(largest)
(H/'fallback_buckets.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
