"""Bounded read-only S174 activation/progress summary."""
import collections,json,pathlib,statistics
H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR/s174_cov')
for key,path in (('labeled',H/'audit_labeled.jsonl'),('nexar750',D/'audit_750.jsonl')):
 if not path.exists():continue
 rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
 reasons=collections.Counter(r['reason'] for r in rows)
 accepted=sum(r['accepted'] for r in rows)
 misses=sum(r.get('detector_cache_misses',0) for r in rows)
 print(key,'rows',len(rows),'accepted',accepted,'misses',misses,'reasons',dict(reasons))
 if key=='labeled':
  print('by_kind',{k:(sum(r['kind']==k for r in rows),sum(r['accepted'] for r in rows if r['kind']==k)) for k in ('ccd','nexar')})
  print('accepted_mismatch',[r['key'] for r in rows if r['accepted'] and not r['matches_s171']][:30])
  print('poor_horizon',sum(r['result'].get('horizon_n',0)<5 for r in rows if r['result'].get('horizon_n') is not None))
  print('crossing_rejected',[(r['key'],r['result'].get('bracket'),r['result'].get('inside_samples'),r['result'].get('outside_samples')) for r in rows if r['reason']=='crossing' and not r['accepted']][:12])
for key in ('build.json','anchors_receipt.json','audit_labeled_summary.json','audit_750_summary.json'):
 p=H/key;print(key,'ready' if p.exists() else 'pending')
