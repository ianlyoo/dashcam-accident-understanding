"""Paired label study for fixed, geometry-based S161 abstention rules.

The primary label is one per clip, selected by documented source precedence.
All original source views are also reported without pooling their denominators.
No candidate ZIP or private evaluation input is read here.
"""
import collections
import json
import math
import pathlib

H = pathlib.Path(__file__).resolve().parent
ROOT = H.parents[1]
D = pathlib.Path('$DATA_DIR')
audit = [json.loads(line) for line in (ROOT/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines()]
evals = {r['key']: r for r in map(json.loads, (D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
assert len(audit) == len(evals) == 243

BUCKETS = ('inside_from_start', 'inside_when_first_seen',
           'far_inside_when_first_seen', 'not_in_lane_at_collision')
SOURCE_ORDER = ('ccd_consensus', 'ccd_labels_A3_confident', 'ccd_labels_A2_confident',
                'ccd_labels_A1_confident', 'nexar_blind_confident', 's117_interval',
                'ccd_labels_A3_low', 'ccd_labels_A2_low', 'ccd_labels_A1_low',
                'nexar_blind_low')
RULES = ('first_clip', 'first_inside', 'minus_0p2s', 'minus_0p5s', 'extrapolate')

def hit(entry, fps, label):
    lo, hi = label['interval']
    return lo - .30000001 <= entry / fps <= hi + .30000001

def first_inside(trace):
    for i, penetration, side in trace:
        if side is None and penetration >= 0:
            return int(i)
    return None

def extrapolate(trace, first):
    """Back-project the first three inside penetration samples to zero.

    Only a positive inward trend identifies a preceding boundary crossing.
    Otherwise return the first inside observation rather than inventing one.
    """
    inside = [(int(i), float(v)) for i, v, side in trace
              if side is None and v >= 0 and int(i) >= first][:3]
    if len(inside) < 2:
        return first
    x = [p[0] for p in inside]
    y = [p[1] for p in inside]
    xbar, ybar = sum(x)/len(x), sum(y)/len(y)
    denom = sum((v-xbar)**2 for v in x)
    slope = sum((i-xbar)*(v-ybar) for i, v in inside)/denom if denom else 0
    if not math.isfinite(slope) or slope <= 0.002:
        return first
    crossing = first - y[0]/slope
    return max(0, min(first, int(round(crossing))))

records = []
for row in audit:
    if row['reason'] not in BUCKETS:
        continue
    paired = evals[row['key']]
    folder = pathlib.Path(row['folder'].replace('/mnt/d/', '$DATA_DIR/'))
    paths = sorted(folder.glob('*.jpg'))
    assert len(paths) == row['n'] and paths, row['key']
    numbers = [int(p.stem.split('_')[-1]) for p in paths]
    trace = row['result'].get('trace') or []
    first = first_inside(trace)
    eligible = paired['s171_entry'] == paired['s174_entry']
    if row['reason'].startswith('inside') or row['reason'].startswith('far_inside'):
        assert first is not None, row['key']
    candidates = {}
    for rule in RULES:
        if rule == 'first_clip': index = 0
        elif first is None: index = None
        elif rule == 'first_inside': index = first
        elif rule == 'minus_0p2s': index = max(0, first - round(.2*row['fps']))
        elif rule == 'minus_0p5s': index = max(0, first - round(.5*row['fps']))
        else: index = extrapolate(trace, first)
        candidates[rule] = min(int(numbers[index]), int(row['s171_collision'])) if index is not None else None
    labels = row['labels']
    priority = min(labels, key=lambda lab: SOURCE_ORDER.index(lab['source']))
    records.append(dict(key=row['key'], bucket=row['reason'], eligible=eligible,
                        s171_entry=paired['s171_entry'], s174_entry=paired['s174_entry'],
                        collision=paired['s171_collision'], fps=row['fps'],
                        n=row['n'], first_inside_index=first,
                        first_inside_frame=numbers[first] if first is not None else None,
                        trace=trace, labels=labels, primary=priority['source'],
                        candidates=candidates))

def tally(items, rule, source=None):
    subset = [r for r in items if source is None or any(l['source'] == source for l in r['labels'])]
    base171 = base174 = gain = loss = proposed = evaluable = 0
    repairs, breaks = [], []
    for r in subset:
        lab = next(l for l in r['labels'] if l['source'] == source) if source else next(l for l in r['labels'] if l['source'] == r['primary'])
        a = hit(r['s171_entry'], r['fps'], lab)
        b = hit(r['s174_entry'], r['fps'], lab)
        base171 += a; base174 += b
        frame = r['candidates'][rule]
        if frame is None:frame = r['s174_entry']
        else:evaluable += 1
        c = hit(frame, r['fps'], lab)
        proposed += c
        gain += c and not b
        loss += b and not c
        if c and not b:repairs.append(r['key'])
        if b and not c:breaks.append(r['key'])
    return dict(n=len(subset), s171=base171, s174=base174, candidate=proposed,
                repairs=gain, breaks=loss, evaluable=evaluable,
                repair_ids=repairs, break_ids=breaks)

summary = dict(n=len(records), source_precedence=SOURCE_ORDER, rules=RULES,
               buckets={}, labels='Public CCD/Nexar reused proxies, no private evaluation data')
for bucket in BUCKETS:
    allrows = [r for r in records if r['bucket'] == bucket]
    eligible = [r for r in allrows if r['eligible']]
    summary['buckets'][bucket] = dict(n=len(allrows), eligible=len(eligible),
        all={rule: tally(allrows, rule) for rule in RULES},
        eligible_primary={rule: tally(eligible, rule) for rule in RULES},
        eligible_sources={source: {rule: tally(eligible, rule, source) for rule in RULES}
                          for source in SOURCE_ORDER if any(l['source'] == source for r in eligible for l in r['labels'])})

(H/'measure_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(H/'measure.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({k: dict(n=v['n'], eligible=v['eligible'],
                          rules={rule: {kk: vv for kk, vv in result.items() if kk not in ('repair_ids','break_ids')}
                                 for rule, result in v['eligible_primary'].items()})
                  for k, v in summary['buckets'].items()}, indent=2))
