"""S181 lawful S109-fallback audit and fixed, held-out entry alternatives."""
import collections,json,math,pathlib,statistics

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR')
rows=[r for r in map(json.loads,(R/'candidates/s180_resid/rows.jsonl').read_text().splitlines())
      if r['source']=='s109_fallback']
audit={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
s174={r['key']:r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
diagnostics={}
for file in sorted((D/'s178_side/qa_phases').glob('candidate_*.json')):
    batch=json.loads(file.read_text())
    diagnostics.update({key.replace('_','/',1):record
                        for key,record in batch['diagnostics']['clips'].items()})
assert len(rows)==91 and len(audit)==243
PANELS=('ccd_consensus_79','ccd_a3_low_45','nexar_confident_13','dkb_human_74')

def hit(row,frame):
    t=frame/row['fps'];lo,hi=row['interval']
    return lo-.30000001<=t<=hi+.30000001
def quantiles(values):
    ordered=sorted(values)
    if not ordered:return None
    return dict(n=len(ordered),min=round(ordered[0],3),q25=round(ordered[(len(ordered)-1)//4],3),
                median=round(statistics.median(ordered),3),mean=round(statistics.mean(ordered),3),
                q75=round(ordered[(3*(len(ordered)-1))//4],3),max=round(ordered[-1],3))
def paired(items,rule):
    before=sum(hit(r,r['entry']) for r in items)
    proposed=[(r,rule(r)) for r in items]
    after=sum(hit(r,frame) for r,frame in proposed)
    return dict(n=len(items),before=before,after=after,gain=after-before,
                changed=sum(frame!=r['entry'] for r,frame in proposed),
                repairs=[r['key'] for r,frame in proposed if not hit(r,r['entry']) and hit(r,frame)],
                breaks=[r['key'] for r,frame in proposed if hit(r,r['entry']) and not hit(r,frame)])

# This gap is a geometric quantity measured on S161's accepted crossings,
# without consulting the fallback labels being evaluated.
accepted={k:r for k,r in audit.items() if r['accepted'] and r['predicted'] is not None}
assert len(accepted)==52
def gap_values(exclude=frozenset()):
    return [(r['s171_collision']-r['predicted'])/r['fps'] for k,r in accepted.items() if k not in exclude]
global_gap=statistics.median(gap_values())
SHIPPED_REASONS=frozenset(('short_track','not_in_lane_at_collision',
                           'far_inside_when_first_seen'))
def prior(gap):
    def rule(row):
        frame=row['collision']-int(round(gap*row['fps']))
        return max(0,min(row['collision'],frame))
    return rule
def gated_prior(gap,reasons=SHIPPED_REASONS):
    proposal=prior(gap)
    return lambda row: proposal(row) if row['s161_reason'] in reasons else row['entry']

def original_trace(row):
    source=audit.get(row['key'])
    return (source.get('result') or {}).get('trace') if source else None
def first_touch(row):
    trace=original_trace(row)
    if not trace:return row['entry']
    outside=False
    for item in sorted(trace,key=lambda x:x[0]):
        index,value=item[:2]
        if value is None:continue
        if value<0:outside=True
        elif outside and index<=row['collision']:
            return int(index)
    return row['entry']

for row in rows:
    source=audit.get(row['key']);extension=s174.get(row['key'])
    record=diagnostics.get(row['key'],{})
    row['s161_reason']=(source['reason'] if source else
                         (record.get('result') or {}).get('reason') or 'not_cached')
    row['s174_reason']=((extension.get('extension_result') or {}).get('reason') if extension
                        else 'not_cached')
    row['label_to_collision_gap_s']=round(row['collision']/row['fps']-
                                           sum(row['interval'])/2,4)
    row['incumbent_hit']=hit(row,row['entry'])
    row['trace_state']='missing'
    trace=original_trace(row)
    if trace:
        signed=[p[1] for p in trace if p[1] is not None]
        if signed:
            row['trace_state']=('all_outside' if all(v<0 for v in signed) else
                                'all_inside' if all(v>=0 for v in signed) else
                                'mixed')
    row['first_touch']=first_touch(row)
    row['gap_prior_entry']=prior(global_gap)(row)

by_panel={panel:[r for r in rows if r['panel']==panel] for panel in PANELS}
prior_results={panel:paired(items,prior(global_gap)) for panel,items in by_panel.items()}
touch_results={panel:paired(items,first_touch) for panel,items in by_panel.items()}
gated_results={panel:paired(items,gated_prior(global_gap)) for panel,items in by_panel.items()}
folds={}
for held in PANELS:
    held_keys={r['key'] for r in by_panel[held]}
    gap=statistics.median(gap_values(held_keys))
    train=[r for r in rows if r['panel']!=held and r['key'] not in held_keys]
    train_gains={reason:paired([r for r in train if r['s161_reason']==reason],prior(gap))['gain']
                 for reason in sorted({r['s161_reason'] for r in train})}
    selected=frozenset(reason for reason,gain in train_gains.items() if gain>0)
    folds[held]=dict(train_crossings=len(accepted)-sum(k in held_keys for k in accepted),
                     gap_s=round(gap,4),prior=paired(by_panel[held],prior(gap)),
                     gated_prior=paired(by_panel[held],gated_prior(gap)),
                     train_fallback_rows=len(train),train_reason_gains=train_gains,
                     train_selected_reasons=sorted(selected),
                     learned_gated_prior=paired(by_panel[held],gated_prior(gap,selected)),
                     first_touch=paired(by_panel[held],first_touch),
                     excluded_overlap_keys=len(held_keys&{r['key'] for r in rows if r['panel']!=held}))

def reason_counter(field,items):
    groups=collections.defaultdict(list)
    for row in items:groups[row[field]].append(row)
    return {k:dict(n=len(v),hits=sum(x['incumbent_hit'] for x in v),
                   misses=sum(not x['incumbent_hit'] for x in v),
                   label_gap_s=quantiles([x['label_to_collision_gap_s'] for x in v]))
            for k,v in sorted(groups.items(),key=lambda kv:-len(kv[1]))}

misses=[r for r in rows if not r['incumbent_hit']]
summary=dict(n=len(rows),unique_keys=len({r['key'] for r in rows}),misses=len(misses),
             accepted_s161_gaps_s=quantiles(gap_values()),global_gap_s=round(global_gap,4),
             label_to_collision_gap_s=quantiles([r['label_to_collision_gap_s'] for r in rows]),
             missed_label_to_collision_gap_s=quantiles([r['label_to_collision_gap_s'] for r in misses]),
             collision_before_labeled_entry=sum(r['collision']/r['fps']<r['interval'][0]-.30000001 for r in rows),
             collision_before_labeled_entry_misses=sum(r['collision']/r['fps']<r['interval'][0]-.30000001 for r in misses),
             s161_reasons=reason_counter('s161_reason',rows),
             s174_reasons=reason_counter('s174_reason',rows),
             trace_states=reason_counter('trace_state',rows),
             prior_pooled=paired(rows,prior(global_gap)),prior_panels=prior_results,
             gated_prior_pooled=paired(rows,gated_prior(global_gap)),
             gated_prior_panels=gated_results,shipped_reasons=sorted(SHIPPED_REASONS),
             touch_pooled=paired(rows,first_touch),touch_panels=touch_results,
             folds=folds,caveat=('CCD and Nexar labels are AI-made; DKB human. '
                                'Cached S161/S174 traces exist for 243 prior clips; '
                                'additional DKB clips have exported final reasons but '
                                'not reconstructed original traces. Panel IDs overlap.'))
(H/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(H/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('s161_reasons','s174_reasons','trace_states')},indent=2))
print('REASONS',json.dumps({k:summary[k] for k in ('s161_reasons','s174_reasons','trace_states')},indent=2))
