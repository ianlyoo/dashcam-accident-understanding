"""S178 entry residuals and leak-resistant panel-held-out shift study."""
import argparse,collections,csv,json,math,pathlib,statistics

parser=argparse.ArgumentParser()
parser.add_argument('--allow-missing',action='store_true')
args=parser.parse_args()

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR')
paired={r['key']:r for r in map(json.loads,(R/'candidates/s176_bucket/paired_rows.jsonl').read_text().splitlines())}
audit={r['key']:r for r in map(json.loads,(R/'candidates/s174_cov/audit_labeled.jsonl').read_text().splitlines())}
s174={r['key']:r for r in map(json.loads,(D/'s174_cov/evaluation_gpu.jsonl').read_text().splitlines())}
assert len(paired)==len(audit)==len(s174)==243
qa={r['ID'].replace('_','/',1):r for r in map(json.loads,(R/'candidates/s178_side/qa_rows.jsonl').read_text().splitlines())}
missing_export=H/'export_missing.jsonl'
if missing_export.exists():
    for item in map(json.loads,missing_export.read_text().splitlines()):
        qa[item['key']]=dict(entry_frame=item['entry_frame'],collision_frame=item['collision_frame'])
diagnostics={}
for file in sorted((D/'s178_side/qa_phases').glob('candidate_*.json')):
    batch=json.loads(file.read_text())
    for ident,result in batch['diagnostics']['clips'].items():
        diagnostics[ident.replace('_','/',1)]=result
assert len(diagnostics)==84

SOURCES=('s161_crossing','s174_crossing','s176_first_frame',
         's176_first_obs_minus_0p2','s109_fallback')
PANELS=('ccd_consensus_79','ccd_a3_low_45','nexar_confident_13','dkb_human_74')
def cached_source(key):
    row=paired[key]
    if row['rule']=='first_clip':return 's176_first_frame'
    if row['rule']=='minus_0p2s':return 's176_first_obs_minus_0p2'
    if audit[key]['accepted']:return 's161_crossing'
    if s174[key]['extension_accepted']:return 's174_crossing'
    return 's109_fallback'
def exported_source(key):
    if key not in diagnostics:
        row=next(r for r in map(json.loads,missing_export.read_text().splitlines()) if r['key']==key)
        reason=row['reason'];side=row['side']
    else:
        result=diagnostics[key].get('result') or {}
        reason=result.get('reason');side=result.get('side')
    if reason=='s176_inside_from_start':return 's176_first_frame'
    if reason=='s176_inside_first_minus_0p2s':return 's176_first_obs_minus_0p2'
    if reason=='crossing':
        return 's161_crossing' if side in ('LEFT','RIGHT') else 's174_crossing'
    return 's109_fallback'
overlap_checks=[]
for key,row in qa.items():
    if key in paired:
        overlap_checks.append(dict(key=key,cached_entry=paired[key]['s176_entry'],
                                   exported_entry=row['entry_frame'],cached_source=cached_source(key),
                                   exported_source=exported_source(key)))
rows=[]
for key,row in paired.items():
    src=exported_source(key) if key in qa else cached_source(key)
    entry=qa[key]['entry_frame'] if key in qa else row['s176_entry']
    collision=qa[key]['collision_frame'] if key in qa else row['collision']
    for label in row['labels']:
        panel={'ccd_consensus':'ccd_consensus_79',
               'ccd_labels_A3_low':'ccd_a3_low_45',
               'nexar_blind_confident':'nexar_confident_13',
               's117_interval':'nexar_confident_13'}.get(label['source'])
        if panel is None:continue
        rows.append(dict(key=key,panel=panel,source=src,entry=int(entry),collision=int(collision),
                         fps=float(row['fps']),interval=list(label['interval']),
                         prediction_origin='exported' if key in qa else 'cached_replay',
                         label_source=label['source']))
with (D/'external_meta/repos/DKB-2000_CrashIntent-AI/data/stage2/labels_manual.csv').open(newline='') as f:
    human=list(csv.DictReader(f))
assert len(human)==74
missing_keys=[]
for label in human:
    key='ccd/'+label['ID']
    if key not in qa and key not in paired:
        missing_keys.append(key)
        if args.allow_missing:continue
        raise AssertionError(key)
    if key in qa:
        entry=qa[key]['entry_frame'];collision=qa[key]['collision_frame'];src=exported_source(key)
        origin='exported'
    else:
        assert key in paired,key
        entry=paired[key]['s176_entry'];collision=paired[key]['collision'];src=cached_source(key)
        origin='cached_replay'
    true=int(label['entry_frame'])/10.
    rows.append(dict(key=key,panel='dkb_human_74',source=src,entry=int(entry),
                     collision=int(collision),fps=10.,interval=[true,true],
                     prediction_origin=origin,label_source='DKB_manual'))
expected=(79,45,13,74-len(missing_keys))
assert collections.Counter(r['panel'] for r in rows)==dict(zip(PANELS,expected))
assert all(0<=r['entry']<=r['collision'] and r['interval'][0]<=r['interval'][1] for r in rows)

def frame_at_shift(row,shift_s):
    delta=int(round(shift_s*row['fps']))
    return max(0,min(row['collision'],row['entry']+delta))
def hit(row,shift_s=0.):
    t=frame_at_shift(row,shift_s)/row['fps']
    lo,hi=row['interval']
    return lo-.30000001<=t<=hi+.30000001
def signed_error(row):
    return row['entry']/row['fps']-sum(row['interval'])/2
def summarize(items):
    errors=[signed_error(r) for r in items]
    if not errors:return dict(n=0,hits=0,misses=0)
    ordered=sorted(errors)
    return dict(n=len(items),hits=sum(hit(r) for r in items),misses=sum(not hit(r) for r in items),
                mean_s=round(statistics.mean(errors),3),median_s=round(statistics.median(errors),3),
                q25_s=round(ordered[(len(ordered)-1)//4],3),
                q75_s=round(ordered[(3*(len(ordered)-1))//4],3),
                early_misses=sum(not hit(r) and signed_error(r)<0 for r in items),
                late_misses=sum(not hit(r) and signed_error(r)>0 for r in items),
                exported=sum(r['prediction_origin']=='exported' for r in items))

baseline={panel:{source:summarize([r for r in rows if r['panel']==panel and r['source']==source])
                 for source in SOURCES} for panel in PANELS}
pooled={source:summarize([r for r in rows if r['source']==source]) for source in SOURCES}
SHIFTS=(-.3,-.2,-.1,0.,.1,.2,.3)
def paired(items,shift):
    before=sum(hit(r) for r in items);after=sum(hit(r,shift) for r in items)
    repairs=[r['key'] for r in items if not hit(r) and hit(r,shift)]
    breaks=[r['key'] for r in items if hit(r) and not hit(r,shift)]
    return dict(n=len(items),before=before,after=after,gain=after-before,
                repairs=repairs,breaks=breaks)
scan={source:{str(shift):dict(pooled=paired([r for r in rows if r['source']==source],shift),
                             panels={p:paired([r for r in rows if r['panel']==p and r['source']==source],shift)
                                     for p in PANELS}) for shift in SHIFTS}
      for source in SOURCES}
def choose(items):
    if not items:return 0.
    choices=[(paired(items,s)['gain'],-abs(s),s) for s in SHIFTS]
    best=max(choices)
    return best[2] if best[0]>0 else 0.
folds={}
for held in PANELS:
    test=[r for r in rows if r['panel']==held]
    held_keys={r['key'] for r in test}
    train=[r for r in rows if r['panel']!=held and r['key'] not in held_keys]
    chosen={source:choose([r for r in train if r['source']==source]) for source in SOURCES}
    by_source={source:paired([r for r in test if r['source']==source],chosen[source]) for source in SOURCES}
    folds[held]=dict(train_n=len(train),held_n=len(test),excluded_overlap_keys=len(held_keys&{r['key'] for r in rows if r['panel']!=held}),
                     shifts=chosen,by_source=by_source,
                     total_before=sum(v['before'] for v in by_source.values()),
                     total_after=sum(v['after'] for v in by_source.values()))

summary=dict(n=len(rows),panels=collections.Counter(r['panel'] for r in rows),
             missing_dkb_keys=missing_keys,
             source_counts=collections.Counter(r['source'] for r in rows),
             baseline=baseline,pooled=pooled,scan=scan,folds=folds,
             overlap_export_checks=overlap_checks,
             overlap_entry_mismatches=[r for r in overlap_checks if r['cached_entry']!=r['exported_entry']],
             overlap_source_mismatches=[r for r in overlap_checks if r['cached_source']!=r['exported_source']],
             interval_error='prediction minus annotated interval midpoint in seconds',
             hit_rule='prediction time within [interval.low-0.3, interval.high+0.3]',
             shift_rule='round(shift_seconds*fps) original frame positions, clamp [0,collision]',
             caveat='CCD consensus/A3 and Nexar labels are AI-made. DKB is human-labeled. '
                    'Most 243-clip predictions are cached trace replay; S178 QA exported 84 clips. '
                    'Held-out folds exclude every training label from a held-out clip ID.')
(H/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(H/'analysis.json').write_text(json.dumps(summary,indent=2,default=dict)+'\n')
for source in SOURCES:
    print(source,pooled[source],[(p,baseline[p][source]['hits'],baseline[p][source]['n']) for p in PANELS])
print('best pooled shifts',[(source,choose([r for r in rows if r['source']==source])) for source in SOURCES])
print('folds',[(p,v['shifts'],v['total_before'],v['total_after']) for p,v in folds.items()])
print('export mismatches',len(summary['overlap_entry_mismatches']),len(summary['overlap_source_mismatches']))
