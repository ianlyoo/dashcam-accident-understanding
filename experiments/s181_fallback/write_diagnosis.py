"""Rewrite phase6 diagnosis from the reproduced failure and passing exports."""
import hashlib,json,pathlib,zipfile
import pandas as pd

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');P=R/'candidates/s167_stack/phase6'
W=D/'s181_fallback';batch=json.loads((W/'diagnosis/batch.json').read_text())
side=json.loads((W/'diagnosis_side/batch.json').read_text())
probe=json.loads((D/'s167_stack/phase6/qa_diagnose/comparison.json').read_text())
qa=json.loads((D/'s167_stack/phase6/qa/public.json').read_text())
def sha(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()
def keyed(path):
    frame=pd.read_csv(path,dtype={'ID':str})
    return {str(r['ID']):r for r in json.loads(frame.to_json(orient='records'))}
saved=keyed(D/'s167_stack/phase5/qa/stage2_public.csv')
live=keyed(D/'s167_stack/phase6/qa_diagnose/live_s177.csv')
assert saved==live
fault=probe['fault_diagnostics']['clips'];first=batch['first_id']
assert 'S182 injected' in fault[first]['error']
assert all(v['error'].startswith('RecursionError:') for k,v in fault.items() if k!=first)
assert not probe['same_values_numpy'] and not probe['same_values_json']
assert probe['normal_dtypes']==probe['fault_dtypes']
for name in ('saved_vs_live_base','normal_vs_frozen','normal_vs_fault_saved_unaffected',
             'normal_vs_fault_live_unaffected','normal_vs_reverse','normal_vs_recovery'):
    assert not batch[name],name
for name in ('normal_vs_frozen','normal_vs_fault_saved_unaffected',
             'normal_vs_fault_live_unaffected','normal_vs_reverse','normal_vs_recovery'):
    assert not side[name],name
assert qa['passed'] and qa['fault']==dict(calls=1,fallback_exact_s177=True,recovery_exact_s182=True)
assert qa['qa_source_sha256']==sha(P/'qa.py')
assert qa['build_sha256']==sha(D/'s167_stack/phase6/build.json')
normal=keyed(W/'diagnosis/normal.csv');side_normal=keyed(W/'diagnosis_side/normal.csv')
fresh=[];full=[];side_fresh=[]
for ident in sorted(normal):
    single=json.loads((W/'diagnosis'/('single_'+ident+'.json')).read_text())
    assert single['row']==normal[ident]
    fresh.append(ident)
    file=W/'diagnosis'/('full_single_'+ident+'.json')
    if file.exists():
        full_single=json.loads(file.read_text())
        assert full_single['row']==normal[ident],(ident,full_single['row'],normal[ident])
        full.append(ident)
for ident in sorted(side_normal):
    single=json.loads((W/'diagnosis_side'/('single_'+ident+'.json')).read_text())
    assert single['row']==side_normal[ident]
    side_fresh.append(ident)
assert len(fresh)==5 and len(side_fresh)==3
assert side['diagnostics']['normal']['clips']['ccd_000080']['rule']=='S178 side'
assert side_normal['ccd_000080']['entry_side']=='RIGHT'
build=json.loads((D/'s167_stack/phase6/build.json').read_text())
with zipfile.ZipFile(D/'releases/S181_fallback.zip') as archive:
    source=archive.read('model/stage2/s161/predict.py')
assert hashlib.sha256(source).hexdigest()==sha(D/'s167_stack/phase6/candidate/model/stage2/s181_predict.py')
assert b'int(round(0.85*fps))' in source and b'int(round(0.2 * fps))' in source
assert b'self._s176_original_result = None' in source
assert b'self._s176_fallback_completed = False' in source
assert build['arms'][-1]['sha256']==sha(D/'releases/S181_fallback.zip')
phase_code=(P/'qa.py').read_text()
assert "fault_source_paths = ns['_s008_frame_paths']" in phase_code
assert 'return fault_source_paths(path)' in phase_code

lines=[
    '# S182 entry-chain isolation diagnosis',
    '',
    'RESULT: harness defect fixed; no S174/S176/S178/S180/S181 candidate-state',
    'leakage was found. The corrected exact integrator public CUDA QA passes.',
    '',
    '## Reproduced cause',
    '',
    'The original `phase6/qa.py` bound `frame_paths()` over the local name',
    '`old_paths`. Its fault block later rebound `old_paths` to the wrapper',
    'itself. Python closures capture the variable binding, so every unaffected',
    'clip called `frame_paths()` recursively. The instrumented original harness',
    'reproduced the failure: the injected first clip raised the intended',
    '`RuntimeError`, while all four unaffected clips recorded',
    '`RecursionError: maximum recursion depth exceeded` and retained S177',
    'entries. DataFrame dtypes were identical; the values differed because',
    'the unaffected clips had hit the harness error path.',
    '',
    'The saved phase5 S177 baseline and the S177 rows captured from the same',
    'normal exported S182 run match exactly on all five public sample clips.',
    'Thus baseline drift is excluded for the reproduced public failure.',
    '',
    'I renamed only the fault block alias to `fault_source_paths`, leaving the',
    'outer `old_paths` binding intact. No candidate model source, patch, ZIP,',
    'or staged S182 runtime changed. The corrected `phase6/qa.py` has a new',
    'source hash; the staged build hash is unchanged.',
    '',
    '## Inference-state audit',
    '',
    'S174 constructs its candidate tracks, lane variants, and detection cache',
    'inside each `Tracker.analyze` call. S176 resets its two instance fields',
    '`_s176_original_result` and `_s176_fallback_completed` at the start of',
    'each clip. S178 reads that clip result to choose side; S180 reads the',
    'current result to apply a fixed 0.2 s shift. S181 uses a literal 0.85 s',
    'crossing-to-collision gap fitted offline on public S161 crossings; the',
    'packaged `predict.py` computes no median or statistic from test clips.',
    'The detector is in eval mode. No module-level clip accumulator, running',
    'median, or previous-clip feature is used by these arms. The S182 wrapper',
    'creates a new tracker and per-call diagnostics for each Stage2 invocation.',
    '',
    '## Exported invariance evidence',
    '',
    'With the same captured S177 rows, the five public sample clips have',
    'identical values in original order, reversed order, a same-base fault run',
    'for all unaffected clips, and same-process recovery. Each of the five',
    'also matches when the S182 chain is run alone in a fresh WSL process.',
    f'Whole S182 Stage2 fresh-process singles match for {len(full)}/5 public',
    'clips as of this report. The original sample panel exercises one S180',
    'crossing shift and two S181 gap decisions.',
    '',
    'A second three-clip panel on frozen S177 rows exercises S178 side:',
    '`ccd_000080` changes LEFT to RIGHT. Original order, reversed order,',
    'faulted unaffected outputs, recovery, and all three fresh-process',
    'single-clip exports match exactly.',
    '',
    'The corrected exact integrator public QA receipt reports five rows,',
    'three changed entries, zero changed sides, one intentional fault call,',
    'exact S177 fallback for the affected clip, and exact S182 recovery.',
    f"Peak aggregate RSS {qa['resources']['peak_aggregate_rss_bytes']/2**30:.2f} GiB,",
    f"CUDA reservation {qa['cuda_peak_reserved']/2**30:.2f} GiB, and minimum",
    f"host free commit {qa['resources']['minimum_host_free_commit_gib']:.2f} GiB;",
    'no network attempts. The QA source and build hashes in the receipt match',
    'the current files.',
    '',
    'Evidence: `$DATA_DIR/s167_stack/phase6/qa_diagnose/`',
    '(original reproduced fault, live base and row comparison);',
    '`$DATA_DIR/s181_fallback/diagnosis/` (public order,',
    'fault, recovery, singles); `.../diagnosis_side/` (active S178 side);',
    'and `$DATA_DIR/s167_stack/phase6/qa/public.json`',
    '(corrected integrator acceptance receipt).',
    '',
    'NEXT: the integrator can reuse the corrected public QA receipt, run its',
    'remaining panels and whole-artifact checks, then package. No entry-arm',
    'patch revision is needed for this issue.',
]
(P/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('WROTE',P/'DIAGNOSIS.md','full singles',len(full))
