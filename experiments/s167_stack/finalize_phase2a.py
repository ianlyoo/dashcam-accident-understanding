"""Finalize S169 only after exact-artifact parity and cached-path fault QA."""
import ast
from datetime import datetime
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from candidates.s167_stack.build_stack import sha
from tools.validate_submission import validate

DATA = Path('$DATA_DIR')
WORK = DATA / 's167_stack/phase2a'
RECORDS = HERE / 'phase2a'
PANELS = ['stage1_public', 'stage1_extras', 'stage2_public', 'stage2_long',
          'stage2_cascade', 'stage3_public', 'stage3_fault']
HOOK = 'model/stage3/s141/predict.py'
S162_SHA = '31bff1cfae4ff312bfe4b371906a2055695bcc29b8de48c4e9ca3ead1408f33a'

def main():
    import pandas as pd
    build = json.loads((WORK / 'build.json').read_text())
    previous = json.loads((DATA / 's167_stack/phase1/build.json').read_text())
    archive = json.loads((WORK / 'archive.json').read_text())
    assert archive['build_sha256'] == sha(WORK / 'build.json')
    assert sha(archive['path']) == archive['sha256']
    assert build['members'].keys() == previous['members'].keys()
    assert [p for p in previous['members'] if previous['members'][p] != build['members'][p]] == [HOOK]
    arm = next(a for a in build['arms'] if Path(a['path']).name == 'S162_acc.zip')
    assert arm['sha256'] == S162_SHA and arm['changed'] == [HOOK] and not arm['added']
    assert build['members'][HOOK] == arm['member_sha256'][HOOK]
    with zipfile.ZipFile(build['base']) as z:
        original = ast.parse(z.read(HOOK))
    current = ast.parse((WORK / 'candidate' / HOOK).read_bytes())
    functions = lambda tree: {n.name:ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('design','continuous','steering'):
        assert functions(original)[name] == functions(current)[name]
    s162_release = json.loads((REPO / 'candidates/s162_accel_dom/release.json').read_text())
    assert sha(s162_release['qa']['path']) == s162_release['qa']['sha256']
    prior_qa = json.loads(Path(s162_release['qa']['path']).read_text())
    assert prior_qa['passed'] and prior_qa['changed_accel_rows'] == 68
    reports, comparisons = {}, {}
    for panel in PANELS:
        path = WORK / 'qa' / (panel + '.json')
        report = json.loads(path.read_text())
        assert report['validation_ok'] and report['byte_identical_to_reference']
        assert report['inference_sha256'] == build['members']['inference.py']
        assert report['s162_source_sha256'] == build['members'][HOOK]
        assert report['qa_source_sha256'] == sha(HERE / 'qa_phase2a.py')
        assert not report['network_attempts']
        assert report['resources']['peak_aggregate_rss_bytes'] <= 3 * 2**30
        assert report['resources']['peak_processes'] <= 2
        assert report['cuda_peak_reserved'] <= 2 * 2**30
        if report['s164']:
            assert report['s164']['runtime_sha256'] == build['members']['model/runtime_s164/runtime.py']
            assert not report['s164'].get('errors')
        csv = WORK / 'qa' / (panel + '.csv')
        assert sha(csv) == report['csv_sha256']
        reference = Path(report['reference_csv'].replace('/mnt/d/', '$DATA_DIR/'))
        expected = reference.read_bytes()
        if panel == 'stage3_fault':
            frame = pd.read_csv(reference)
            expected = frame[frame.ID.isin(report['reference_subset_ids'])].to_csv(index=False, lineterminator='\n').encode()
        assert csv.read_bytes() == expected
        reports[panel] = report
        comparisons[panel] = dict(rows=report['rows'], seconds=report['seconds'], byte_identical=True,
            csv_sha256=sha(csv), reference_csv=str(reference), reference_csv_sha256=sha(reference),
            qa_report_sha256=sha(path))
    normal, fault = reports['stage3_public'], reports['stage3_fault']['fault_injection']
    assert normal['rows'] == 2998 and normal['changed_accel_rows'] == 68
    assert normal['steering_equal_s156'] and normal['stopped_equal_s156']
    assert normal['s164']['prefetched'] == 5 and normal['s164']['observer_replay_frames'] == 2998
    assert fault['partial_mutation'] and fault['next_call_recovers_s162'] and fault['all_hooks_restored']
    assert fault['changed_back_to_s156_rows'] == 58 and fault['other_clip_retains_s162']
    assert fault['fault_s164']['prefetched'] == 2 and fault['fault_s164']['observer_replay_frames'] == 1200
    assert sha(WORK / 'qa/stage3_fault_injected.csv') == fault['fault_csv_sha256']
    failed = pd.read_csv(WORK / 'qa/stage3_fault_injected.csv')
    base = pd.read_csv(DATA / 's162_accel_dom/S162_acc_r3/qa/source_stage3.csv')
    assert failed[failed.ID == 'OPEN_004'].reset_index(drop=True).equals(base[base.ID == 'OPEN_004'].reset_index(drop=True))
    s164 = json.loads((REPO / 'candidates/s164_runtime/release.json').read_text())
    baseline_times = {p:s164['comparisons'][p]['base_seconds'] for p in ('stage1_public','stage2_public','stage2_long','stage2_cascade')}
    baseline_times['stage3_public'] = json.loads((DATA / 's162_accel_dom/capture/report.json').read_text())['public']['seconds']
    base_total = sum(baseline_times.values())
    stack_total = sum(reports[p]['seconds'] for p in baseline_times)
    estimate = 2980 * stack_total / base_total
    errors = validate(Path(archive['path']))
    assert not errors, errors
    receipt = dict(archive, candidate='S169', phase='2a', release_ready=True,
        completed_at=datetime.now().astimezone().isoformat(),
        base_sha256=build['base_sha256'], s162_sha256=S162_SHA,
        phase1_changed_members=[HOOK], unchanged_from_s167=96,
        protected_steering_functions_identical=True, comparisons=comparisons,
        fault_injection=fault, static_validation_errors=errors,
        timing=dict(base_same_panel_seconds=baseline_times, base_total_seconds=base_total,
            stack_total_seconds=stack_total, saved_fraction=1-stack_total/base_total,
            stage3_s162_seconds=prior_qa['seconds']['stage3'],
            inherited_server_base_seconds=2980, ratio_scaled_server_seconds=estimate,
            caveat='Historical S156 references under variable shared load. Stage3 uses the same 2998-row 10Hz panel as S162; never compare this timing directly with the prior 5992-row native panel. Excludes framework imports and diagnostic/fault panels.'),
        resources=dict(peak_aggregate_rss_bytes=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
            peak_processes=max(r['resources']['peak_processes'] for r in reports.values()),
            peak_cuda_reserved_bytes=max(r['cuda_peak_reserved'] for r in reports.values()),
            torch_allocator_limit_gib=1.75, gpu_lease_cap_gb=2,
            caveat='Aggregate RSS sampled every .25s. CUDA reserved excludes driver/context.'),
        source_sha256={name:sha(HERE / name) for name in ('build_stack.py','prepare_phase2a.py','qa_phase2a.py','launch_phase2a.ps1','finalize_phase2a.py','finish_phase2a.ps1')},
        s162_qa_receipt_sha256=s162_release['qa']['sha256'], qa_reports=reports,
        infrastructure_history=['Initial idle waiter exited on another worker deleting its lease between enumeration and read; no CUDA panel had started. Fixed deletion-race handling, retained launcher.err, and restarted only the waiter.'],
        uploaded=False, official_score=None)
    RECORDS.mkdir(exist_ok=True)
    (RECORDS / 'release.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (RECORDS / 'member_manifest.json').write_text(json.dumps(build, indent=2) + '\n')
    table = '\n'.join(f"| {p} | {reports[p]['rows']} | {baseline_times[p]:.2f} | {reports[p]['seconds']:.2f} |" for p in baseline_times)
    resources = receipt['resources']
    readme = f'''# S169 stack: phase 2a

Ready {receipt['completed_at']}. ZIP `{Path(archive['path']).as_posix()}`,
{archive['bytes']:,} bytes, SHA256 `{archive['sha256']}`.

Exact S156 + S163 Stage1 cue + S164 runtime + S162 conservative acceleration
decoder. S160 collision is not included. Compared with S167, exactly one of
97 members changes: `{HOOK}`. It is byte-identical to the pinned S162 release;
the other 96 members, including inference.py, S164 runtime, weights and
requirements, are unchanged. `member_manifest.json` records every arm hash
and base-relative change. The original multi-arm builder is reused unchanged.

S164 wraps final Stage3 dispatch and prefetches only original CPU motion arrays.
S141/S162 still runs in the parent, inside its original lock, and temporarily
replaces S109's acceleration combiner. S162 does not modify motion extraction
or the frame observer. Normal CUDA QA prefetched all 5 clips and replayed all
2,998 frames through the original visual observer; no legacy fallback occurred.
S162's steering function ASTs match S156. All scoped hooks are restored.

Actual offline WSL/CUDA checks passed:

- Stage1: all 10 public and 3 diagnostic cases equal S163 CSV bytes.
- Stage2: all 5 public and 7 existing Nexar clips equal S156 CSV bytes.
- Stage3: all 2,998 rows equal S162, with exactly 68 acceleration changes
  versus S156; steering, IDs, sample indices and STOPPED remain identical.
- Fault injection: two clips enter S164 prefetch and replay (1,200 frames).
  OPEN_004's decoder mutates its logits and raises once; all its columns match
  S156, observably restoring 58 acceleration rows. OPEN_002 retains its normal
  S162 output. No visual-model fallback occurs. A subsequent two-clip call
  returns normal S162 outputs, and decoder/combiner/extractor hooks restore.

The fault targets a worker-prefetched clip with real S162 label differences,
so it exercises the combined cached-feature path, not a single-file serial
shortcut. Fault and recovered CSVs, diagnostics and hashes are retained under
`$DATA_DIR/s167_stack/phase2a/qa`.

| Same panel | Rows | S156 seconds | S169 seconds |
|---|---:|---:|---:|
{table}

Same-panel total {base_total:.2f}s -> {stack_total:.2f}s
({100*(1-stack_total/base_total):.1f}% reduction). Conditional scaling of the
inherited 49m40s server reference gives **{int(estimate//60)}m{round(estimate%60):02d}s**.
These are historical baseline comparisons under changing shared load, not a
controlled speedup or measured server result. Stage3 uses S162's existing
10Hz public panel (2,998 rows); S167/S164's earlier 184.03 -> 126.48 seconds used
5,992 native rows and is not directly comparable. S162 alone previously took
{prior_qa['seconds']['stage3']:.2f}s on the same 10Hz panel. Framework imports,
three Stage1 diagnostic clips, fault injection and recovery are excluded from
the matched total. S164's more conservative inherited estimate remains about
43m30s plus small, workload-dependent cue/decoder overhead, also unverified.

Peak aggregate RSS {resources['peak_aggregate_rss_bytes']/2**30:.3f} GiB,
maximum {resources['peak_processes']} processes; CUDA reserved
{resources['peak_cuda_reserved_bytes']/2**30:.3f} GiB. QA used a 1.75 GiB Torch
allocator cap, assigned 2 GB video-stack lease, machine-wide qa.lock, serial
DataLoader loading and S164_WORKERS=2. RSS sampled every .25s; CUDA reserved
excludes driver/context. No network attempts or cap violations were observed.
All ZIP member hashes/CRCs and tools/validate_submission.py passed.
The initial idle waiter exited when another worker removed its lease during
enumeration, before CUDA began. Lease scanning was corrected to tolerate that
deletion and to defer admission on extant unreadable leases. The original
launcher.err is preserved; the replacement waiter reuses completed evidence.

S162 remains exploratory: its reused proxy evidence does not establish hidden
accuracy, and its sparse public acceleration labels regressed 40/50 -> 39/50.
S163's known false-positive risk remains. No new accuracy improvement, private
evaluation access, upload, commit or official score is claimed.

Reproduction from repo root, Windows Python `.venv/Scripts/python.exe -B`:

```text
python -B candidates/s167_stack/prepare_phase2a.py
powershell -NoProfile -File candidates/s167_stack/launch_phase2a.ps1
powershell -NoProfile -File candidates/s167_stack/finish_phase2a.ps1
```

Preparation calls the unchanged build_stack.py with arms S163_s1.zip,
S162_acc.zip and S164_fast.zip; output is s167_stack/phase2a/candidate.
Existing trees and ZIPs are preserved. Run long jobs detached with logs.
Phase 2a stops here; phase 2b requires the later explicit S160 prompt.
'''
    (RECORDS / 'README.md').write_text(readme, encoding='utf-8')
    (RECORDS / 'WORKER_REPORT.md').write_text(f'''RESULT done

CHANGED: S169_stk.zip = exact S167 + S162, only {HOOK} differs;
96 other members unchanged. Reused multi-arm builder. Receipt and README saved.
ZIP: {Path(archive['path']).as_posix()}
SHA256: {archive['sha256']}

VERIFIED: Real offline CUDA Stage1=S163 (13 clips), Stage2=S156 (12 clips),
Stage3=S162 (2,998 rows; exactly 68 accel changes; steering/STOPPED=S156).
Two-clip cached-path fault injection restores 58 OPEN_004 rows to S156 after
partial mutation, preserves the other clip, and recovers S162 on the next call.
Hooks restore; no legacy fallback. CRC/member hashes and submission validation pass.
Peak aggregate RAM {resources['peak_aggregate_rss_bytes']/2**30:.3f} GiB;
max 2 processes; CUDA reserved {resources['peak_cuda_reserved_bytes']/2**30:.3f} GiB.

RISKS: Historical shared-load runtime {base_total:.2f}s -> {stack_total:.2f}s;
conditional server estimate {int(estimate//60)}m{round(estimate%60):02d}s, not official.
Exploratory S162/S163 accuracy risks remain. No upload or commit.

NEXT: Stopped after phase 2a; await S160 phase-2b prompt.
''', encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('path','sha256','bytes','release_ready','completed_at','timing','resources')}, indent=2))

if __name__ == '__main__':
    main()
