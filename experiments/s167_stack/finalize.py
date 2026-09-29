"""Bind phase-1 archive, member identities, real CUDA parity and timing evidence."""
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

WORK = Path('$DATA_DIR/s167_stack/phase1')
DATA = WORK.parents[1]
PANELS = ['stage1_public', 'stage2_public', 'stage2_long', 'stage2_cascade', 'stage3_public']

def main():
    build = json.loads((WORK / 'build.json').read_text())
    receipt = json.loads((WORK / 'archive.json').read_text())
    archive_path = Path(receipt['path'])
    assert sha(archive_path) == receipt['sha256']
    assert sha(WORK / 'build.json') == receipt['build_sha256']
    errors = validate(archive_path)
    assert not errors, errors
    # Independently prove exact append composition and unchanged dispatch ASTs.
    with zipfile.ZipFile(build['base']) as base, zipfile.ZipFile(build['arms'][0]['path']) as cue, zipfile.ZipFile(build['arms'][1]['path']) as runtime:
        original, s1, fast = base.read('inference.py'), cue.read('inference.py'), runtime.read('inference.py')
        assert s1.startswith(original) and fast.startswith(original)
        composed = original + s1[len(original):] + fast[len(original):]
        assert (WORK / 'candidate/inference.py').read_bytes() == composed
        def functions(code):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
        combined_functions = functions(composed)
        assert combined_functions['predict_stage1'] == functions(s1)['predict_stage1']
        for key in ('predict_stage2', 'predict_stage3'):
            assert combined_functions[key] == functions(fast)[key]
    s164 = json.loads((REPO / 'candidates/s164_runtime/release.json').read_text())
    s163 = json.loads((REPO / 'candidates/s163_s1_cue/release.json').read_text())
    assert sha(DATA / 's163_s1_cue/qa/qa.json') == s163['qa_sha256']
    comparisons, reports, original_reports = {}, {}, {}
    for panel in PANELS + ['stage1_extras']:
        original_reports[panel] = json.loads((WORK / 'qa' / (panel + '.json')).read_text())
        assert original_reports[panel]['validation_ok']
        assert original_reports[panel]['inference_sha256'] == build['members']['inference.py']
        assert sha(WORK / 'qa' / (panel + '.csv')) == original_reports[panel]['csv_sha256']
        result_dir = WORK / ('qa_timing' if panel == 'stage1_public' and (WORK / 'qa_timing/stage1_public.json').exists() else 'qa')
        result_path = result_dir / (panel + '.json')
        result = json.loads(result_path.read_text())
        assert result['validation_ok'] and result['byte_identical_to_reference']
        assert result['inference_sha256'] == build['members']['inference.py']
        assert result['qa_source_sha256'] == sha(HERE / 'qa.py')
        csv = result_dir / (panel + '.csv')
        assert sha(csv) == result['csv_sha256']
        reference = Path(result['reference_csv'].replace('/mnt/d/', '$DATA_DIR/'))
        assert csv.read_bytes() == reference.read_bytes()
        assert result['resources']['peak_aggregate_rss_bytes'] <= 3 * 2**30
        assert result['resources']['peak_processes'] <= 2
        assert not result['network_attempts']
        if result['s164']:
            assert result['s164']['runtime_sha256'] == build['members']['model/runtime_s164/runtime.py']
        reports[panel] = result
        row = dict(rows=result['rows'], csv_sha256=sha(csv), byte_identical=True,
                   reference='S163' if panel.startswith('stage1') else 'S156',
                   stack_seconds=original_reports[panel]['seconds'],
                   timing_repeat_seconds=result['seconds'] if result_dir.name == 'qa_timing' else None,
                   qa_report_sha256=sha(result_path))
        if panel in PANELS:
            previous = s164['comparisons'][panel]
            base_csv = DATA / 's164_runtime/qa_base' / (panel + '.csv')
            assert sha(base_csv) == previous['csv_sha256']
            row.update(base_seconds=previous['base_seconds'], s164_seconds=previous['fast_seconds'])
        comparisons[panel] = row
    base_seconds = sum(comparisons[p]['base_seconds'] for p in PANELS)
    repeat_adjusted_seconds = sum(reports[p]['seconds'] for p in PANELS)
    # Keep the first complete pass primary; do not select the faster Stage1 run.
    stack_seconds = sum(original_reports[p]['seconds'] for p in PANELS)
    ratio = stack_seconds / base_seconds
    release = dict(receipt, release_ready=True, phase=1, completed_at=datetime.now().astimezone().isoformat(), base_sha256=build['base_sha256'],
        arms=[{k: arm[k] for k in ('path', 'sha256', 'changed', 'added')} for arm in build['arms']],
        changed_from_base=build['changed_from_base'], added_to_base=build['added_to_base'],
        unchanged_base_members=build['unchanged_base_members'], overlaps=build['overlaps'],
        static_validation_errors=errors, exact_append_composition=True, comparisons=comparisons,
        timing=dict(base_seconds=base_seconds, stack_seconds=stack_seconds, saved_fraction=1-ratio,
                    inherited_s164_seconds=sum(comparisons[p]['s164_seconds'] for p in PANELS),
                    server_base_seconds=2980, ratio_scaled_server_seconds=2980*ratio,
                    repeat_adjusted_seconds=repeat_adjusted_seconds,
                    repeat_adjusted_server_seconds=2980*repeat_adjusted_seconds/base_seconds,
                    method='Primary comparison retains all first-pass timings, including the contended Stage1. S156 reported 49m40s multiplied by whole matched-panel S167/S156 ratio; historical base reference, variable shared load; framework import excluded. Separate repeat-adjusted scenario is diagnostic.'),
        resources=dict(peak_aggregate_rss_bytes=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
                       peak_processes=max(r['resources']['peak_processes'] for r in reports.values()),
                       peak_cuda_reserved_bytes=max(r['cuda_peak_reserved'] for r in reports.values()),
                       torch_allocator_limit_gib=1.75, lease_cap_gb=2,
                       note='RAM sampled every .25s across parent and descendants, using sum RSS; CUDA reserved excludes driver/context.'),
        source_sha256={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.ps1')},
        timing_context_sha256=sha(HERE / 'timing_context.json'),
        qa_reports=reports, original_qa_reports=original_reports,
        stage1_timing_repeat_used=(WORK / 'qa_timing/stage1_public.json').exists(),
        uploaded=False, official_score=None)
    (HERE / 'release.json').write_text(json.dumps(release, indent=2) + '\n')
    (HERE / 'member_manifest.json').write_text(json.dumps(build, indent=2) + '\n')
    table = '\n'.join(f"| {p} | {comparisons[p]['rows']} | {comparisons[p]['base_seconds']:.2f} | {comparisons[p]['s164_seconds']:.2f} | {original_reports[p]['seconds']:.2f} |" for p in PANELS)
    estimate = 2980 * ratio
    readme = f'''# S167 stack, phase 1

Release: `{archive_path.as_posix()}` ({receipt['bytes']:,} bytes).
SHA256: `{receipt['sha256']}`.

Exact S156 carrier + S163 one-way Stage1 recapture cue + S164 runtime patch.
S160 collision and S162 acceleration are not included in phase 1.
Member-by-member decompressed SHA256 comparison finds 92 original members
unchanged, two changed (`inference.py`, `model/stage2/s144/predict.py`), and
three added (`model/stage1/s163/cue.py`, `model/stage1/s163/model.json`,
`model/runtime_s164/runtime.py`). All inherited weights and requirements match.
`member_manifest.json` records every base-arm and composed member identity.

The only cross-arm overlap is `inference.py`. Its exact bytes are S156 followed
by S163's original append and then S164's original append. S164 wraps the final
Stage2/3 functions; S163 wraps only Stage1. Final dispatch ASTs match their arms.
The S144 detector-loop change and all new files are copied byte-for-byte from
their arm ZIPs. Detector batching stays at its default 1.

Real offline WSL CUDA QA imported the staged export through an unregistered
dynamic loader. Stage1 matches S163 on 10 public clips plus all 3 prior cue
cases (the repaired recapture, known CCD false positive and safe original).
Stage2 matches S156 on 5 public and 7 existing Nexar clips; Stage3 matches S156
on all 5 full public clips / 5,992 rows. Every CSV matches reference bytes.
No network attempts, runtime prefetch errors, lazy-scene errors or Stage3 visual
fallbacks occurred. Public/reused training panels are integration evidence,
not a fresh accuracy holdout. S163's known false positive remains a score risk.

| Same panel | Rows | S156 seconds | S164 seconds | S167 first-pass seconds |
|---|---:|---:|---:|---:|
{table}

Matched-panel total: {base_seconds:.2f}s -> {stack_seconds:.2f}s
({100*(1-ratio):.1f}% reduction). S164's prior total was
{release['timing']['inherited_s164_seconds']:.2f}s. Scaling the requested
S156 reported ~49m40s by this measured ratio estimates **{int(estimate//60)}m{round(estimate%60):02d}s**.
This is a conditional projection, not measured server runtime. Baseline timings
are reused S164-worker measurements on the identical panels; current shared
load, hidden workload, storage and six-process server execution differ. QA
uses two processes and serial DataLoader loading in both reference and stack.
Stage2/3 runtime code is byte-identical to S164. Apparent extra gains relative
to its historical run are run-condition variation, not a new optimization.
Timing excludes framework import and the extra three Stage1 diagnostic cases.
The first Stage1 pass took {original_reports['stage1_public']['seconds']:.2f}s
during an overlapping DeepVoice 14 GB smoke lease (near-full physical GPU use).
It passed parity and is preserved in `original_qa_reports` and the table.
A separate Stage1 timing repeat was completed: {release['stage1_timing_repeat_used']}.
That repeat took {reports['stage1_public']['seconds']:.2f}s. Substituting it gives
a diagnostic panel total of {repeat_adjusted_seconds:.2f}s and ratio-scaled
server time of {release['timing']['repeat_adjusted_server_seconds']/60:.2f} minutes.
The first complete pass remains the primary comparison; the repeat does not
replace it or establish an isolated server-speed estimate.
Any repeat uses the same source and public inputs; no prediction was selected
or changed based on timing. `qa_timing` holds the separate repeat evidence.
The supplied ~49m40s is an inherited assumption, not independently verified by
this integration task. S164's more conservative motion-only projection is
~43m30s before S163's small workload-dependent CPU cue overhead.

QA acquired `$DATA_DIR/qa.lock` and the assigned `video-stack.json`
2 GB GPU lease for each panel, releasing between runs. Aggregate parent/child
RSS peaked at {release['resources']['peak_aggregate_rss_bytes']/2**30:.3f} GiB
(sampled every .25s), maximum {release['resources']['peak_processes']} processes;
CUDA reserved peaked at {release['resources']['peak_cuda_reserved_bytes']/2**30:.3f} GiB.
The Torch allocator cap was 1.75 GiB; CUDA reserved excludes driver/context.
Static submission validation, full ZIP CRC/member hash checks and exact
staged-tree identity passed. No private evaluation data, upload or commit.

Reproduction from repo root, using `.venv/Scripts/python.exe -B`:

```text
python -B candidates/s167_stack/build_stack.py --action stage --arms $DATA_DIR/releases/S163_s1.zip $DATA_DIR/releases/S164_fast.zip
powershell -NoProfile -File candidates/s167_stack/launch_qa.ps1
powershell -NoProfile -File candidates/s167_stack/launch_qa.ps1 -RepeatStage1
python -B candidates/s167_stack/build_stack.py --action package --arms $DATA_DIR/releases/S163_s1.zip $DATA_DIR/releases/S164_fast.zip
python -B tools/validate_submission.py $DATA_DIR/releases/S167_stk.zip
python -B candidates/s167_stack/finalize.py
```

Run long operations detached with logs as in `launch_qa.ps1`. Builder accepts
`--arms` as a list and always orders S164 last. Phase 2 uses a new `--out` and
`--zip` to preserve phase-1 evidence, adding the completed S160/S162 ZIPs.
All arm differences are computed against S156, with disjoint textual edits
merged automatically; incompatible overlapping edits stop for explicit review.
Phase 2 must update QA reference expectations for its changed Stage2/3 outputs
and review S164 motion-prefetch compatibility with those new arms.
Existing trees and release ZIPs are never overwritten. ZIP entries have fixed
timestamps, sorted names and fixed compression settings. Builder works from
the supplied release ZIPs, not mutable worker source files.
'''
    (HERE / 'README.md').write_text(readme, encoding='utf-8')
    (HERE / 'WORKER_REPORT.md').write_text(f'''RESULT done

CHANGED: Phase-1 S167 release built from exact S156 + S163 + S164; 97 members.
Reproducible multi-arm builder, QA launcher, source/member manifests and release receipt saved.
ZIP: {archive_path.as_posix()}
SHA256: {receipt['sha256']}

VERIFIED: Real offline WSL CUDA parity: Stage1=S163 (13 clips); Stage2=S156
(12 clips); Stage3=S156 (5 clips, 5,992 rows). Full CRC/member hashes and
tools/validate_submission.py pass. Matched timing {base_seconds:.2f}s ->
{stack_seconds:.2f}s ({100*(1-ratio):.1f}% reduction); ratio-scaled server estimate
{int(estimate//60)}m{round(estimate%60):02d}s from inherited 49m40s. Peak aggregate RAM
{release['resources']['peak_aggregate_rss_bytes']/2**30:.3f} GiB; maximum 2 processes;
CUDA reserved {release['resources']['peak_cuda_reserved_bytes']/2**30:.3f} GiB.

RISKS: Timing uses historical base runs under variable shared load; no official
score or server measurement. S163 retains its known exploratory false-positive
risk. CUDA allocator statistics exclude driver/context. No upload/commit.

NEXT: Stopped after phase 1. Await explicit phase-2 prompt for S160/S162 arms.
''', encoding='utf-8')
    print(json.dumps({k:release[k] for k in ('path','sha256','release_ready','timing','resources')}, indent=2))

if __name__ == '__main__':
    main()
