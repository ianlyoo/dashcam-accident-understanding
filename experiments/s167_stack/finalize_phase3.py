"""Bind exact S171 entry-only integration evidence before packaging/promotion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from candidates.s167_stack.build_stack import sha
from tools.validate_submission import validate
DATA=Path('$DATA_DIR');WORK=DATA/'s167_stack/phase3';RECORDS=HERE/'phase3'
PANELS=['stage2_public','stage2_long','stage2_cascade','stage2_entry','stage2_fault','stage1_public','stage1_extras','stage3_public']

def integrity():
    build=json.loads((WORK/'build.json').read_text());merge=json.loads((WORK/'merge.json').read_text())
    cpu=json.loads((WORK/'cpu.json').read_text())
    assert merge['build_sha256']==cpu['build_sha256']==sha(WORK/'build.json')
    assert cpu['passed'] and cpu['source_sha256']==sha(HERE/'cpu_phase3.py')
    root=WORK/'candidate'
    assert {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}==build['members']
    previous=json.loads((DATA/'s167_stack/phase2b/build.json').read_text())
    prevroot=DATA/'s167_stack/phase2b/candidate'
    assert {p.relative_to(prevroot).as_posix():sha(p) for p in prevroot.rglob('*') if p.is_file()}==previous['members']
    assert sha(DATA/'releases/S170_all.zip')==merge['s170_sha256']
    assert merge['unchanged_s170_members']==99 and len(build['members'])==104
    for name,digest in merge['reference_s161_members'].items():
        assert sha(WORK/'reference_s161'/name)==digest==build['members'][name]
    return build,merge,cpu,previous

def check_qa(build,merge,previous):
    import pandas as pd
    reports={}
    for name in ['baseline_stage2_entry']+PANELS:
        isbase=name.startswith('baseline_');label=name.removeprefix('baseline_')
        folder='qa_baseline' if isbase else 'qa';path=WORK/folder/(label+'.json')
        r=json.loads(path.read_text());csv=WORK/folder/(label+'.csv')
        identity=previous['members'] if isbase else build['members']
        assert r['validation_ok'] and r['baseline_generation']==isbase
        retained = name in ('baseline_stage2_entry','stage2_public','stage2_long')
        qa_source = RECORDS/'harness_before_atomic/qa_phase3.py' if retained else HERE/'qa_phase3.py'
        monitor_source = RECORDS/'harness_before_atomic/qa_phase2b_4g.py' if retained else HERE/'monitor_phase3.py'
        assert r['build_sha256']==sha(WORK/'build.json') and r['qa_source_sha256']==sha(qa_source)
        assert r['resource_monitor_sha256']==sha(monitor_source)
        assert all(digest==identity[name] for name,digest in r['source_sha256'].items())
        assert sha(csv)==r['csv_sha256'] and not r['network_attempts']
        assert r['resources']['peak_aggregate_rss_bytes']<=4*2**30 and r['resources']['peak_processes']<=2
        assert r['resources']['minimum_host_free_commit_gib']>=12 and r['cuda_peak_reserved']<=2*2**30
        actual=pd.read_csv(csv,dtype={'ID':str})
        if not isbase:
            ref=Path(r['reference_csv'].replace('/mnt/d/','$DATA_DIR/'))
            assert sha(ref)==r['reference_sha256']
            baseline=pd.read_csv(ref,dtype={'ID':str})
            if r['stage']==2:
                assert actual.drop(columns='entry_frame').equals(baseline.drop(columns='entry_frame'))
                assert all(actual.entry_frame<=actual.collision_frame)
                if label!='stage2_fault':
                    independent=r['independent_entry']
                    assert independent['pinned_source_sha256']==build['members']['model/stage2/s161/predict.py']
                    assert actual.entry_frame.tolist()==[x['expected_entry'] for x in independent['rows']]
                else:
                    fault=r['fault'];assert fault['all_rows_equal_s170'] and fault['next_call_recovers_s171'] and fault['hooks_restored']
                    assert fault['observable_entry_changes']>0
                    injected=WORK/'qa/stage2_fault_injected.csv'
                    assert sha(injected)==fault['csv_sha256']
                    assert pd.read_csv(injected,dtype={'ID':str}).equals(baseline)
                    assert csv.read_bytes()==(WORK/'qa/stage2_entry.csv').read_bytes()
            else:
                assert csv.read_bytes()==ref.read_bytes()
        r['qa_report_sha256']=sha(path);reports[name]=r
    return reports

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--integrity-only',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    build,merge,cpu,previous=integrity()
    if a.integrity_only:
        print('S171 integrity PASS: 104 members; exact S170 + pinned S161; CPU mode/fallback PASS',flush=True);return
    reports=check_qa(build,merge,previous)
    if a.check_only:
        print('S171 CUDA acceptance PASS; packaging permitted',flush=True);return
    archive=json.loads((WORK/'archive.json').read_text())
    assert archive['build_sha256']==sha(WORK/'build.json') and archive['sha256']==sha(archive['path'])
    assert archive['crc_and_member_hashes_passed']
    errors=validate(Path(archive['path']));assert not errors,errors
    s170=json.loads((HERE/'phase2b/release.json').read_text())
    keys=['stage1_public','stage2_public','stage2_long','stage2_cascade','stage3_public']
    total=sum(reports[k]['seconds'] for k in keys);base=s170['timing']['base_total_seconds'];estimate=2980*total/base
    timing=dict(s171_seconds={k:reports[k]['seconds'] for k in keys},s170_seconds={k:s170['qa_reports'][k]['seconds'] for k in keys},
        s156_total_seconds=base,s171_total_seconds=total,conditional_server_seconds=estimate,
        caveat='Historical/shared-load matched panels. Excludes imports, diagnostics, independent tracker replay and faults. No measured server result.')
    resources=dict(rss_cap_gib=4,peak_aggregate_rss_bytes=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
        peak_cuda_reserved_bytes=max(r['cuda_peak_reserved'] for r in reports.values()),max_processes=max(r['resources']['peak_processes'] for r in reports.values()),
        minimum_host_free_commit_gib=min(r['resources']['minimum_host_free_commit_gib'] for r in reports.values()))
    receipt=dict(archive,candidate='S171',phase='3',release_ready=True,completed_at=datetime.now().astimezone().isoformat(),
        s170_uploaded_row=105253,merge=merge,cpu=cpu,qa_reports=reports,timing=timing,resources=resources,static_validation_errors=errors,
        source_sha256={p.name:sha(p) for p in [HERE/'build_stack.py',HERE/'qa_phase2b_4g.py',*HERE.glob('*phase3*.py'),*HERE.glob('*phase3*.ps1')]},
        infrastructure_history=['CPU fixture initially omitted diagnostics clips and returned a scalar instead of the tracker result dictionary. Both fixture-only issues were corrected and all CPU checks passed. A relative child interpreter invocation was also corrected to the absolute path. These were harness failures, not candidate output failures; team authorized continuation. Candidate unchanged.'],uploaded=False,official_score=None)
    receipt['infrastructure_history'].append('The detached launcher stopped while queued because a concurrent bounded supervisor-log read briefly denied the PowerShell Add-Content writer. A shared-access log writer with bounded retries fixed this harness-only issue. Completed CUDA panels were retained; candidate and QA comparison code were unchanged.')
    receipt['infrastructure_history'].append('A later host_commit.json sharing violation stopped its writer and caused a stale-sample abort. team authorized unique-temp atomic os.replace writes with five 0.5-second PermissionError retries, skipped samples on failure, and warning-only stale reads. Fresh QA uses this monitor; baseline/public/long passing evidence retains its exact historical QA/monitor source snapshots. Candidate inference and comparison assertions are unchanged.')
    receipt['infrastructure_history'].append('The first run with the separated monitor stopped before inference because THREAD_ENV initialization had previously been an import side effect of the old monitor module. Explicit initialization restored the same environment; setup logs are preserved. No candidate output mismatch occurred.')
    RECORDS.mkdir(exist_ok=True)
    (RECORDS/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (RECORDS/'member_manifest.json').write_text(json.dumps(build,indent=2)+'\n')
    details=f'''# S171 entry stack

ZIP: {Path(archive['path']).as_posix()}
SHA256: {archive['sha256']}
Bytes: {archive['bytes']}; members: 104. Ready: {receipt['completed_at']}.

Exact S170 (uploaded row 105253) plus S161. Only S118 adapter.py and rule.json
change; three S161 files are added. The other 99 S170 members are identical.
All added/modified files are byte-identical to the pinned mode-repaired S161 arm.
The existing build_stack.py reproduces the five-arm stack from S156; S164 wraps
the final dispatch. The prior reviewed S144 CRLF merge remains unchanged.

Final entry rule: compute S160 collision first, retaining S170's current entry.
Run S161's observed-crossing tracker anchored to that final S160 collision.
If S161 accepts a crossing, select its frame number; otherwise retain S170 entry.
Then clamp entry = min(selected entry, S160 collision). A tracker load failure
returns the supplied S170 rows; a per-clip error retains that S170 entry. Side
and evasion are untouched. S161's fixed validator permits entry-only tracking
with preserve_s109_fields, while rejecting track side=true. No cumulative
cross-file timing budget is used.

Independent pinned-S161 tracker replay verifies the intended entry rule on every
public/long/cascade and three-clip entry diagnostic row, using actual S170
collisions as anchors. Standalone S161 CSVs use S156 collision anchors and are
not assumed to be the correct S171 entries. Protected Stage2 fields equal S170.
Real injected tracker errors retain complete S170 rows; the next same-process
call recovers S171 with observable entry changes and restored runtime hooks.
CPU fixtures also verify load fallback, clipping, recovery and the mode guard.
Stage1 (13 clips) and Stage3 (2998 rows) match S170 CSV bytes. CRC/member SHA256
and tools/validate_submission.py pass. No private evaluation data was accessed.

Matched inference total {total:.2f}s; conditional scaling from S156's 49m40s gives
{int(estimate//60)}m{round(estimate%60):02d}s. Shared load and server stage mix remain
uncertain; this is not an official runtime. Independent replay/fault/diagnostic
work and framework imports are excluded. Per-panel S170/S171 times are in release.json.
RSS peak {resources['peak_aggregate_rss_bytes']/2**30:.3f} GiB under 4 GiB;
CUDA reserved {resources['peak_cuda_reserved_bytes']/2**30:.3f} GiB under the 2 GB lease;
minimum sampled free commit {resources['minimum_host_free_commit_gib']:.2f} GiB;
at most two QA processes. Machine-wide qa.lock is shared with S168 and released
between panels. PowerShell is NonInteractive; command/QA waits are bounded.
The earlier 3 GiB limit was a local machine-sharing guard, not a competition
limit. Fault recovery can exceed it; team explicitly approved 4 GiB with at least
12 GiB free commit. The measured peak above records that approved exception.

S161 remains exploratory: observed entry proxies do not establish hidden-score
gain, and anchor/geometry limits remain. Other arms retain their recorded risks.
No upload, commit or push. Prior failure evidence is preserved.

team monitor-policy revision: telemetry uses unique temporary files and os.replace,
with five retries at 0.5 seconds on PermissionError. Failed writes and stale
samples warn and skip; they never stop inference. Actual observed RSS/process
or fresh free-commit breaches still enforce resource caps. Passed baseline,
public and long evidence was retained against preserved historical source
snapshots; only remaining checks were run. The earlier cascade stale-monitor
abort was a harness failure, with RSS below 1 GiB and free commit over 70 GiB.
'''
    (RECORDS/'README.md').write_text(details,encoding='utf-8')
    report=f'''RESULT done

CHANGED: S171_ent.zip = exact S170 + S161; 104 members, 99 S170 members unchanged.
SHA256: {archive['sha256']}

VERIFIED: Stage1/3 and Stage2 collision/side/evasion equal S170; independent
S161 entry replay anchored to S160 collision, then clamped. Real entry errors
retain S170 rows, recovery restores S171, hooks restore. CPU mode/load tests,
resource guards, CRC/member hashes and validator passed.

RISKS: Conditional server runtime {int(estimate//60)}m{round(estimate%60):02d}s under shared-load scaling;
not an official measurement. Entry proxy transfer remains uncertain.

NEXT: Stopped. No upload, commit or push. See README.md and release.json.
'''
    (RECORDS/'WORKER_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(path=archive['path'],sha256=archive['sha256'],release_ready=True,timing=timing,resources=resources),indent=2),flush=True)

if __name__=='__main__':main()
