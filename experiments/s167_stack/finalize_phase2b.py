"""Gate S170 packaging on bound CUDA evidence; finalize only a validated ZIP."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from candidates.s167_stack.build_stack import sha
from tools.validate_submission import validate
DATA=Path('$DATA_DIR')
WORK=DATA/'s167_stack/phase2b'
RECORDS=HERE/'phase2b'
PANELS=['stage1_public','stage1_extras','stage2_public','stage2_long','stage2_cascade','stage2_fault','stage3_public']

def integrity():
    build=json.loads((WORK/'build.json').read_text())
    merge=json.loads((WORK/'cpu_merge.json').read_text())
    cpu=json.loads((WORK/'cpu_fault.json').read_text())
    assert merge['build_sha256']==cpu['build_sha256']==sha(WORK/'build.json')
    assert cpu['passed'] and cpu['source_sha256']==sha(HERE/'cpu_phase2b.py')
    assert merge['exact_batch_patch_of_s160'] and merge['unchanged_from_s169']==96
    assert merge['changed_from_s169']==['model/stage2/s144/predict.py']
    for folder,expected in [('candidate',build['members']),('reference_s160',merge['reference_s160_members'])]:
        root=WORK/folder
        actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
        assert actual==expected, 'Tree changed: '+folder
    return build,merge,cpu

def check_qa(build,merge):
    import pandas as pd
    reports={}
    for name in ['reference_stage2_cascade']+PANELS:
        isref=name.startswith('reference_')
        label=name.removeprefix('reference_')
        folder='qa_reference' if isref else 'qa'
        path=WORK/folder/(label+'.json')
        r=json.loads(path.read_text())
        identity=merge['reference_s160_members'] if isref else build['members']
        assert r['validation_ok'] and r['byte_identical_to_reference']
        for field,member in [('inference_sha256','inference.py'),('s144_source_sha256','model/stage2/s144/predict.py'),('s162_source_sha256','model/stage3/s141/predict.py')]:
            assert r[field]==identity[member], (name,field)
        revised_cap = label=='stage2_fault' or label.startswith(('stage1_', 'stage3_'))
        assert r['qa_source_sha256']==sha(HERE/('qa_phase2b_4g.py' if revised_cap else 'qa_phase2b.py'))
        assert r['build_sha256']==sha(WORK/'build.json')
        assert not r['network_attempts']
        assert r['resources']['peak_aggregate_rss_bytes']<=(4 if revised_cap else 3)*2**30 and r['resources']['peak_processes']<=2
        if revised_cap:
            assert r['resources']['rss_cap_gib']==4
            assert r['resources']['minimum_host_free_commit_gib']>=12
        assert r['cuda_peak_reserved']<=2*2**30
        csv=WORK/folder/(label+'.csv')
        assert sha(csv)==r['csv_sha256']
        reference=Path(r['reference_csv'].replace('/mnt/d/','$DATA_DIR/'))
        assert csv.read_bytes()==reference.read_bytes()
        r['reference_sha256']=sha(reference)
        r['qa_report_sha256']=sha(path)
        if not isref and label.startswith(('stage2_', 'stage3_')):
            assert r['s164']['runtime_sha256']==build['members']['model/runtime_s164/runtime.py']
            assert not r['s164'].get('errors') and not r['s164'].get('lazy_scene_error')
            assert r['s164']['unused_features']==0
        if label.startswith('stage2_'):
            assert r['protected_fields_equal_s156']
        reports[name]=r
    assert reports['reference_stage2_cascade']['reference_generation_only']
    fault=reports['stage2_fault']['fault_injection']
    assert fault['next_call_recovers_s160'] and fault['all_hooks_restored']
    assert sorted(fault['injected_ids'])==['00019','00060']
    assert fault['fault_s164']['prefetched']==3 and fault['fault_s164']['unused_features']==0
    assert sha(WORK/'qa/stage2_fault_injected.csv')==fault['fault_csv_sha256']
    failed=pd.read_csv(WORK/'qa/stage2_fault_injected.csv',dtype={'ID':str})
    base=pd.read_csv(DATA/'s164_runtime/qa_base/stage2_long.csv',dtype={'ID':str})
    for ident in fault['injected_ids']:
        assert failed[failed.ID==ident].reset_index(drop=True).equals(base[base.ID==ident].reset_index(drop=True))
    s3=reports['stage3_public']
    assert s3['rows']==2998 and s3['changed_accel_rows']==68 and s3['steering_equal_s156'] and s3['stopped_equal_s156']
    assert s3['s164']['observer_replay_frames']==2998
    return reports

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--integrity-only',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args()
    build,merge,cpu=integrity()
    if a.integrity_only:
        print('CPU integrity passed: 101 candidate members, pinned S160 reference, CPU fault evidence')
        return
    reports=check_qa(build,merge)
    if a.check_only:
        print('CUDA evidence gate passed; packaging permitted')
        return
    archive=json.loads((WORK/'archive.json').read_text())
    assert archive['build_sha256']==sha(WORK/'build.json')
    assert archive['sha256']==sha(archive['path']) and archive['crc_and_member_hashes_passed']
    errors=validate(Path(archive['path']))
    assert not errors,errors
    s164=json.loads((HERE.parent/'s164_runtime/release.json').read_text())
    baseline={p:s164['comparisons'][p]['base_seconds'] for p in ('stage1_public','stage2_public','stage2_long','stage2_cascade')}
    baseline['stage3_public']=json.loads((DATA/'s162_accel_dom/capture/report.json').read_text())['public']['seconds']
    total=sum(reports[p]['seconds'] for p in baseline)
    estimate=2980*total/sum(baseline.values())
    timing=dict(base_seconds=baseline,stack_seconds={p:reports[p]['seconds'] for p in baseline},
        base_total_seconds=sum(baseline.values()),stack_total_seconds=total,
        saved_fraction=1-total/sum(baseline.values()),ratio_scaled_server_seconds=estimate,
        inherited_server_base_seconds=2980,
        caveat='Historical matched panels under variable shared load, not a measured server result. Stage3 is 2998 rows at 10Hz, not the earlier 5992-row native panel. Framework setup, extras, reference and faults excluded.')
    receipt=dict(archive,candidate='S170',phase='2b',release_ready=True,completed_at=datetime.now().astimezone().isoformat(),
        static_validation_errors=errors,merge=merge,cpu_fault=cpu,qa_reports=reports,timing=timing,
        source_sha256={p.name:sha(p) for p in [HERE/'build_stack.py',*HERE.glob('*phase2b*.py'),*HERE.glob('*phase2b.ps1')]},
        infrastructure_history=['Two recovery attempts exceeded the original 3 GiB machine-sharing guard: 3260489728 and 3222745088 bytes. Injected-output parity passed both times. Both attempts and logs are preserved under qa/attempt1, qa/attempt2 and logs/*attempt*. team explicitly approved 4 GiB RSS for this QA on 2026-09-28, recognizing 3 GiB was a local sharing guard, not a contest/server limit. Free host commit must remain >=12 GiB; checked before launch and monitored every 5 seconds. Separate 4 GiB QA source retains allocator trimming. Candidate unchanged.'],
        resources=dict(rss_cap_gib=4, earlier_normal_stage2_rss_cap_gib=3,
            peak_aggregate_rss_bytes=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
            peak_cuda_reserved_bytes=max(r['cuda_peak_reserved'] for r in reports.values()),
            max_processes=max(r['resources']['peak_processes'] for r in reports.values()),
            minimum_host_free_commit_gib=min(r['resources']['minimum_host_free_commit_gib'] for r in reports.values() if 'minimum_host_free_commit_gib' in r['resources'])),
        monitor_history='First 4 GiB attempt stopped on a Windows/WSL host-commit file sharing read race, at RSS 3287781376 bytes and minimum free commit 52.554 GiB. Preserved as attempt3. Reader now tolerates transient sharing errors for at most 30 seconds since the last valid sample; stale or low commit still stops QA.',
        uploaded=False,official_score=None)
    RECORDS.mkdir(exist_ok=True)
    (RECORDS/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (RECORDS/'member_manifest.json').write_text(json.dumps(build,indent=2)+'\n')
    summary=f'''S170_all.zip: {archive['path']}
SHA256: {archive['sha256']}
Bytes: {archive['bytes']}; members: {archive['members']}
Exact S169 plus S160 collision localizer. One changed S144 Python file and four additions;
96 S169 members unchanged. CRLF normalization resolves the S164/S160 textual overlap;
merged source equals S164 batch_patch applied to pinned S160 source. Default detector batch=1.

Real offline CUDA Stage1 equals S163; Stage2 public/long equals retained S160,
cascade equals a fresh pinned S160 reference; other Stage2 fields equal S156.
Stage3 equals S162 (2998 rows, 68 acceleration changes, steering/STOPPED unchanged).
Localizer exceptions on 00019 and worker-prefetched 00060 return complete S156 rows;
00019 collision restores 609->612. Unaffected 00076 remains 597. Next call recovers S160.
S164 lazy path and prefetch remain active, hooks restore. CPU fixtures additionally
exercise batch=2 detector retry and invalid localizer output. Actual CUDA uses batch=1.

Historical matched total {sum(baseline.values()):.2f}s -> {total:.2f}s;
conditional server estimate {int(estimate//60)}m{round(estimate%60):02d}s from 49m40s.
Not an official runtime prediction: shared load and unseen stage mix remain uncertain.
Long/cascade timings and all resource measurements are in release.json.
Two fault recovery attempts hit the original 3 GiB aggregate RAM guard at 3.037
and 3.001 GiB; both were stopped and their logs/injected CSVs are preserved.
The team explicitly approved a revised 4 GiB RSS cap on 2026-09-28 because 3 GiB
was a local machine-sharing guard, not a competition/server limit. Free host
commit was checked before each remaining run and every 5 seconds thereafter;
the QA guard required >=12 GiB. Allocator trimming was retained; the candidate
was unchanged. This is a documented criterion revision, not an unchanged pass.
The first revised-cap attempt stopped on a transient Windows/WSL commit-file
read race, while RSS and free commit both met the approved limits. That attempt
is retained; the reader now tolerates sharing races for at most 30 seconds
since the last valid sample, while preserving the low/stale-commit guard.
Final peak aggregate RSS: {receipt['resources']['peak_aggregate_rss_bytes']/2**30:.3f} GiB;
CUDA reserved: {receipt['resources']['peak_cuda_reserved_bytes']/2**30:.3f} GiB;
minimum monitored free host commit: {receipt['resources']['minimum_host_free_commit_gib']:.2f} GiB.
No private data, upload, commit or push. S160 proxy selection bias and inherited
S162/S163 accuracy risks remain; no hidden-score improvement is claimed.
'''
    (RECORDS/'README.md').write_text(summary,encoding='utf-8')
    (RECORDS/'WORKER_REPORT.md').write_text('RESULT done\n\nCHANGED: '+summary+'\nVERIFIED: CUDA parity, faults, caps, CRC/member hashes and validator passed.\nRISKS: Transfer and official runtime unmeasured.\nNEXT: Stopped; upload remains manual.\n',encoding='utf-8')
    print(json.dumps(dict(path=archive['path'],sha256=archive['sha256'],release_ready=True,timing=timing),indent=2))

if __name__=='__main__': main()
