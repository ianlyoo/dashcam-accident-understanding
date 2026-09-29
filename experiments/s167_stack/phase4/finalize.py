"""Gate S175 packaging on exact input identities and every CUDA acceptance row."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

H=Path(__file__).resolve().parent
REPO=H.parents[2]
sys.path.insert(0,str(REPO))
from tools.validate_submission import validate
from candidates.s167_stack.build_stack import members,sha

D=Path('$DATA_DIR')
W=D/'s167_stack/phase4'
ROOT=W/'candidate'
BASES={'S171_ent.zip':'f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315',
       'S172_vis.zip':'2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549',
       'S174_cov.zip':'d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc'}
JOBS=[('stage2','s172','public'),('stage2','s175','public'),('independent','s175','public'),
      ('stage2','s172','long'),('stage2','s175','long'),('independent','s175','long'),
      ('stage2','s172','cascade'),('stage2','s175','cascade'),('independent','s175','cascade'),
      ('stage2','s172','entry'),('stage2','s175','entry'),('independent','s175','entry'),
      ('fault','s175','entry'),('stage1','s175','public'),('stage1','s175','extras'),
      ('stage3','s175','public')]


def integrity():
    build=json.loads((W/'build.json').read_text())
    assert len(build['members'])==118
    for filename,digest in BASES.items():
        assert sha(D/'releases'/filename)==digest
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    assert actual==build['members']
    with zipfile.ZipFile(D/'releases/S172_vis.zip') as carrier,zipfile.ZipFile(D/'releases/S174_cov.zip') as arm:
        carrier_members=members(carrier)
        arm_members=members(arm)
        assert set(carrier_members)==set(actual)-set(build['added_to_carrier'])
        assert {k for k in carrier_members if carrier_members[k]!=actual[k]}==set(build['changed_from_carrier'])
        for name in build['arm_changed']+build['arm_added']:
            assert actual[name]==arm_members[name]
        for name in carrier_members:
            if name not in build['changed_from_carrier']:
                assert actual[name]==carrier_members[name]
        assert (ROOT/'inference.py').read_bytes().startswith(carrier.read('inference.py'))
    assert build['postprocessor_sha256']==sha(H/'s175_entry.py')==actual['model/stage2/s175_entry.py']
    assert build['build_script_sha256']==sha(H.parent/'build_stack.py')
    return build


def qa(build):
    import pandas as pd
    reports={}
    for mode,tree,panel in JOBS:
        name=f'{mode}_{tree}_{panel}'
        path=W/'qa'/f'{name}.json'
        report=json.loads(path.read_text())
        assert report['passed'] and (report['mode'],report['tree'],report['panel'])==(mode,tree,panel)
        assert report['build_sha256']==sha(W/'build.json')
        prior=(H/'harness_before_fault/qa.py')
        assert report['qa_source_sha256'] in {sha(H/'qa.py'),sha(prior)}
        if name in ('stage2_s172_public','stage2_s175_public','independent_s175_public',
                    'stage2_s172_long','stage2_s175_long','independent_s175_long',
                    'stage2_s172_cascade','stage2_s175_cascade','independent_s175_cascade',
                    'stage2_s172_entry','stage2_s175_entry'):
            assert report['qa_source_sha256']==sha(prior)
        assert report['monitor_sha256']==sha(H.parent/'monitor_phase3.py')
        root=D/'s172_vis_coll/revision3/candidate' if tree=='s172' else ROOT
        assert report['artifact_sha256']==sha(root/'inference.py')
        assert report['s161_sha256']==sha(root/'model/stage2/s161/predict.py')
        resources=report['resources']
        assert resources['peak_aggregate_rss_bytes']<=4*2**30
        assert resources['minimum_host_free_commit_gib']>=12
        assert resources['peak_processes']<=2 and report['cuda_peak_reserved']<=2*2**30
        assert not report['network_attempts']
        if mode!='independent':
            csv=W/'qa'/f'{name}.csv'
            assert report['detail']['csv_sha256']==sha(csv)
            frame=pd.read_csv(csv,dtype={'ID':str})
            if mode=='stage2':
                if tree=='s175':
                    ref=pd.read_csv(W/'qa'/f'stage2_s172_{panel}.csv',dtype={'ID':str})
                    assert frame.drop(columns='entry_frame').equals(ref.drop(columns='entry_frame'))
                    assert all(frame.entry_frame<=frame.collision_frame)
                    assert not report['detail']['s175'].get('load_error')
                    assert not any(r.get('error') for r in report['detail']['s175']['clips'].values())
                if tree=='s172' and panel in ('public','long'):
                    assert csv.read_bytes()==(D/'s172_vis_coll/revision3/qa'/f'{panel}.csv').read_bytes()
            elif mode=='stage1':
                assert csv.read_bytes()==(D/'s167_stack/phase3/qa'/f'stage1_{panel}.csv').read_bytes()
            elif mode=='stage3':
                assert len(frame)==2998
                assert csv.read_bytes()==(D/'s167_stack/phase3/qa/stage3_public.csv').read_bytes()
            elif mode=='fault':
                assert report['detail']['fault_source_panel']=='cascade'
                assert csv.read_bytes()==(W/'qa/stage2_s175_cascade.csv').read_bytes()
                injected=W/'qa'/f'{name}_injected.csv'
                assert injected.read_bytes()==(W/'qa/stage2_s172_cascade.csv').read_bytes()
                assert report['detail']['changed_entry_rows']>0
        else:
            assert report['detail']['pinned_s174_sha256']==sha(H/'source/S174_cov/model/stage2/s161/predict.py')
            candidate=W/'qa'/f'stage2_s175_{panel}.csv'
            reference=W/'qa'/f'stage2_s172_{panel}.csv'
            assert report['detail']['candidate_csv_sha256']==sha(candidate)
            assert report['detail']['reference_csv_sha256']==sha(reference)
            got=pd.read_csv(candidate,dtype={'ID':str})
            assert got.entry_frame.tolist()==[r['expected_entry'] for r in report['detail']['rows']]
            assert all(r['collision']==int(v) for r,v in zip(report['detail']['rows'],got.collision_frame))
        reports[name]=report
    return reports


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--integrity-only',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    build=integrity()
    if args.integrity_only:
        print('S175 integrity PASS: exact S172 carrier with S174 entry-only arm and reviewed dispatcher')
        return
    reports=qa(build)
    if args.check_only:
        print('S175 acceptance PASS: exact overlay, Stage1/3 parity, S172 protected fields, independent S174 replay, fault/recovery, resource caps')
        return
    archive=json.loads((W/'archive.json').read_text())
    path=Path(archive['path'])
    assert archive['crc_and_member_hashes_passed'] and archive['sha256']==sha(path)
    assert archive['build_sha256']==sha(W/'build.json') and archive['members']==len(build['members'])
    errors=validate(path)
    assert not errors,errors
    s172=reports['stage2_s172_long']['detail']['seconds']
    s175=reports['stage2_s175_long']['detail']['seconds']
    measured_delta=s175-s172
    # The current shared GPU load made full S175 faster than the separate S172
    # run, so their wall difference is not a causal estimate. The S175-specific
    # tracker is timed per clip inside the exported path; scale that work.
    entry_rows=reports['stage2_s175_long']['detail']['s175']['clips']
    entry_seconds=sum(row['seconds'] for row in entry_rows.values())
    prior_tracker_load=json.loads((D/'s167_stack/phase3/qa/stage2_long.json').read_text())['s118']['load_seconds']
    assert entry_seconds>0 and prior_tracker_load>0
    estimate=34*60+18+137/len(entry_rows)*entry_seconds+prior_tracker_load
    peak_rss=max(x['resources']['peak_aggregate_rss_bytes'] for x in reports.values())
    peak_cuda=max(x['cuda_peak_reserved'] for x in reports.values())
    min_commit=min(x['resources']['minimum_host_free_commit_gib'] for x in reports.values())
    receipt=dict(archive,release_ready=True,carrier_official_score=0.66245,
                 s171_official_score=0.65632,s172_official_runtime_seconds=2058,
                 official_s175_result=None,uploaded=False,
                 qa_reports={name:dict(seconds=r['detail']['seconds'],
                    peak_rss=r['resources']['peak_aggregate_rss_bytes'],
                    min_free_commit=r['resources']['minimum_host_free_commit_gib'],
                    peak_cuda=r['cuda_peak_reserved']) for name,r in reports.items()},
                 timing=dict(matched_long_s172_seconds=s172,matched_long_s175_seconds=s175,
                             matched_long_delta_seconds=measured_delta,
                             s175_entry_pass_seconds=entry_seconds,
                             estimated_tracker_load_seconds=prior_tracker_load,
                             conditional_server_seconds=estimate,
                             s174_local_137_clip_projection_seconds=74.62633222423061,
                             caveat='S172 official runtime plus measured S175-only entry-pass seconds on three public long clips scaled to 137, plus prior observed tracker load. Separate full-panel wall times are confounded by changing shared GPU load. No S175 server measurement; stage mix and hardware differ.'),
                 resources=dict(rss_cap_gib=4,peak_aggregate_rss_bytes=peak_rss,
                                peak_cuda_reserved_bytes=peak_cuda,max_processes=2,
                                minimum_host_free_commit_gib=min_commit),
                 validator_errors=errors,build_sha256=sha(W/'build.json'),
                 qa_source_sha256=sha(H/'qa.py'),launcher_sha256=sha(H/'launch.ps1'))
    (H/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
    minutes=int(estimate//60);seconds=round(estimate%60)
    readme=f'''# S175 S172 + S174 stack

Release: `{path.as_posix()}`. SHA256 `{archive['sha256']}`.
Bytes {archive['bytes']}; {archive['members']} members. S172 carrier official
score 0.66245 and 34m18s; S175 has no official result or upload.

`build_stack.py --action overlay` uses exact SHA-pinned S171 as the common base,
S172 as carrier, and S174 as the entry-only arm. S172's member delta is
`inference.py`, `model/stage2/s144/predict.py`, and twelve additions; S174
changes only `model/stage2/s161/predict.py` and adds its notice. These deltas
do not overlap. S175 adds a tiny dispatcher and postprocessor; all other S172
members are byte-identical. The pinned S174 code is byte-identical to its ZIP.

Final order: complete exact S172 Stage2 prediction, including visual DINOv2-L
collision; run S161/S174 entry on each final S172 collision anchor; select an
accepted crossing or retain S172 entry, then clamp to S172 collision. A tracker
load, clip, or close error retains the exact S172 entry. Collision, side and
evasion remain exact S172. S172's original S161 pass is retained to provide
the exact fallback; the final entry pass adds work and is included in timing.

QA: S172 and S175 exported public, long, cascade and entry panels; protected
Stage2 columns identical; independent pinned-S174 replay on S172 collisions
matched every entry. Cascade-panel injected tracker errors returned exact S172
rows, with two observable changes after recovery, and
same-process recovery returned S175. Stage1 13 clips and Stage3 2998 rows
match S172 by unchanged members and fresh outputs. Network audit, at most two
processes, <=2 GiB CUDA lease, <=4 GiB aggregate RSS, >=12 GiB host free
commit, streamed member CRC/SHA256 and static validator passed. `qa.lock`,
GPU lease and `gpu.request` per-clip yielding were used.

Matched long panel S172 {s172:.2f}s -> S175 {s175:.2f}s; the negative wall
difference {measured_delta:.2f}s reflects changing shared load and is not
credited as a speedup. The isolated S175 final entry pass took
{entry_seconds:.2f}s for three clips. Conditional runtime: {minutes}m{seconds:02d}s
from official S172 34m18s plus that pass scaled to 137 clips and a
{prior_tracker_load:.2f}s prior observed tracker load. S174 standalone local
projection was 74.63s/137 clips; S175 also repeats S161 on final collisions.
Shared RTX load, hidden stage mix and server hardware make this an estimate,
not an official runtime. Peak RSS {peak_rss/2**30:.3f} GiB, CUDA reserved
{peak_cuda/2**30:.3f} GiB, minimum free commit {min_commit:.2f} GiB.
No private evaluation data, upload, commit or push.
'''
    (H/'README.md').write_text(readme)
    report=f'''RESULT done

CHANGED: S175_stk.zip = exact S172 + pinned S174 entry and reviewed postprocessor.
SHA256: {archive['sha256']}

VERIFIED: S172 Stage1/3 and protected Stage2 fields identical; independent
S174 replay on final S172 collision anchors; error retains S172 entry and
same-process recovery passes. CRC/member SHA256, validator, resource and
network checks pass.

RISKS: Conditional server runtime {minutes}m{seconds:02d}s from shared-RTX panels;
no S175 official result. Entry coverage proxy transfer unproven.

NEXT: Stop. No upload, commit or push. See README.md/release.json.
'''
    (H/'WORKER_REPORT.md').write_text(report)
    print(json.dumps(dict(sha256=archive['sha256'],path=str(path),runtime_seconds=estimate,
                          release_ready=True,resources=receipt['resources']),indent=2))

if __name__=='__main__':main()
