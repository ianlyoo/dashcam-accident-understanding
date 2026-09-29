"""Release only a selected and actually validated S172 tree."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from build import sha
from revision_path import revision_work

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
WORK=revision_work(Path('$DATA_DIR/s172_vis_coll'))
def main():
    root=WORK/'candidate'
    build=json.loads((WORK/'build.json').read_text())
    parity=None; fp32=None
    if WORK.name=='revision3':
        assert build['speed_revision']['rate_oof']==json.loads((HERE/'rate_2p5.json').read_text())
        parity=json.loads((HERE/'parity_150.json').read_text())
        assert parity['clips']==150 and len(parity['picks'])==150
        fp32=json.loads((HERE/'fp32_150.json').read_text())
        assert fp32['clips']==150 and len(fp32['picks'])==150
    qa={panel:json.loads((WORK/f'qa/{panel}.json').read_text()) for panel in ('public','long')}
    assert all(r['passed'] and r['build_sha256']==sha(WORK/'build.json') for r in qa.values())
    assert all(r['qa_source_sha256']==sha(HERE/'qa.py') for r in qa.values())
    assert all(r['resources']['peak_aggregate_rss_bytes']<=4*2**30 for r in qa.values())
    assert all(r['resources']['minimum_host_free_commit_gib']>=12 for r in qa.values())
    assert all(r['cuda_peak_reserved_gib']<=1.65 for r in qa.values())
    assert build['config']['selected_oof_hits']>468
    if WORK.name=='revision3':
        assert build['config']['selected_oof_hits']>=480
        assert qa['long']['guard']['exact_s171']
        assert qa['long']['guard']['kind']=='process_wall_45m'
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual==build['members'],'Tree changed after build'
    # Unchanged Stage1/3 implementation and weights reuse actual exact-S171 QA.
    with zipfile.ZipFile(build['base']) as base:
        original=base.read('inference.py')
        assert (root/'inference.py').read_bytes().startswith(original)
        for name in base.namelist():
            if name.startswith(('model/stage1','model/stage3')) and not name.endswith('/'):
                assert hashlib.sha256(base.read(name)).hexdigest()==actual[name]
    old=json.loads((REPO/'candidates/s167_stack/phase3/release.json').read_text())
    assert old['sha256']==build['base_sha256'] and old['release_ready']
    timings=[r['seconds'] for r in qa['long']['diagnostics']['clips'].values()]
    runtime=dict(qa_long_mean_seconds=sum(timings)/len(timings),qa_long_max_seconds=max(timings),
        qa_load_seconds=qa['long']['diagnostics']['load_seconds'],clips=len(timings),
        server_l40s_measured=False)
    runtime['prior_capture_seconds_per_clip']=qa['long']['prior_capture_seconds']/len(timings)
    if WORK.name=='revision3':
        measured=json.loads((WORK/'qa/benchmark.json').read_text())
        assert measured['passed'] and measured['build_sha256']==sha(WORK/'build.json')
        assert measured['source_sha256']==sha(HERE/'bench_runtime.py')
        assert measured['resources']['peak_aggregate_rss_bytes']<=4*2**30
        assert measured['resources']['minimum_host_free_commit_gib']>=12
        assert measured['cuda_peak_reserved_gib']<=1.65
        assert measured['diagnostics']['budget_triggered'] is None
        assert measured['diagnostics']['covered_clips']==3
        runtime.update(local_long_mean_seconds=measured['local_long_mean_seconds'],
            local_long_max_seconds=measured['local_long_max_seconds'],
            fixed_overhead_seconds=measured['fixed_overhead_seconds'],
            conditional_137_clip_seconds=measured['conditional_137_clip_seconds'],
            benchmark=measured,hard_guard_s172_seconds=600,hard_guard_process_seconds=2700,
            source='same exported arm, one WSL three-clip pass after QA, public OOF priors for timing only')
        assert runtime['conditional_137_clip_seconds']<=480,'Measured local 137-clip projection exceeds eight minutes'
    else:
        runtime['local_long_mean_seconds']=runtime['qa_long_mean_seconds']
        runtime['local_long_max_seconds']=runtime['qa_long_max_seconds']
        runtime['fixed_overhead_seconds']=max(runtime['qa_load_seconds'],
            qa['long']['added_seconds']-qa['long']['prior_capture_seconds']-sum(timings))
        runtime['conditional_137_clip_seconds']=137*(runtime['local_long_mean_seconds']+runtime['prior_capture_seconds_per_clip'])+runtime['fixed_overhead_seconds']
        assert runtime['conditional_137_clip_seconds']<=480,'Added 8-minute allowance not established locally'
    dest=Path('$DATA_DIR/releases/S172_vis.zip')
    assert not dest.exists(),'Preserve existing release'
    pending=WORK/'S172_vis.zip'
    if not pending.exists():
        from pack_pending import prepare_archive
        prepare_archive()
    archive_record=json.loads((WORK/'pending_archive.json').read_text())
    assert archive_record['build_sha256']==sha(WORK/'build.json')
    assert archive_record['sha256']==sha(pending),'Pending archive changed after integrity validation'
    assert archive_record['crc_and_member_hashes_passed'] and not archive_record['validator_errors']
    errors=archive_record['validator_errors']
    assert not dest.exists(),'Release appeared during validation'
    pending.rename(dest)
    receipt=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(actual),
        base_sha256=build['base_sha256'],qa=qa,runtime=runtime,changed=build['changed'],added=build['added'],
        validator_errors=errors,stage1_stage3_reused_by_identity=True,release_ready=True,
        selection=build['selection'],official_gain_measured=False,
        precision_conversion=build.get('precision_conversion'),
        initial_qa_failure=build.get('initial_qa_failure'),parity_150=parity,fp32_150=fp32,
        build_sha256=sha(WORK/'build.json'),
        final_source_sha256={name:sha(HERE/name) for name in
            ('qa.py','qa_supervise.py','half_revision.py','pack_pending.py','release.py','revision_path.py',
             'runtime_fast.py','rate_oof.py','stage_fast.py','bench_runtime.py','bench_supervise.py',
             'test_guard.py','parity_150.py','score_parity_150.py','parity_fp32_150.py','score_fp32_150.py')})
    (HERE/'release.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps({k:receipt[k] for k in ('path','sha256','bytes','members','release_ready')}),flush=True)
if __name__=='__main__': main()
