"""Bind successful public CUDA QA to exact S156 member identities and validate S160 ZIP."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from candidates.s118_s2rule import build
from tools.validate_submission import validate
from candidates.s160_coll_loc.s160_stage import zip_members, diff

DATA = build.DATA
WORK = DATA / 's160_coll_loc'
STAGE = DATA / 'stage2_s118/S160_loc'
PKG = 'model/stage2/s144/'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def main():
    source = DATA / 'releases/S156_casc.zip'
    root = STAGE / 'candidate'
    staged = read(STAGE / 'stage.json')
    source_id = build.identity(source)
    assert source_id == staged['s160']['source_identity']
    assert source_id['sha256'].startswith('e7fb70a6')
    current = build.tree(root)
    assert current == staged['members'], 'Candidate changed after staging'
    base_members = zip_members(source)
    identity_diff = diff(base_members, current)
    qa = {}
    names = ['base_public_none','candidate_public_none','base_long_none','candidate_long_none',
             'candidate_long_runtime','candidate_long_load','candidate_long_invalid','integration']
    for name in names:
        report = read(WORK / 'qa' / (name + '.json'))
        assert report.get('passed', report.get('validation_ok')), name
        assert not report['network_attempts'], name
        assert report['host_memory']['high_water_bytes'] <= 3 * 1024**3, name
        assert report.get('cuda_peak_reserved', report.get('cuda_peak_reserved_bytes_candidate')) <= 2 * 1024**3, name
        expected = base_members if name.startswith('base_') else current
        for member, sha in report['package_identity'].items():
            assert sha == expected[PKG + member]['sha256'], (name, member)
        if name != 'integration':
            actual_sha = build.identity(WORK / 'qa' / (name + '.csv'))['sha256']
            assert actual_sha == report['csv_sha256'], name
        qa[name] = report
    changes = []
    runtime = {}
    for panel in ('public', 'long'):
        before = csv_rows(WORK / 'qa' / ('base_' + panel + '_none.csv'))
        after = csv_rows(WORK / 'qa' / ('candidate_' + panel + '_none.csv'))
        assert len(before) == len(after)
        for old,new in zip(before,after):
            assert old['ID'] == new['ID']
            for key in old:
                if old[key] != new[key]:
                    assert key in ('collision_frame','entry_frame'), (panel, old['ID'], key)
                    changes.append(dict(panel=panel, ID=old['ID'], field=key, before=old[key], after=new[key]))
        b, c = qa['base_' + panel + '_none'], qa['candidate_' + panel + '_none']
        runtime[panel] = dict(clips=len(before), base_seconds=b['seconds'], candidate_seconds=c['seconds'],
                             added_seconds_per_clip=(c['seconds']-b['seconds'])/len(before),
                             localizer_seconds=c['localizer_seconds'],
                             localizer_rows_seconds=c.get('localizer_rows_seconds', []),
                             localizer_rows_and_scoring_seconds_per_clip=(sum(c['localizer_seconds'])+sum(c.get('localizer_rows_seconds', [])))/len(before))
    base_long = {r['ID']: r for r in csv_rows(WORK / 'qa/base_long_none.csv')}
    faults = {}
    for fault in ('load','runtime','invalid'):
        actual = csv_rows(WORK / 'qa' / ('candidate_long_' + fault + '.csv'))
        assert actual and all(r == base_long[r['ID']] for r in actual), fault
        faults[fault] = dict(rows=len(actual), exact_s156_output=True, ids=[r['ID'] for r in actual])
    independent = {r['ID']: r for r in qa['integration']['rows']}
    for row in csv_rows(WORK / 'qa/candidate_long_none.csv'):
        assert int(row['collision_frame']) == independent[row['ID']]['independent']
    prior = read(DATA / 'stage2_s118/S156_casc_qa/qa.json')
    assert prior['passed'] and prior['candidate_manifest_sha256'] == build.manifest_sha(base_members)
    release = DATA / 'releases/S160_loc.zip'
    assert release.name.isascii() and len(release.name) <= 30
    with zipfile.ZipFile(release, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for member in sorted(current):
            archive.write(root / member, member)
    assert zip_members(release) == current, 'ZIP member identity mismatch'
    with zipfile.ZipFile(release) as archive:
        assert archive.testzip() is None, 'CRC failure'
    errors = validate(release)
    assert not errors, errors
    selection = read(HERE / 'selection.json')
    result = dict(candidate='S160_loc', ready=True, completed_local=datetime.now().astimezone().isoformat(),
                  zip=str(release), **build.identity(release), source_zip=str(source), source_identity=source_id,
                  candidate_manifest_sha256=build.manifest_sha(current), identity_diff=identity_diff,
                  selected=selection['selected'], selection=selection, runtime=runtime, changed_cells=changes,
                  fault_injection=faults, validator_errors=errors, crc_checked=True,
                  zip_members_rehashed=len(current), network_attempts=0,
                  qa={name: {'path':str(WORK / 'qa' / (name + '.json')),
                             'identity':build.identity(WORK / 'qa' / (name + '.json')),
                             'peak_rss_bytes':r['host_memory']['high_water_bytes'],
                             'cuda_peak_reserved_bytes':r.get('cuda_peak_reserved',r.get('cuda_peak_reserved_bytes_candidate'))}
                      for name,r in qa.items()},
                  inherited_stage1_stage3_qa=dict(path=str(DATA / 'stage2_s118/S156_casc_qa/qa.json'),
                       identity=build.identity(DATA / 'stage2_s118/S156_casc_qa/qa.json'),
                       reason='All Stage1/Stage3 sources, weights, entrypoint and dependencies byte-identical; unchanged passing QA reused'),
                  official_whole_inference_runtime_measured=False,
                  private_evaluation_access=False, uploaded=False)
    first_attempt = WORK / 'logs/qa_integration_attempt1.err'
    if first_attempt.is_file():
        result['reference_harness_recovery'] = dict(
            first_attempt_passed=False, error='FileNotFoundError: descriptive E10_base used as checkpoint prefix',
            correction='Use saved E10b checkpoint prefix; no artifact change; rerun final reference only',
            retained_error_log=str(first_attempt), error_log_identity=build.identity(first_attempt),
            final_attempt_passed=True)
    (HERE / 'release.json').write_text(json.dumps(result, indent=2))
    (WORK / 'release.json').write_text(json.dumps(result, indent=2))
    proxy = selection['fixed_oof_proxy']
    lines = ['# S160 collision localizer', '',
             f'Ready: `{release}` ({result["bytes"]:,} bytes).', '',
             f'SHA256: `{result["sha256"]}`', '',
             f'Exact base: `{source}`; SHA256 `{source_id["sha256"]}`.', '',
             '## Selected export and evidence', '', selection['selected'] + '.', '',
             'The finalize log and exported configuration agree. The handoff description of an '
             '`nn_soft1_ep40 halves ensemble` was inaccurate: halves 0/3/6/9 were four alternative decoders. '
             'Those alternatives scored 412/411/419/423 of 750 versus S156 393. The existing final export is retained.', '',
             f'The fixed available-OOF fusion proxy scores {proxy["hits_0p3"]}/750 versus 393/750. '
             'It includes five Conv seeds and only three MLP seeds, while the final model has five of each. '
             'Both final and OOF AdamW weight decay are .001. This is not exact final-ensemble OOF. '
             'Prior architecture/hyperparameter/decoder searches used the same 750 public clips; selection bias remains. '
             'No new search or training was performed for this release. See selection.json for every saved summary.', '',
             'Existing full export parity: 750/750, zero row/logit differences; mean localizer time 0.9358 s/clip '
             '(p95 1.1288 s). The 659/750 final-fit hit count is in-sample and is not evidence of transfer.', '',
             '## Integration and public CUDA QA', '',
             'Only model/stage2/s144/predict.py changed; four localizer files were added. '
             'All 93 other S156 members, including Stage1, Stage3, inference.py, requirements, and S156 cascade weights '
             'are byte-identical. Only collision_frame and entry_frame=min(S109 entry, collision) may differ.', '',
             'Fresh WSL Hermes-Ubuntu CUDA runs passed on five official public sample clips and three public '
             'Nexar long clips. Unchanged Stage1/Stage3 passing QA was reused by exact member identity. '
             'The independent integration_s160.py detector/embedding/sklearn/PyTorch reference agreed on all three '
             'long collision outputs and the S156 starting frames. Offline audit blocked network calls; zero attempts.', '',
             'Three injected localizer failures (weight loading, runtime exception, invalid index/nonfinite output) '
             'returned the complete exact S156 row on a public long clip. Localizer exceptions retain the already '
             'computed S156 collision. Existing base-model errors retain the inherited S156 behavior.', '',
             '| Panel | Clips | S156 seconds | S160 seconds | Added seconds/clip |',
             '|---|---:|---:|---:|---:|']
    for panel,r in runtime.items():
        lines.append(f'| {panel} | {r["clips"]} | {r["base_seconds"]:.3f} | {r["candidate_seconds"]:.3f} | {r["added_seconds_per_clip"]:.3f} |')
    lines += ['', 'These paired end-to-end timings include process-local cold loads and host/GPU scheduling noise; '
              'direct localizer row-construction and scoring/decode timings (excluding package loading) are in release.json. '
              'Whole official inference under 60 minutes is unmeasured.', '',
              f'Observed added localizer row/scoring work on long clips: '
              f'{runtime["long"]["localizer_rows_and_scoring_seconds_per_clip"]:.3f} s/clip. '
              'The negative whole-pipeline elapsed delta is confounded by shared-machine load and cache state; '
              'it is not evidence that adding S160 speeds up S156.', '',
              f'Max observed process RSS: {max(r["peak_rss_bytes"] for r in result["qa"].values())/1024**3:.3f} GiB; '
              f'max CUDA reserved: {max(r["cuda_peak_reserved_bytes"] for r in result["qa"].values())/1024**3:.3f} GiB. '
              'Each QA ran with zero loader workers under the shared qa.lock and video-s160.json 2 GiB GPU lease. '
              'A live RSS guard enforced 3 GiB. These are preparation settings; shipped inference was not changed.', '',
              '## Validation and reproduction', '',
              'ZIP CRC, all 98 member hashes, static tools/validate_submission.py, schema and protected-field checks passed. '
              'Original source licenses/notices and Nexar provenance were retained; S160-NOTICE.txt documents the new fit. '
              'No private evaluation data, upload, commit or push.', '',
              'From repository root, run `.venv/Scripts/python.exe -B candidates/s160_coll_loc/s160_stage.py S160` '
              'in a fresh staging directory, then `candidates/s160_coll_loc/launch_qa.ps1`; after QA, run '
              '`.venv/Scripts/python.exe -B candidates/s160_coll_loc/release.py`. Existing stage/ZIP paths are intentionally '
              'not overwritten. release.json binds the retained QA and artifact hashes.', '']
    lines += ['Stacking note: S164 build.batch_patch applies cleanly to this S160 predict.py and the combined source '
              'compiles. Both arms modify that file, so carry both changes when stacking. Combined exported runtime '
              'still needs its own check.', '']
    if first_attempt.is_file():
        lines += ['Reference harness recovery: the first attempt used descriptive ensemble name E10_base as the '
                  'checkpoint prefix and failed with FileNotFoundError. The saved models use E10b. The corrected '
                  'reference passed on the unchanged artifact; first-attempt logs are retained under '
                  '$DATA_DIR/s160_coll_loc/logs/qa_integration_attempt1.*.', '']
    (HERE / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    report = ['RESULT done', '', f'CHANGED: Built `{release}` from exact S156; one modified and four added ZIP members. '
              'Selected existing two-HGB/ten-network 25-epoch export; corrected handoff model description.', '',
              'VERIFIED: Eight offline CUDA QA cases passed; public short/long fields, independent reference, three '
              'exact-S156 fault fallbacks, 98 member identities, CRC and static validator. Stage1/3 unchanged QA reused. '
              f'Localizer row/scoring work {runtime["long"]["localizer_rows_and_scoring_seconds_per_clip"]:.3f} s/clip; '
              f'SHA256 `{result["sha256"]}`. Completed {result["completed_local"]}.', '',
              f'RISKS: Fixed OOF proxy {proxy["hits_0p3"]}/750 vs 393, not exact final ensemble; missing MLP-seed OOF '
              'and prior selection bias documented. Whole official runtime and score unmeasured.', '',
              'NEXT: Orchestrator may stack the five collision-package changes and recheck combined runtime; upload remains manual.', '']
    (HERE / 'WORKER_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('ready','zip','bytes','sha256','runtime','changed_cells')}), flush=True)


if __name__ == '__main__':
    main()
