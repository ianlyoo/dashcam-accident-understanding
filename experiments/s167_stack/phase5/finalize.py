"""Gate S177 on exact overlay identity, CUDA receipts, archive and validator."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

H = Path(__file__).resolve().parent
D = Path('$DATA_DIR')
W = D/'s167_stack/phase5'
ROOT = W/'candidate'
ZIP = D/'releases/S177_bkt.zip'
PANELS = ('public', 'bucket', 'long', 'cascade', 'entry')
BASES = {
    'S174_cov.zip': 'd2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc',
    'S175_stk.zip': '3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be',
    'S176_bucket.zip': 'fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def members(archive):
    found = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                digest.update(block)
        assert info.filename not in found
        found[info.filename] = digest.hexdigest()
    return found


def integrity():
    for name, expected in BASES.items():
        assert sha(D/'releases'/name) == expected, name
    build = json.loads((W/'build.json').read_text())
    assert build['base_sha256'] == BASES['S174_cov.zip']
    assert build['carrier_sha256'] == BASES['S175_stk.zip']
    assert build['arm_sha256'] == BASES['S176_bucket.zip']
    assert build['arm_changed'] == ['model/stage2/s161/predict.py']
    assert build['arm_added'] == ['model/licenses/S176-NOTICE.txt']
    assert build['changed_from_carrier'] == ['inference.py']
    assert set(build['added_to_carrier']) == {
        'model/licenses/S176-NOTICE.txt',
        'model/stage2/s176_predict.py',
        'model/stage2/s177_entry.py',
    }
    actual = {path.relative_to(ROOT).as_posix(): sha(path)
              for path in ROOT.rglob('*') if path.is_file()}
    assert actual == build['members'] and len(actual) == 121
    assert build['postprocessor_sha256'] == sha(H/'s177_entry.py')
    with zipfile.ZipFile(D/'releases/S175_stk.zip') as carrier, \
         zipfile.ZipFile(D/'releases/S176_bucket.zip') as arm:
        old = members(carrier)
        assert set(old) <= set(actual)
        assert all(actual[name] == digest for name, digest in old.items()
                   if name != 'inference.py')
        assert ROOT.joinpath('inference.py').read_bytes().startswith(
            carrier.read('inference.py'))
        assert actual['model/stage2/s176_predict.py'] == hashlib.sha256(
            arm.read('model/stage2/s161/predict.py')).hexdigest()
        assert actual['model/licenses/S176-NOTICE.txt'] == hashlib.sha256(
            arm.read('model/licenses/S176-NOTICE.txt')).hexdigest()
    return build


def acceptance():
    build = integrity()
    reports = {}
    peak_rss = peak_cuda = 0
    min_commit = float('inf')
    for mode, panel in [(mode, panel) for panel in PANELS
                        for mode in ('stage2', 'independent')] + [('fault', 'bucket')]:
        name = f'{mode}_{panel}'
        report = json.loads((W/'qa'/f'{name}.json').read_text())
        assert report['passed'] and (report['mode'], report['panel']) == (mode, panel)
        assert report['build_sha256'] == sha(W/'build.json')
        assert report['qa_source_sha256'] == sha(H/'qa.py')
        assert report['artifact_sha256'] == sha(ROOT/'inference.py')
        assert report['s176_sha256'] == sha(ROOT/'model/stage2/s176_predict.py')
        assert not report['network_attempts']
        resources = report['resources']
        assert resources['peak_aggregate_rss_bytes'] <= 4*2**30
        assert resources['peak_processes'] <= 2
        assert resources['minimum_host_free_commit_gib'] >= 12
        assert report['cuda_peak_reserved'] <= 2*2**30
        peak_rss = max(peak_rss, resources['peak_aggregate_rss_bytes'])
        peak_cuda = max(peak_cuda, report['cuda_peak_reserved'])
        min_commit = min(min_commit, resources['minimum_host_free_commit_gib'])
        reports[name] = report
    total_changed = 0
    for panel in PANELS:
        baseline_path = (W/'qa/stage2_bucket_baseline.csv' if panel == 'bucket'
                         else D/'s167_stack/phase4/qa'/f'stage2_s175_{panel}.csv')
        base = pd.read_csv(baseline_path, dtype={'ID': str})
        got = pd.read_csv(W/'qa'/f'stage2_{panel}.csv', dtype={'ID': str})
        protected = [column for column in base if column != 'entry_frame']
        assert got[protected].equals(base[protected])
        assert all(got.entry_frame <= got.collision_frame)
        changed = int((got.entry_frame != base.entry_frame).sum())
        assert changed == reports[f'stage2_{panel}']['detail']['changed_entry_rows']
        assert changed == reports[f'independent_{panel}']['detail']['changed_entry_rows']
        assert got.entry_frame.tolist() == [row['expected_entry'] for row in
                                            reports[f'independent_{panel}']['detail']['rows']]
        assert sha(W/'qa'/f'stage2_{panel}.csv') == reports[f'stage2_{panel}']['detail']['csv_sha256']
        total_changed += changed
    assert reports['stage2_bucket']['detail']['changed_entry_rows'] > 0, \
        'S176 bucket panel did not exercise a natural entry change'
    fault = reports['fault_bucket']['detail']
    assert fault['fault_calls'] > 0 and fault['fallback_exact_s175']
    assert fault['recovery_exact_s177']
    assert total_changed > 0
    return build, reports, dict(peak_rss=peak_rss, peak_cuda=peak_cuda,
                                min_commit=min_commit, changed=total_changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrity-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.integrity_only:
        integrity()
        print('S177 integrity PASS: exact S175 carrier plus pinned S176 bucket-only pass')
        return
    build, reports, totals = acceptance()
    if args.check_only:
        print('S177 acceptance PASS: identity, protected parity, independent bucket replay, fault/recovery, resources')
        return
    archive = json.loads((W/'archive.json').read_text())
    assert archive['crc_and_member_hashes_passed'] and archive['sha256'] == sha(ZIP)
    assert archive['members'] == 121 and archive['build_sha256'] == sha(W/'build.json')
    assert ZIP.name.isascii() and len(ZIP.name) <= 30
    with zipfile.ZipFile(ZIP) as packed:
        assert members(packed) == build['members']
    validator_bytes = (H/'validator.txt').read_bytes()
    validator_encoding = ('utf-16' if validator_bytes.startswith((b'\xff\xfe', b'\xfe\xff'))
                          else 'utf-8')
    assert 'OK:' in validator_bytes.decode(validator_encoding)
    prior = json.loads((H.parent/'phase4/release.json').read_text())
    extra_rows = reports['stage2_long']['detail']['s177']['clips']
    entry_seconds = sum(row['seconds'] for row in extra_rows.values())
    assert len(extra_rows) == 3 and entry_seconds > 0
    s175_seconds = prior['timing']['conditional_server_seconds']
    estimate = s175_seconds + 137/len(extra_rows)*entry_seconds + 0.975
    minutes, seconds = divmod(round(estimate), 60)
    result = dict(path=str(ZIP), sha256=archive['sha256'], bytes=archive['bytes'],
                  members=archive['members'], crc_and_member_hashes_passed=True,
                  build_sha256=archive['build_sha256'], release_ready=True,
                  carrier_official_score=0.66245, official_s177_result=None,
                  uploaded=False, qa_reports={name: dict(seconds=report['detail']['seconds'],
                       changed_entry_rows=report['detail']['changed_entry_rows'],
                       peak_rss=report['resources']['peak_aggregate_rss_bytes'],
                       peak_cuda=report['cuda_peak_reserved']) for name, report in reports.items()},
                  resources=dict(rss_cap_gib=4, peak_aggregate_rss_bytes=totals['peak_rss'],
                                 peak_cuda_reserved_bytes=totals['peak_cuda'],
                                 max_processes=2,
                                 minimum_host_free_commit_gib=totals['min_commit']),
                  runtime_estimate=dict(seconds=estimate, formatted=f'{minutes}m{seconds:02d}s',
                      s175_conditional_seconds=s175_seconds,
                      s177_entry_pass_long_panel_seconds=entry_seconds,
                      long_panel_clips=3, assumed_server_clips=137,
                      caveat='Conditional S175 runtime plus isolated S177 bucket-pass time on three long clips scaled to 137 and 0.975s tracker load. Shared GPU load, hidden stage mix and server hardware differ.'),
                  qa_source_sha256=sha(H/'qa.py'), launcher_sha256=sha(H/'launch.ps1'))
    (H/'release.json').write_text(json.dumps(result, indent=2)+'\n')
    (W/'archive.json').write_text(json.dumps({**archive, 'release_ready': True}, indent=2)+'\n')
    readme = f'''# S177 S175 + S176 bucket stack

Release: `$DATA_DIR/releases/S177_bkt.zip`. SHA256 `{archive['sha256']}`.
Bytes {archive['bytes']}; 121 members. S172 carrier's official score was 0.66245;
S177 has no official result or upload.

The builder SHA-pins exact S175 and S176 ZIPs. Every S175 member except the
appended `inference.py` dispatcher is byte-identical. S176's modified S161
module is included byte-for-byte as a separate entry module, leaving S175's
original S161 and S174 path intact. The S176 notice is included.

Final order: complete exact S175 prediction, including S172 visual collision
and S174 entry on that collision; run S176 on the same final S172 collision
anchor. Only when S174 abstains and S161 reports `inside_from_start`, choose
the first clip frame. Only when S161 reports `inside_when_first_seen`, choose
the first inside observation minus 0.2s. Then clamp entry to collision.
Accepted S161/S174 crossings and all other abstention reasons retain S175.
Load, per-clip or close errors retain exact S175 entry.

CUDA QA exported public, long, cascade, entry and extra S176 bucket panels.
All non-entry columns matched S175; independent S176-from-ZIP replay on S172
collision anchors matched every entry. The bucket panel exercised
{reports['stage2_bucket']['detail']['changed_entry_rows']} natural entry changes;
{totals['changed']} entry differences occurred across panels (not independent clips).
Injected failures returned exact S175 rows and same-process recovery returned
S177. Stage1/3 model members are byte-identical to S175, and their inference
functions are unchanged before the appended Stage2 dispatcher; S175 QA applies.
Network audit, at most two processes, <=2 GiB CUDA lease, <=4 GiB aggregate
RSS, >=12 GiB host free commit, streamed ZIP member CRC/SHA256 and the static
submission validator passed. `qa.lock`, GPU lease and `gpu.request` per-clip
yielding were used.

The isolated S177 entry pass took {entry_seconds:.2f}s for three long clips.
Conditional server estimate: {minutes}m{seconds:02d}s, from S175's conditional
40m23s plus that pass scaled to 137 clips and 0.975s tracker load. This is not
an official runtime measurement; shared GPU load, hidden stage mix and server
hardware limit precision. Peak RSS {totals['peak_rss']/2**30:.3f} GiB, CUDA
reserved {totals['peak_cuda']/2**30:.3f} GiB, minimum free commit
{totals['min_commit']:.2f} GiB. First-frame transfer remains the main score risk.
No private evaluation data, upload, commit or push.
'''
    (H/'README.md').write_text(readme)
    report = f'''RESULT done

CHANGED: S177_bkt.zip = exact S175 plus pinned S176 bucket-only pass.
SHA256: {archive['sha256']}

VERIFIED: Exact S175 protected outputs; independent S176-on-S172 entry replay,
fault fallback to exact S175 and same-process recovery; CRC/member SHA256,
validator and resource limits pass.

RISKS: Conditional server runtime {minutes}m{seconds:02d}s; no S177 official
result. First-frame transfer is unproven.

NEXT: Stop. No upload, commit or push. See README.md/release.json.
'''
    (H/'WORKER_REPORT.md').write_text(report)
    print(json.dumps(dict(sha256=archive['sha256'], path=str(ZIP),
                          runtime_seconds=estimate, release_ready=True,
                          changed_entries_across_panels=totals['changed']), indent=2))


if __name__ == '__main__':
    main()
