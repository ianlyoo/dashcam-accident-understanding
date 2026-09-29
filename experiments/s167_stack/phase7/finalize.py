"""Gate and describe S179 only after exact exported CUDA QA and validator."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

H = Path(__file__).resolve().parent
D = Path('$DATA_DIR')
W = D/'s167_stack/phase7'
ROOT = W/'candidate'
ZIP = D/'releases/S179_side.zip'
PANELS = ('public', 'long', 'cascade', 'entry', 'bucket')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def integrity():
    build = json.loads((W/'build.json').read_text())
    assert build['carrier_sha256'] == 'b30e56829cfbbad98bd3121b1c7ced6760fb8bb1da97f73aba006d45cb26adea'
    assert len(build['arms']) == 1
    assert [a['sha256'] for a in build['arms']] == [
        'bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047']
    assert build['postprocessor_sha256'] == sha(H.parent/'phase6/chain_entry.py')
    assert build['build_script_sha256'] == sha(H.parent/'build_stack.py')
    assert build['changed_from_carrier'] == ['inference.py']
    staged = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    assert staged == build['members']
    with zipfile.ZipFile(D/'releases/S177_bkt.zip') as archive:
        base = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    digest.update(block)
            base[info.filename] = digest.hexdigest()
        assert set(base) <= set(staged)
        assert all(staged[name] == digest for name, digest in base.items() if name != 'inference.py')
        assert (ROOT/'inference.py').read_bytes().startswith(archive.read('inference.py'))
    # Stage1 and Stage3 code paths, weights and dispatch definitions are in
    # exact S177 bytes. Only a final predict_stage2 wrapper was appended.
    return build


def acceptance():
    build = integrity()
    reports = {}
    for panel in PANELS:
        report = json.loads((W/'qa'/f'{panel}.json').read_text())
        assert report['passed'] and report['panel'] == panel
        assert report['build_sha256'] == sha(W/'build.json')
        assert report['qa_source_sha256'] == sha(H/'qa.py')
        assert report['artifact_sha256'] == build['members']['inference.py']
        assert report['candidate_csv_sha256'] == sha(W/'qa'/f'stage2_{panel}.csv')
        assert report['network_attempts'] == []
        assert report['resources']['peak_aggregate_rss_bytes'] <= 4*2**30
        assert report['resources']['peak_processes'] <= 2
        assert report['resources']['minimum_host_free_commit_gib'] >= 12
        assert report['cuda_peak_reserved'] <= 2*2**30
        assert report['rows'] == len(report['replay'])
        assert report['changed_entry_rows'] == 0
        reports[panel] = report
    assert reports['public']['fault']['fallback_exact_s177']
    assert reports['public']['fault']['recovery_exact_s179']
    assert sum(r['changed_side_rows'] for r in reports.values()) > 0
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post', action='store_true')
    args = parser.parse_args()
    reports = acceptance()
    print('S179 ACCEPTANCE PASS: exact carrier; CUDA protected parity; arm replay; fault; resources')
    if not args.post:
        return
    archive = json.loads((W/'archive.json').read_text())
    assert ZIP.exists() and archive['sha256'] == sha(ZIP)
    assert archive['crc_and_member_hashes_passed']
    assert archive['members'] == len(json.loads((W/'build.json').read_text())['members'])
    validator = (W/'validator.txt').read_text()
    assert validator.strip().startswith('OK:'), validator
    prior = json.loads((H.parent/'phase5/release.json').read_text())
    prior_seconds = prior['runtime_estimate']['seconds']
    long_chain = reports['long']['chain_seconds']
    extra = long_chain / reports['long']['rows'] * 137 + 1.0
    estimate = prior_seconds + extra
    estimate_m, estimate_s = divmod(round(estimate), 60)
    flag = estimate > 50*60
    totals = dict(peak_rss=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
                  peak_cuda=max(r['cuda_peak_reserved'] for r in reports.values()),
                  min_free_commit=min(r['resources']['minimum_host_free_commit_gib'] for r in reports.values()))
    release = dict(path=str(ZIP), sha256=archive['sha256'], bytes=archive['bytes'],
                   members=archive['members'], base_sha256=prior['sha256'],
                   build_sha256=archive['build_sha256'], release_ready=True,
                   qa_panels={p: dict(rows=r['rows'], seconds=r['seconds'],
                                      changed_entry_rows=r['changed_entry_rows'],
                                      changed_side_rows=r['changed_side_rows'], rules=r['rules'])
                              for p, r in reports.items()},
                   resources=totals,
                   runtime_estimate=dict(seconds=estimate, display=f'{estimate_m}m{estimate_s:02d}s',
                                         flag_over_50m=flag,
                                         method='S177 conditional estimate plus S179 isolated tracker time on three long clips scaled to 137, plus 1s load; S172 official 34m18s vs ~41m local estimate; server hardware and stage mix differ.'),
                   validator_passed=True)
    (H/'release.json').write_text(json.dumps(release, indent=2)+'\n')
    (W/'archive.json').write_text(json.dumps({**archive, 'release_ready': True}, indent=2)+'\n')
    readme = f'''# S179 side-only stack on S177

Release: `$DATA_DIR/releases/S179_side.zip`. SHA256 `{archive['sha256']}`.
No upload, commit or push.

Exact S177 carries S172 visual collision, S161/S174 entry and S176 bucket entry.
The final pass replays the SHA-pinned S178 crossing/bucket side rule on
**S172's final collision frame**. Entry and all other fields remain exact
S177. A per-clip error retains that clip's exact
S177 row; a load/close error retains the whole exact S177 output. Original
S178 adapter source is included as provenance; its side-consumption semantics
are applied by the final postprocessor to avoid changing S177's earlier pass.

The S178 ZIP/source and exact S177 carrier were hash-checked.
Five real WSL CUDA panels (public, long, cascade, entry, bucket) passed Stage2
collision/evasion/entry parity and independent S178 side replay on the
same S172 anchors; a per-clip injected fault and same-process recovery passed.
Stage1/3 code, model files and dispatcher prefix are byte-identical to S177.
Validator and ZIP member SHA256/CRC passed. Peak aggregate RSS
{totals['peak_rss']/2**30:.3f} GiB, peak CUDA reservation
{totals['peak_cuda']/2**30:.3f} GiB, minimum host free commit
{totals['min_free_commit']:.2f} GiB.

Conditional server runtime estimate: **{estimate_m}m{estimate_s:02d}s** = S177
estimate {prior_seconds:.1f}s plus {extra:.1f}s isolated S179 tracker time.
{'FLAG: this exceeds the 50-minute projection threshold. ' if flag else ''}The
official S172 run was 34m18s versus approximately 41m local projection,
so the estimate is conservative and hardware/stage mix may differ. Official
whole-inference limit remains 60 minutes. No S179 official score is claimed.
'''
    (H/'README.md').write_text(readme)
    print(json.dumps(dict(sha256=archive['sha256'], runtime=release['runtime_estimate'],
                          resources=totals)))


if __name__ == '__main__':
    main()
