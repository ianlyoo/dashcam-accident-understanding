"""Gate S185 on no-S176 identity, CUDA replay, isolation and validator."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

H = Path(__file__).resolve().parent
D = Path('$DATA_DIR')
W = D/'s167_stack/phase8'
ROOT = W/'candidate'
ZIP = D/'releases/S185_nob.zip'
PANELS = ('public', 'bucket', 'long', 'cascade', 'entry')
GAP = frozenset(('short_track', 'not_in_lane_at_collision',
                 'far_inside_when_first_seen'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def integrity():
    build = json.loads((W/'build.json').read_text())
    assert build['carrier_sha256'] == '3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be'
    assert [x['sha256'] for x in build['arms']] == [
        'bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047',
        'd5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b',
        '547323b4f5a5c34ee273c67ac2cba91694844ea179bda248aa870c4b89548b77']
    assert build['extension_sha256'] == sha(H/'s185_extension.py')
    assert build['postprocessor_sha256'] == sha(H/'s185_postprocess.py')
    assert build['build_script_sha256'] == sha(H.parent/'build_stack.py')
    assert build['changed_from_carrier'] == ['inference.py']
    staged = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    assert staged == build['members']
    assert not any('s176' in name.lower() for name in staged)
    with zipfile.ZipFile(D/'releases/S175_stk.zip') as archive:
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
        assert (ROOT/'model/stage2/s185_predict.py').read_bytes().startswith(
            archive.read('model/stage2/s161/predict.py'))
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
        assert report['s174_source_sha256'] == build['s174_source_sha256']
        assert report['network_attempts'] == []
        assert report['resources']['peak_aggregate_rss_bytes'] <= 4*2**30
        assert report['resources']['peak_processes'] <= 2
        assert report['resources']['minimum_host_free_commit_gib'] >= 12
        assert report['cuda_peak_reserved'] <= 2*2**30
        assert report['rows'] == len(report['replay'])
        reports[panel] = report
    assert reports['public']['fault']['fallback_exact_s175']
    assert reports['public']['fault']['recovery_exact_s185']
    assert sum(r['rules']['S180 S174-only shift'] for r in reports.values()) > 0
    assert sum(r['rules']['S181 gap'] for r in reports.values()) > 0
    assert sum(r['changed_side_rows'] for r in reports.values()) > 0
    isolation = json.loads((H/'isolation.json').read_text())
    assert isolation['passed'] and set(isolation['panels']) == set(PANELS)
    return reports, isolation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post', action='store_true')
    args = parser.parse_args()
    reports, isolation = acceptance()
    print('S185 ACCEPTANCE PASS: exact S175 carrier; no S176; CUDA replay/fault; S182 bucket isolation; resources')
    if not args.post:
        return
    archive = json.loads((W/'archive.json').read_text())
    assert ZIP.exists() and archive['sha256'] == sha(ZIP)
    assert archive['crc_and_member_hashes_passed']
    assert archive['members'] == len(json.loads((W/'build.json').read_text())['members'])
    validator = (W/'validator.txt').read_text()
    assert validator.strip().startswith('OK:'), validator
    prior = json.loads((H.parent/'phase4/release.json').read_text())
    prior_seconds = prior['timing']['conditional_server_seconds']
    long_chain = reports['long']['chain_seconds']
    extra = long_chain / reports['long']['rows'] * 137 + 1.0
    estimate = prior_seconds + extra
    estimate_m, estimate_s = divmod(round(estimate), 60)
    flag = estimate > 50*60
    totals = dict(peak_rss=max(r['resources']['peak_aggregate_rss_bytes'] for r in reports.values()),
                  peak_cuda=max(r['cuda_peak_reserved'] for r in reports.values()),
                  min_free_commit=min(r['resources']['minimum_host_free_commit_gib'] for r in reports.values()))
    reason_counts = {}
    for report in reports.values():
        for row in report['replay']:
            if row['source'] != 'S109':
                continue
            reason = row['s161_reason'] or 'unknown'
            item = reason_counts.setdefault(reason, dict(rows=0, s174_completed=0, gap=0))
            item['rows'] += 1
            item['s174_completed'] += bool(row['s174_completed'])
            item['gap'] += bool(row['s174_completed'] and reason in GAP)
    release = dict(path=str(ZIP), sha256=archive['sha256'], bytes=archive['bytes'],
                   members=archive['members'], base_sha256=prior['sha256'],
                   build_sha256=archive['build_sha256'], release_ready=True,
                   qa_panels={p: dict(rows=r['rows'], seconds=r['seconds'],
                                      changed_entry_rows=r['changed_entry_rows'],
                                      changed_side_rows=r['changed_side_rows'], rules=r['rules'])
                              for p, r in reports.items()},
                   s109_fallback_reasons=reason_counts,
                   s182_isolation_bucket_rows=sum(len(x['bucket_rows'])
                                                  for x in isolation['panels'].values()),
                   resources=totals,
                   runtime_estimate=dict(seconds=estimate, display=f'{estimate_m}m{estimate_s:02d}s',
                                         flag_over_50m=flag,
                                         method='S175 conditional estimate plus isolated S185 tracker time on three long clips scaled to 137, plus 1s load; server hardware/stage mix differ.'),
                   validator_passed=True)
    (H/'release.json').write_text(json.dumps(release, indent=2)+'\n')
    (W/'archive.json').write_text(json.dumps({**archive, 'release_ready': True}, indent=2)+'\n')
    reasons = '\n'.join(f'| `{reason}` | {data["rows"]} | {data["s174_completed"]} | {data["gap"]} |'
                        for reason, data in sorted(reason_counts.items()))
    readme = f'''# S185: no-S176 stack on S175

Release: `$DATA_DIR/releases/S185_nob.zip`. SHA256 `{archive['sha256']}`.
No upload, commit or push.

S185 starts from exact S175, which already applies S174 entry after S172's
final visual collision. A final tracker pass on that same collision anchor
retains **S178 side only for accepted original S161 crossings**. S178 never
changed side on S174-only crossings; those keep S175 side, and S180 shifts
their accepted entry +0.2 s, clamped to collision. S181 applies collision
minus 0.85 s only after S174 completed and abstained, for original S161
`short_track`, `not_in_lane_at_collision`, and
`far_inside_when_first_seen`. Per-clip errors retain exact S175 rows.

S176 source and both bucket entry decisions are absent. The S178 side changes
from S176 bucket traces are also absent. In particular,
`inside_from_start` and `inside_when_first_seen` retain S175/S109 entry;
S181 does not cover them. `no_anchor`, `short`, `far_crossing`,
`frame_budget`, and rejected crossings also retain S175/S109 unless S174
itself accepts a crossing. The S181 gate further requires a completed S174
fallback; if it faults, S175/S109 remains. These are the pieces cleanly
separated from the original S176-based patch chain; no S178 S174-only side
rule was dropped because S178 had already abstained there.

Observed S109-source reasons on the five QA panels (panel rows may overlap):

| Original S161 reason | Rows | S174 completed | S181 gap |
|---|---:|---:|---:|
{reasons}

Five real WSL CUDA panels passed exact S175 parity for Stage2 collision and
evasion, independent S174-ZIP tracker replay of entry and side on final S172
anchors, a per-clip fault and same-process recovery. Paired S182 QA confirms
S185 equals S175 on {release['s182_isolation_bucket_rows']} S176-bucket panel
rows and equals S182 on every other panel row, isolating S176's effect.
Stage1/3 code, model files and inference prefix are byte-identical to S175.
ZIP member SHA256/CRC and submission validator passed. Peak aggregate RSS
{totals['peak_rss']/2**30:.3f} GiB, peak CUDA reservation
{totals['peak_cuda']/2**30:.3f} GiB, minimum host free commit
{totals['min_free_commit']:.2f} GiB.

Conditional server runtime estimate: **{estimate_m}m{estimate_s:02d}s** = S175
estimate {prior_seconds:.1f}s plus {extra:.1f}s isolated S185 tracker time.
{'FLAG: this exceeds the 50-minute projection threshold. ' if flag else ''}S172
official runtime was 34m18s versus approximately 41m local projection;
hardware and hidden stage mix differ. The official whole-inference limit is
60 minutes. No S185 official score is claimed.
'''
    (H/'README.md').write_text(readme)
    print(json.dumps(dict(sha256=archive['sha256'], runtime=release['runtime_estimate'],
                          resources=totals, s109_fallback_reasons=reason_counts)))


if __name__ == '__main__':
    main()
