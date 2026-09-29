"""Package only after actual export QA, full CSV parity and long integration pass."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from candidates.s118_s2rule import build

WORK=build.DATA/'s161_entry_cross'


def main():
    qa=build.read(WORK/'qa_v3/qa.json')
    integration=build.read(WORK/'integration.json')
    assert qa['passed'] and integration['passed']
    assert qa['checks']['actual_linux_cuda'] and not qa['mock_models']
    assert all(qa['s156_output_parity'].values())
    parity={}
    for stage in ['stage1','stage2','stage3']:
        paths=[WORK/'qa_v3/ipc'/('%s_%s.csv'%(name,stage)) for name in ['s156','s161']]
        rows=[list(csv.DictReader(p.open())) for p in paths]
        assert len(rows[0])==len(rows[1])
        changed=set()
        for a,b in zip(*rows):
            assert set(a)==set(b)
            changed.update(k for k in a if a[k]!=b[k])
        assert changed <= ({'entry_frame'} if stage=='stage2' else set()),(stage,changed)
        parity[stage]=dict(rows=len(rows[0]),changed_columns=sorted(changed),
                           s156_csv=build.identity(paths[0]),s161_csv=build.identity(paths[1]))
    archive=build.DATA/'releases/S161_entry.zip'
    stage=build.read(WORK/'stage/stage.json')
    assert qa['candidate_manifest_sha256']==stage['manifest_sha256']
    assert build.tree(WORK/'stage/candidate')==stage['members']
    if not archive.exists():
        build.package(WORK/'stage',WORK/'qa_v3/qa.json','S161_entry.zip')
    else:
        with zipfile.ZipFile(archive) as z:
            assert sorted(z.namelist())==sorted(stage['members'])
            for name,v in stage['members'].items():
                with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==v['sha256']
    validated=subprocess.run([sys.executable,'-B',str(REPO/'tools/validate_submission.py'),str(archive)],
                             cwd=REPO,text=True,capture_output=True)
    (WORK/'logs/validator.log').write_text(validated.stdout+validated.stderr)
    assert validated.returncode==0,validated.stdout+validated.stderr
    report=dict(zip_path=str(archive),zip=build.identity(archive),validator_returncode=validated.returncode,
                parity=parity,qa=build.identity(WORK/'qa_v3/qa.json'),
                integration=build.identity(WORK/'integration.json'),
                qa_seconds=qa['seconds'],qa_host_memory=qa['host_memory'],
                qa_cuda_peak_bytes=qa['cuda_peak_allocated_bytes'],
                integration_seconds=integration['seconds'],
                long_measured_wall_delta_seconds=integration['measured_wall_delta_seconds'],
                long_tracker_seconds=integration['s161_diagnostics']['track_seconds'],
                long_tracker_load_seconds=integration['s161_diagnostics']['load_seconds'],
                long_changed_rows=integration['changed_rows'],
                stage_diff=build.read(WORK/'stage/stage.json')['s161'])
    (HERE/'release.json').write_text(json.dumps(report,indent=2)+'\n')
    readme=(HERE/'README.md').read_text()
    readme=readme.replace('Status: source and complete public-label evaluation ready; corrected WSL CUDA QA queued.',
                          'Status: **release ready for manual review**. WSL CUDA QA, export parity and ZIP validation passed.')
    readme=readme.replace('Full WSL QA,\nthree long-clip exported parity checks, validator and release hash are pending.',
                          'Full WSL CUDA QA, three long-clip exported parity checks and the ZIP validator passed.\nSee `release.json` for hashes, runtime and CSV identities.')
    readme=readme.replace('CSV comparisons and runtime evidence will be under',
                          'CSV comparisons and runtime evidence are under')
    if '## Release identity' not in readme:
        readme+='\n## Release identity\n\n'
        readme+='ZIP: `$DATA_DIR/releases/S161_entry.zip`\n\n'
        readme+='SHA256: `'+report['zip']['sha256']+'`\n\n'
        readme+='Size: '+str(report['zip']['bytes'])+' bytes. Validator exit 0.\n\n'
        readme+='Exact non-entry CSV parity passed on the public three-stage QA set and\n'
        readme+='three native-rate Nexar clips; '+str(report['long_changed_rows'])+' long-clip entries changed.\n'
        readme+='Long-clip tracker time: %.3f s / 3 clips; detector load: %.3f s.\n'%(report['long_tracker_seconds'],report['long_tracker_load_seconds'])
        readme+='Matched whole-call wall delta: %.3f s (includes uncontrolled cache/load effects).\n'%report['long_measured_wall_delta_seconds']
        readme+='The integration script packages only after every prerequisite passes; finish.py\n'
        readme+='is safe to rerun and verifies any existing ZIP instead of overwriting it.\n'
    (HERE/'README.md').write_text(readme)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
