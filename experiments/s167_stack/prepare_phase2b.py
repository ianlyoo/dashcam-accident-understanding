"""Pinned S170 merge; no CUDA imports. Preserve the standalone S160 reference."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidates.s167_stack.build_stack import sha, stage, members
from candidates.s164_runtime.build import batch_patch

DATA = Path('$DATA_DIR')
WORK = DATA / 's167_stack/phase2b'
HOOK = 'model/stage2/s144/predict.py'
PINS = {'S162_acc.zip': '31bff1cfae4ff312bfe4b371906a2055695bcc29b8de48c4e9ca3ead1408f33a',
        'S160_loc.zip': '042e49a6dd9f1091a158becbb8107caf8ca1bb74e06d620df81f555cd262db6b'}

def main():
    for name, digest in PINS.items():
        assert sha(DATA / 'releases' / name) == digest, name
    args = SimpleNamespace(base=str(DATA / 'releases/S156_casc.zip'),
        arms=[str(DATA / 'releases' / p) for p in ('S163_s1.zip','S162_acc.zip','S160_loc.zip','S164_fast.zip')],
        out=str(WORK / 'candidate'), normalize_merge_path=[HOOK])
    stage(args)
    current = json.loads((WORK / 'build.json').read_text())
    previous = json.loads((DATA / 's167_stack/phase2a/build.json').read_text())
    changed = sorted(p for p in previous['members'] if previous['members'][p] != current['members'][p])
    added = sorted(set(current['members']) - set(previous['members']))
    assert changed == [HOOK]
    assert added == ['model/stage2/s144/' + p for p in ('S160-NOTICE.txt','loc_nn.npz','localizer.json','s160_loc.py')]
    with zipfile.ZipFile(DATA / 'releases/S160_loc.zip') as archive:
        expected = batch_patch(archive.read(HOOK).decode().replace('\r\n','\n')).encode()
        assert (WORK / 'candidate' / HOOK).read_bytes() == expected
        reference = WORK / 'reference_s160'
        assert not reference.exists()
        archive.extractall(reference)  # member paths already validated by stage()
        reference_members = members(archive)
    assert {p.relative_to(reference).as_posix():sha(p) for p in reference.rglob('*') if p.is_file()} == reference_members
    compiled = []
    for p in (WORK / 'candidate').rglob('*.py'):
        compile(p.read_bytes(), str(p), 'exec')
        compiled.append(p.relative_to(WORK / 'candidate').as_posix())
    proof = dict(changed_from_s169=changed, added_to_s169=added, unchanged_from_s169=96,
        exact_batch_patch_of_s160=True, compiled_python_members=compiled,
        build_sha256=sha(WORK / 'build.json'), previous_build_sha256=sha(DATA / 's167_stack/phase2a/build.json'),
        reference_s160_members=reference_members, reference_zip_sha256=PINS['S160_loc.zip'],
        batch_patch_source_sha256=sha(ROOT / 'candidates/s164_runtime/build.py'))
    (WORK / 'cpu_merge.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k != 'reference_s160_members'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
