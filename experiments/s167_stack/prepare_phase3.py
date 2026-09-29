"""S171 = exact S170 stack plus the pinned, mode-repaired S161 entry arm."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from candidates.s167_stack.build_stack import sha,stage,members
DATA=Path('$DATA_DIR')
WORK=DATA/'s167_stack/phase3'
S161='7bcabece62a8c82907f997a431cc1673ba36c81270b92133fdfcd0eb738f9e44'
S170='66d0be5c0efbe5256055a1eb6662b3106aa0b4b4813a61d29b35314fe02a94f4'

def main():
    previous=json.loads((DATA/'s167_stack/phase2b/build.json').read_text())
    assert sha(DATA/'releases/S170_all.zip')==S170
    assert sha(DATA/'releases/S161_entry.zip')==S161
    for arm in previous['arms']:
        assert sha(arm['path'])==arm['sha256']
    stage(SimpleNamespace(base=str(DATA/'releases/S156_casc.zip'),
        arms=[a['path'] for a in previous['arms']]+[str(DATA/'releases/S161_entry.zip')],
        out=str(WORK/'candidate'),normalize_merge_path=['model/stage2/s144/predict.py']))
    build=json.loads((WORK/'build.json').read_text())
    changed=sorted(k for k in previous['members'] if build['members'][k]!=previous['members'][k])
    added=sorted(set(build['members'])-set(previous['members']))
    assert changed==['model/stage2/s118/adapter.py','model/stage2/s118/rule.json']
    assert added==['model/licenses/S161-NOTICE.txt','model/stage2/s161/params.json','model/stage2/s161/predict.py']
    with zipfile.ZipFile(DATA/'releases/S170_all.zip') as z:
        assert members(z)==previous['members']
    with zipfile.ZipFile(DATA/'releases/S161_entry.zip') as z:
        arm=next(a for a in build['arms'] if Path(a['path']).name=='S161_entry.zip')
        for name in changed+added:
            assert build['members'][name]==arm['member_sha256'][name]
        ref=WORK/'reference_s161'
        for name in added:
            p=ref/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    root=WORK/'candidate'
    compiled=[]
    for p in root.rglob('*.py'):
        compile(p.read_bytes(),str(p),'exec');compiled.append(p.relative_to(root).as_posix())
    spec=importlib.util.spec_from_file_location('_s171_validate',root/'model/stage2/s118/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    rule=json.loads((root/'model/stage2/s118/rule.json').read_text())
    adapter.validate_rule(rule)
    assert rule['entry_frame']['side'] is False and rule['entry_frame']['fallback']=='s109'
    assert rule['collision_frame']['preserve_s109_fields'] is True
    proof=dict(build_sha256=sha(WORK/'build.json'),s170_sha256=S170,s161_sha256=S161,
        s170_uploaded_row=105253,changed=changed,added=added,unchanged_s170_members=99,
        compiled_python_members=compiled,mode_validation_passed=True,
        entry_rule='S161 observed crossing anchored to final S160 collision, then min(entry, collision); abstention/load/clip error retains S170 entry.',
        reference_s161_members={name:build['members'][name] for name in added},
        previous_build_sha256=sha(DATA/'s167_stack/phase2b/build.json'))
    (WORK/'merge.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2),flush=True)

if __name__=='__main__':main()
