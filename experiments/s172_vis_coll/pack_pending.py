"""Prepare and validate a pending archive; never promotes it to releases."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile
from build import sha
from revision_path import revision_work

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
WORK=revision_work(Path('$DATA_DIR/s172_vis_coll'))
def prepare_archive():
    root=WORK/'candidate'; pending=WORK/'S172_vis.zip'
    build=json.loads((WORK/'build.json').read_text())
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual==build['members'],'Staged tree changed'
    assert not pending.exists(),'Preserve previous pending archive'
    with zipfile.ZipFile(pending,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name in sorted(actual):
            info=zipfile.ZipInfo(name,date_time=(2026,9,28,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info._compresslevel=1; info.external_attr=0o100644<<16
            with archive.open(info,'w',force_zip64=True) as out,(root/name).open('rb') as source:
                shutil.copyfileobj(source,out,2**20)
    with zipfile.ZipFile(pending) as archive:
        assert set(archive.namelist())==set(actual)
        for name in actual:
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(2**20),b''): h.update(chunk)
            assert h.hexdigest()==actual[name],name
    spec=importlib.util.spec_from_file_location('_s172_validator',REPO/'tools/validate_submission.py')
    validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    errors=validator.validate(pending); assert not errors,errors
    result=dict(path=str(pending),sha256=sha(pending),bytes=pending.stat().st_size,members=len(actual),
        build_sha256=sha(WORK/'build.json'),crc_and_member_hashes_passed=True,validator_errors=errors,
        release_ready=False,reason='Awaiting actual CUDA QA and runtime gate')
    (WORK/'pending_archive.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result
if __name__=='__main__': prepare_archive()
