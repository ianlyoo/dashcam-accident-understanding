"""Release exact-S174 S176 after paired evidence and actual exported CUDA QA."""
import datetime
import difflib
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');ROOT=D/'s176_bucket/candidate'
PRED='model/stage2/s161/predict.py'
BASE='d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

build=json.loads((H/'build.json').read_text())
qa=json.loads((H/'qa.json').read_text())
paired=json.loads((H/'paired.json').read_text())
measure=json.loads((H/'measure.json').read_text())
assert sha(D/'releases/S174_cov.zip')==build['base_sha256']==BASE
assert qa['passed']
assert paired['build_sha256']==qa['build_sha256']
qa_build=H/'build_qa.json'
if qa_build.exists():
    prior=json.loads(qa_build.read_text())
    rebind=json.loads((H/'patch_rebind.json').read_text())
    assert qa['build_sha256']==sha(qa_build)
    for key in ('base_sha256','members','changed','added','extension_sha256',
                'measure_sha256','unchanged_base_members'):
        assert prior[key]==build[key],key
    old_patch=H/'patch_crlf_qa.diff'
    assert sha(old_patch)==prior['patch_sha256']
    notice=(ROOT/'model/licenses/S176-NOTICE.txt').read_text()
    notice_diff=''.join(difflib.unified_diff([],notice.splitlines(keepends=True),
                                              fromfile='/dev/null',
                                              tofile='b/model/licenses/S176-NOTICE.txt')).encode()
    assert old_patch.read_bytes().replace(b'\r\n',b'\n')+notice_diff==(H/'S176_on_S174.patch').read_bytes()
    assert rebind['all_package_member_sha256_identical'] and rebind['reverse_git_apply_check']
    assert rebind['qa_build_sha256']==sha(qa_build) and rebind['final_build_sha256']==sha(H/'build.json')
else:
    assert qa['build_sha256']==sha(H/'build.json')
assert build['measure_sha256']==sha(H/'measure.json') and build['patch_sha256']==sha(H/'S176_on_S174.patch')
assert build['changed']==[PRED] and build['unchanged_base_members']==104
assert build['added']==['model/licenses/S176-NOTICE.txt']
assert paired['n']==243 and measure['n']==96 and paired['changed']==50
assert qa['only_entry_changed'] and qa['source_frame_ids'] and qa['entry_clamped']
assert qa['single_batch_equal'] and qa['per_clip_failure_retains_s174'] and qa['recovery_same_process']
assert qa['injected_crossing_maps_original_frame_id'] and qa['injected_bucket_maps_original_frame_id']
assert not qa['network_attempts'] and qa['dependency_install_seconds']<600
assert qa['changed']>=1,'Exported target panel did not exercise a natural S176 change'
assert qa['resources']['peak_rss']<=4*2**30
assert qa['cuda_peak_reserved']<=2*2**30
assert qa['resources']['min_free_commit_gib'] is not None and qa['resources']['min_free_commit_gib']>=12

actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
assert actual==build['members'],'Staged tree changed after QA'
with zipfile.ZipFile(D/'releases/S174_cov.zip') as z:
    source=z.read(PRED)
assert (ROOT/PRED).read_bytes()==source+b'\n\n'+(H/'enhance.py').read_bytes()
patch_check=subprocess.run(['git','apply','--reverse','--check',
                            str(H/'S176_on_S174.patch')],cwd=ROOT,
                           capture_output=True,text=True,timeout=30)
assert patch_check.returncode==0,patch_check.stdout+patch_check.stderr
uncompressed=sum((ROOT/name).stat().st_size for name in actual)
assert uncompressed<=32_000_000_000
dest=D/'releases/S176_bucket.zip'
assert len(dest.name)<=30 and dest.name.isascii() and not dest.exists()
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in sorted(actual):
        info=zipfile.ZipInfo(name,date_time=(2026,9,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        info._compresslevel=1;info.external_attr=0o100644<<16
        with z.open(info,'w',force_zip64=True) as out,(ROOT/name).open('rb') as src:
            shutil.copyfileobj(src,out,1024*1024)
with zipfile.ZipFile(dest) as z:
    for name in actual:
        with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==actual[name],name
assert dest.stat().st_size<=10_000_000_000
validator=subprocess.run([sys.executable,'-B',str(R/'tools/validate_submission.py'),str(dest)],
                         capture_output=True,text=True,timeout=180)
(H/'validator.txt').write_text(validator.stdout+validator.stderr)
assert validator.returncode==0,validator.stdout+validator.stderr
receipt=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(actual),
             uncompressed_bytes=uncompressed,base_sha256=BASE,
             changed=build['changed'],added=build['added'],unchanged_base_members=104,
             patch_path=str(H/'S176_on_S174.patch'),patch_sha256=build['patch_sha256'],
             validator_exit=0,crc_and_member_sha256_verified=True,official_size_limits_verified=True,
             completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
             build_sha256=sha(H/'build.json'),qa_build_receipt_sha256=qa['build_sha256'],
             patch_line_endings_normalized_after_qa=qa_build.exists(),
             qa_sha256=sha(H/'qa.json'),
             paired_sha256=sha(H/'paired.json'),measure_sha256=sha(H/'measure.json'),
             release_ready=True)
(H/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
