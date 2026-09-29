"""Package S181 only after exact-base exported CUDA QA and policy checks."""
import datetime,hashlib,json,pathlib,shutil,subprocess,sys,zipfile

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');ROOT=D/'s181_fallback/candidate'
BASE='d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b'
PRED='model/stage2/s161/predict.py'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

build=json.loads((H/'build.json').read_text());qa=json.loads((H/'qa.json').read_text())
analysis=json.loads((H/'analysis.json').read_text())
assert sha(D/'releases/S180_resid.zip')==build['base_sha256']==BASE
assert qa['passed'] and qa['only_entry_changed'] and qa['entry_clamped'] and qa['source_frame_ids']
assert qa['per_clip_failure_retains_s180'] and qa['recovery_same_process']
assert qa['dependency_install_seconds']<600 and not qa['network_attempts']
assert qa['resources']['peak_rss']<=4*2**30 and qa['cuda_peak_reserved']<=2*2**30
assert qa['resources']['min_free_commit_gib']>=12
assert qa['build_sha256']==sha(H/'build.json')
assert build['analysis_sha256']==sha(H/'analysis.json')
assert analysis['gated_prior_pooled']['gain']==14
assert all(analysis['gated_prior_panels'][p]['gain']>0 for p in analysis['gated_prior_panels'])
assert all(analysis['folds'][p]['learned_gated_prior']['gain']>0 for p in analysis['folds'])
assert qa['changed']>=2
actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
assert actual==build['members'],'Staged tree changed after QA'
with zipfile.ZipFile(D/'releases/S180_resid.zip') as z:
    names=z.namelist()
    for name in names:
        if name==PRED:
            assert (ROOT/name).read_bytes()==z.read(name)+b'\n\n'+(H/'enhance_predict.py').read_bytes()
        else:
            with z.open(name) as source:
                assert hashlib.file_digest(source,'sha256').hexdigest()==actual[name]
check=subprocess.run(['git','apply','--reverse','--check',
                      str(H/'S181_on_S180.patch')],cwd=ROOT,capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
assert sha(H/'S181_on_S180.patch')==build['patch_sha256']
uncompressed=sum((ROOT/name).stat().st_size for name in actual)
assert uncompressed<=32_000_000_000
dest=D/'releases/S181_fallback.zip'
assert len(dest.name)<=30 and dest.name.isascii() and not dest.exists()
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in sorted(actual):
        info=zipfile.ZipInfo(name,date_time=(2026,9,28,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1
        info.external_attr=0o100644<<16
        with z.open(info,'w',force_zip64=True) as out,(ROOT/name).open('rb') as source:
            shutil.copyfileobj(source,out,1024*1024)
with zipfile.ZipFile(dest) as z:
    assert set(z.namelist())==set(actual)
    for name in actual:
        with z.open(name) as source:
            assert hashlib.file_digest(source,'sha256').hexdigest()==actual[name],name
assert dest.stat().st_size<=10_000_000_000
validator=subprocess.run([sys.executable,'-B',str(R/'tools/validate_submission.py'),str(dest)],
                         capture_output=True,text=True,timeout=180)
(H/'validator.txt').write_text(validator.stdout+validator.stderr)
assert validator.returncode==0,validator.stdout+validator.stderr
receipt=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(actual),
             uncompressed_bytes=uncompressed,base_sha256=BASE,changed=build['changed'],
             added=build['added'],unchanged_base_members=build['unchanged_base_members'],
             patch_path=str(H/'S181_on_S180.patch'),patch_sha256=build['patch_sha256'],
             validator_exit=0,crc_and_member_sha256_verified=True,official_size_limits_verified=True,
             completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
             build_sha256=sha(H/'build.json'),qa_sha256=sha(H/'qa.json'),
             analysis_sha256=sha(H/'analysis.json'),release_ready=True)
(H/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
