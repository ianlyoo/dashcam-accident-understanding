"""Release S178 only after exported side-only QA and full human-panel scoring."""
import datetime,hashlib,json,pathlib,shutil,subprocess,sys,zipfile

H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');ROOT=D/'s178_side/candidate'
BASE='fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68'
PRED='model/stage2/s161/predict.py';ADAPTER='model/stage2/s118/adapter.py'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
build=json.loads((H/'build.json').read_text());qa=json.loads((H/'qa.json').read_text())
human=json.loads((H/'human.json').read_text());study=json.loads((H/'study.json').read_text())
assert sha(D/'releases/S176_bucket.zip')==build['base_sha256']==BASE
assert qa['passed'] and qa['only_side_changed'] and qa['source_frame_ids'] and qa['entry_clamped']
assert qa['per_clip_failure_retains_s176'] and qa['recovery_same_process']
assert qa['dependency_install_seconds']<600 and not qa['network_attempts']
assert qa['resources']['peak_rss']<=4*2**30 and qa['cuda_peak_reserved']<=2*2**30
assert qa['resources']['min_free_commit_gib']>=12
assert qa['build_sha256']==sha(H/'build.json')
assert build['study_sha256']==sha(H/'study.json')
assert (study['side']['ccd_consensus_side']['s161_center_0.15']['candidate_f1'] >
        study['side']['ccd_consensus_side']['incumbent']['incumbent_f1'])
assert (study['side']['nexar_confident_side']['s161_center_0.15']['candidate_f1'] >
        study['side']['nexar_confident_side']['incumbent']['incumbent_f1'])
assert human['dkb']['n']==74 and human['jungmin']['n']==36
assert qa['changed']>=1
actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
assert actual==build['members'],'Staged tree changed after QA'
with zipfile.ZipFile(D/'releases/S176_bucket.zip') as z:
    names=z.namelist()
    assert len(names)==106
    for name in names:
        if name in (PRED,ADAPTER):
            extension=(H/('enhance_predict.py' if name==PRED else 'enhance_adapter.py')).read_bytes()
            assert (ROOT/name).read_bytes()==z.read(name)+b'\n\n'+extension
        else:
            with z.open(name) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==actual[name]
check=subprocess.run(['git','apply','--reverse','--check',
                      str(H/'S178_on_S176.patch')],cwd=ROOT,capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
assert sha(H/'S178_on_S176.patch')==build['patch_sha256']
uncompressed=sum((ROOT/name).stat().st_size for name in actual)
assert uncompressed<=32_000_000_000
dest=D/'releases/S178_side.zip'
assert len(dest.name)<=30 and dest.name.isascii() and not dest.exists()
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in sorted(actual):
        info=zipfile.ZipInfo(name,date_time=(2026,9,28,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info._compresslevel=1
        info.external_attr=0o100644<<16
        with z.open(info,'w',force_zip64=True) as out,(ROOT/name).open('rb') as src:
            shutil.copyfileobj(src,out,1024*1024)
with zipfile.ZipFile(dest) as z:
    assert set(z.namelist())==set(actual)
    for name in actual:
        with z.open(name) as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==actual[name],name
assert dest.stat().st_size<=10_000_000_000
validator=subprocess.run([sys.executable,'-B',str(R/'tools/validate_submission.py'),str(dest)],
                         capture_output=True,text=True,timeout=180)
(H/'validator.txt').write_text(validator.stdout+validator.stderr)
assert validator.returncode==0,validator.stdout+validator.stderr
receipt=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(actual),
             uncompressed_bytes=uncompressed,base_sha256=BASE,
             changed=build['changed'],added=build['added'],
             unchanged_base_members=build['unchanged_base_members'],
             patch_path=str(H/'S178_on_S176.patch'),patch_sha256=build['patch_sha256'],
             validator_exit=0,crc_and_member_sha256_verified=True,official_size_limits_verified=True,
             completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
             build_sha256=sha(H/'build.json'),qa_sha256=sha(H/'qa.json'),human_sha256=sha(H/'human.json'),
             study_sha256=sha(H/'study.json'),release_ready=True)
(H/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
