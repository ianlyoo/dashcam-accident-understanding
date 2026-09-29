"""Bind exact-base build, measured activation, paired evidence and exported QA."""
import datetime,hashlib,json,pathlib,shutil,subprocess,sys,zipfile
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1]
D=pathlib.Path('$DATA_DIR');W=D/'s174_cov';ROOT=W/'candidate'
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
build=json.loads((H/'build.json').read_text());qa=json.loads((H/'qa.json').read_text())
metrics=json.loads((H/'metrics.json').read_text());labeled=json.loads((H/'audit_labeled_summary.json').read_text())
nexar=json.loads((H/'audit_750_summary.json').read_text());anchors=json.loads((H/'anchors_receipt.json').read_text())
assert build['base_sha256']=='f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315'
assert sha(D/'releases/S171_ent.zip')==build['base_sha256']
assert qa['passed'] and qa['build_sha256']==sha(H/'build.json')==metrics['build_sha256']
assert build['changed']==['model/stage2/s161/predict.py'] and build['unchanged_base_members']==103
assert metrics['n']==243 and labeled['n']==243 and nexar['n']==750 and anchors['n']==750
assert anchors['in_sample_hits']==659 and labeled['s161_source_sha256']==nexar['s161_source_sha256']
assert metrics['sources']['ccd_consensus']['n']==79 and metrics['sources']['ccd_labels_A3_low']['n']==45
assert metrics['sources']['nexar_blind_confident']['n']==10 and metrics['sources']['s117_interval']['n']==3
assert qa['only_entry_changed'] and qa['source_frame_ids'] and qa['entry_clamped']
assert qa['single_batch_equal'] and qa['per_clip_failure_retains_s171'] and qa['recovery_same_process']
assert qa['injected_crossing_maps_original_frame_id'] and not qa['network_attempts']
assert qa['dependency_install_seconds']<600
for report in (qa,metrics,nexar):
 assert report['resources']['peak_rss']<=4*2**30,report['resources']
 assert report['cuda_peak_reserved']<=2*2**30
 assert report['resources']['min_free_commit_gib'] is not None and report['resources']['min_free_commit_gib']>=12
assert metrics['long_n']==26 and metrics['long_137_seconds']<=360,metrics['long_137_seconds']
actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
assert actual==build['members'],'Staged tree changed after QA'
uncompressed=sum((ROOT/name).stat().st_size for name in actual)
assert uncompressed<=32_000_000_000
dest=D/'releases/S174_cov.zip';assert len(dest.name)<=30 and dest.name.isascii() and not dest.exists()
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name in sorted(actual):
  info=zipfile.ZipInfo(name,date_time=(2026,9,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
  info._compresslevel=1;info.external_attr=0o100644<<16
  with z.open(info,'w',force_zip64=True) as out,(ROOT/name).open('rb') as src:shutil.copyfileobj(src,out,1024*1024)
with zipfile.ZipFile(dest) as z:
 for name in actual:
  with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==actual[name],name
assert dest.stat().st_size<=10_000_000_000
validator=subprocess.run([sys.executable,'-B',str(R/'tools/validate_submission.py'),str(dest)],capture_output=True,text=True,timeout=180)
(H/'validator.txt').write_text(validator.stdout+validator.stderr)
assert validator.returncode==0,validator.stdout+validator.stderr
receipt=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,members=len(actual),
 uncompressed_bytes=uncompressed,base_sha256=build['base_sha256'],unchanged_base_members=103,
 changed=build['changed'],added=build['added'],validator_exit=0,crc_and_member_sha256_verified=True,
 official_size_limits_verified=True,added_seconds_137=metrics['long_137_seconds'],
 official_server_runtime_unmeasured=True,
 completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
 build_sha256=sha(H/'build.json'),qa_sha256=sha(H/'qa.json'),metrics_sha256=sha(H/'metrics.json'),
 labeled_audit_sha256=sha(H/'audit_labeled_summary.json'),nexar_audit_sha256=sha(H/'audit_750_summary.json'),
 release_ready=True)
(H/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
