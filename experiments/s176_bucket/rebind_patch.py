"""Normalize patch line endings after QA; prove package bytes are identical."""
import difflib
import hashlib
import json
import pathlib
import subprocess
import sys

H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR')
PRED='model/stage2/s161/predict.py'
old_build=H/'build_qa.json';old_patch=H/'patch_crlf_qa.diff'
new_build=H/'build.json';new_patch=H/'S176_on_S174.patch'
qa=json.loads((H/'qa.json').read_text());assert qa['passed']
assert not (H/'release.json').exists()

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

with old_build.open('xb') as f:f.write(new_build.read_bytes())
with old_patch.open('xb') as f:f.write(new_patch.read_bytes())
prior=json.loads(old_build.read_text())
assert qa['build_sha256']==sha(old_build)
assert sha(old_patch)==prior['patch_sha256']
assert b'\r\n' in old_patch.read_bytes()
before={n:sha(D/'s176_bucket/candidate'/n) for n in prior['members']}
assert before==prior['members']
run=subprocess.run([sys.executable,'-B',str(H/'stage.py')],
                   capture_output=True,text=True,timeout=600)
assert run.returncode==0,run.stdout+run.stderr
current=json.loads(new_build.read_text())
assert before==current['members']==prior['members']
for key in ('base_sha256','changed','added','extension_sha256','measure_sha256','unchanged_base_members'):
    assert current[key]==prior[key],key
notice=(D/'s176_bucket/candidate/model/licenses/S176-NOTICE.txt').read_text()
notice_diff=''.join(difflib.unified_diff([],notice.splitlines(keepends=True),
                                         fromfile='/dev/null',
                                         tofile='b/model/licenses/S176-NOTICE.txt')).encode()
assert old_patch.read_bytes().replace(b'\r\n',b'\n')+notice_diff==new_patch.read_bytes()
assert sha(new_patch)==current['patch_sha256']
check=subprocess.run(['git','apply','--reverse','--check',str(new_patch)],
                     cwd=D/'s176_bucket/candidate',capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
record=dict(qa_build_sha256=sha(old_build),final_build_sha256=sha(new_build),
            old_patch_sha256=sha(old_patch),final_patch_sha256=sha(new_patch),
            patch_line_endings_normalized=True,notice_added_to_patch=True,
            all_package_member_sha256_identical=True,
            reverse_git_apply_check=True,prediction_sha256=before[PRED])
(H/'patch_rebind.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
