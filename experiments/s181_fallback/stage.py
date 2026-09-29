"""Apply the S181 fallback rule to exact S180 and produce a clean patch."""
import difflib,hashlib,json,pathlib,shutil,subprocess,zipfile

H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR');W=D/'s181_fallback';ROOT=W/'candidate'
BASE='d5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b'
PRED='model/stage2/s161/predict.py';NOTICE='model/licenses/S181-NOTICE.txt'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

assert sha(D/'releases/S180_resid.zip')==BASE
assert not ROOT.exists(),'Candidate root already exists; preserve it for review'
ROOT.mkdir(parents=True)
with zipfile.ZipFile(D/'releases/S180_resid.zip') as z:
    names=z.namelist();before={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
    source=z.read(PRED)
    for name in names:
        target=ROOT/name
        assert target.resolve().is_relative_to(ROOT.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        with z.open(name) as input_file,target.open('wb') as output:
            shutil.copyfileobj(input_file,output,1024*1024)
extension=(H/'enhance_predict.py').read_bytes()
(ROOT/PRED).write_bytes(source+b'\n\n'+extension)
notice=('S181 collision-relative entry fallback on exact S180. Only S161\n'
        'short_track, not_in_lane_at_collision, and far_inside_when_first_seen\n'
        'after S174/S176 abstain use collision minus 0.85 seconds. The gap is\n'
        'the median of 52 public accepted S161 crossings, not a fitted hidden\n'
        'label. It is clamped to the clip start and collision. Other outputs\n'
        'remain S180. No new model weights or inference dependencies.\n')
(ROOT/NOTICE).write_text(notice,encoding='utf-8',newline='\n')
actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
changed=sorted(n for n in names if actual[n]!=before[n]);added=sorted(set(actual)-set(before))
assert changed==[PRED] and added==[NOTICE],(changed,added)
for path in ROOT.rglob('*.py'):compile(path.read_bytes(),str(path),'exec')
patch=''.join(difflib.unified_diff(source.decode().splitlines(keepends=True),
                                   (ROOT/PRED).read_bytes().decode().splitlines(keepends=True),
                                   fromfile='a/'+PRED,tofile='b/'+PRED,n=0))
patch+=''.join(difflib.unified_diff([],notice.splitlines(keepends=True),
                                   fromfile='/dev/null',tofile='b/'+NOTICE,n=0))
(H/'S181_on_S180.patch').write_bytes(patch.encode())
check=subprocess.run(['git','apply','--reverse','--check',
                      str(H/'S181_on_S180.patch')],cwd=ROOT,capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
receipt=dict(base_sha256=BASE,source_sha256=before[PRED],members=actual,
             changed=changed,added=added,extension_sha256=hashlib.sha256(extension).hexdigest(),
             analysis_sha256=sha(H/'analysis.json'),patch_sha256=sha(H/'S181_on_S180.patch'),
             unchanged_base_members=len(before)-1,reverse_git_apply_check=True)
(H/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))
