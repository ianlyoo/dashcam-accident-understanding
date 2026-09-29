"""Apply S180's one-source entry shift to the exact S178 release."""
import difflib,hashlib,json,pathlib,shutil,subprocess,zipfile

H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR');W=D/'s180_resid';ROOT=W/'candidate'
BASE='bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047'
PRED='model/stage2/s161/predict.py';NOTICE='model/licenses/S180-NOTICE.txt'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

assert sha(D/'releases/S178_side.zip')==BASE
assert not ROOT.exists(),'Candidate root already exists; preserve it for review'
ROOT.mkdir(parents=True)
with zipfile.ZipFile(D/'releases/S178_side.zip') as z:
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
notice=('S180 residual entry adjustment on exact S178. Only accepted S174-only\n'
        'geometric crossings receive +0.2 seconds, clamped to collision.\n'
        'The shift models a tracked vehicle box entering the ego corridor\n'
        'before the wheel contact. Every other S178 output remains unchanged.\n'
        'No new model weights or inference dependencies; original notices remain.\n')
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
(H/'S180_on_S178.patch').write_bytes(patch.encode())
check=subprocess.run(['git','apply','--reverse','--check',
                      str(H/'S180_on_S178.patch')],cwd=ROOT,capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
receipt=dict(base_sha256=BASE,source_sha256=before[PRED],members=actual,
             changed=changed,added=added,extension_sha256=hashlib.sha256(extension).hexdigest(),
             analysis_sha256=sha(H/'analysis.json'),patch_sha256=sha(H/'S180_on_S178.patch'),
             unchanged_base_members=len(before)-1,reverse_git_apply_check=True)
(H/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))
