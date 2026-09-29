"""Stage a side-only append patch against the exact S176 release."""
import difflib,hashlib,json,pathlib,shutil,subprocess,zipfile

H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR');W=D/'s178_side';ROOT=W/'candidate'
BASE='fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68'
PRED='model/stage2/s161/predict.py'
ADAPTER='model/stage2/s118/adapter.py'
NOTICE='model/licenses/S178-NOTICE.txt'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(D/'releases/S176_bucket.zip')==BASE
ROOT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(D/'releases/S176_bucket.zip') as z:
    names=z.namelist();before={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
    sources={n:z.read(n) for n in (PRED,ADAPTER)}
    for name in names:
        target=ROOT/name
        assert target.resolve().is_relative_to(ROOT.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        with z.open(name) as source,target.open('wb') as output:
            shutil.copyfileobj(source,output,1024*1024)
extensions={PRED:(H/'enhance_predict.py').read_bytes(),
            ADAPTER:(H/'enhance_adapter.py').read_bytes()}
for name,extension in extensions.items():
    (ROOT/name).write_bytes(sources[name]+b'\n\n'+extension)
notice=('S178 geometric entry-side rule on exact S176. Original S161 accepted\n'
        'crossings may provide the screen-relative boundary side. In the two\n'
        'S176 already-inside buckets, a collision-anchor center left of 0.35 or\n'
        'right of 0.65 may provide side. S174-only crossings abstain from side.\n'
        'All other side predictions and every entry, collision and evasion\n'
        'prediction retain S176. Per-clip faults retain the original side.\n'
        'No new model weights or inference dependencies; original notices remain.\n')
(ROOT/NOTICE).write_text(notice,encoding='utf-8',newline='\n')
actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
changed=sorted(n for n in names if actual[n]!=before[n]);added=sorted(set(actual)-set(before))
assert changed==sorted((PRED,ADAPTER)) and added==[NOTICE],(changed,added)
for path in ROOT.rglob('*.py'):compile(path.read_bytes(),str(path),'exec')
patch=''
for name in (PRED,ADAPTER):
    patch+=''.join(difflib.unified_diff(sources[name].decode().splitlines(keepends=True),
                                       (ROOT/name).read_bytes().decode().splitlines(keepends=True),
                                       fromfile='a/'+name,tofile='b/'+name,n=0))
patch+=''.join(difflib.unified_diff([],notice.splitlines(keepends=True),
                                   fromfile='/dev/null',tofile='b/'+NOTICE,n=0))
(H/'S178_on_S176.patch').write_bytes(patch.encode())
check=subprocess.run(['git','apply','--reverse','--check',
                      str(H/'S178_on_S176.patch')],cwd=ROOT,capture_output=True,text=True,timeout=30)
assert check.returncode==0,check.stdout+check.stderr
receipt=dict(base_sha256=BASE,source_sha256={n:before[n] for n in (PRED,ADAPTER)},
             members=actual,changed=changed,added=added,
             extension_sha256={n:hashlib.sha256(v).hexdigest() for n,v in extensions.items()},
             study_sha256=sha(H/'study.json'),patch_sha256=sha(H/'S178_on_S176.patch'),
             unchanged_base_members=len(before)-len(changed),reverse_git_apply_check=True)
(H/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))
