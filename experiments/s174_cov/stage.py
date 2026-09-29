"""Stage exact S171 with only an append-only S161 fallback extension."""
import hashlib,json,pathlib,shutil,zipfile
H=pathlib.Path(__file__).resolve().parent
D=pathlib.Path('$DATA_DIR');W=D/'s174_cov';R=W/'candidate'
BASE='f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(D/'releases/S171_ent.zip')==BASE
if not R.exists():
 with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:
  for name in z.namelist():
   p=R/name;assert p.resolve().is_relative_to(R.resolve())
   p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(name) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:
 source=z.read('model/stage2/s161/predict.py')
 before={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
enhancement=(H/'enhance.py').read_bytes()
assert hashlib.sha256(source).hexdigest()=='b20b957a24a991534aaf678c18465087a86b22a38b078c44e16631d2ad87d2b3'
wrapped=b'\n\n# S174 optional extension: initialization failure keeps exact S171 S161.\ntry:\n'
wrapped+=b'\n'.join(b'    '+line for line in enhancement.splitlines())
wrapped+=b'\nexcept Exception:\n    pass\n'
predict=R/'model/stage2/s161/predict.py';predict.write_bytes(source+wrapped)
notice='''S174 coverage extension to exact S171 S161, no new pretrained weights.
The original S161 accepted crossing is returned unchanged. On its abstentions,
S174 tests up to three COCO vehicles close to the final S160 collision,
alternative fixed camera corridor widths/horizons, and native-frame crossing
refinement. Original S161 and COCO detector notices/licenses are retained.
No entry-label fitting or private evaluation input. Only entry_frame can change;
the inherited S171 adapter clamps it to collision_frame.
'''
(R/'model/licenses/S174-NOTICE.txt').write_text(notice,encoding='utf-8')
actual={p.relative_to(R).as_posix():sha(p) for p in R.rglob('*') if p.is_file()}
changed=[n for n in before if actual[n]!=before[n]]
assert changed==['model/stage2/s161/predict.py'],changed
assert sorted(set(actual)-set(before))==['model/licenses/S174-NOTICE.txt']
for p in R.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
receipt=dict(base_sha256=BASE,members=actual,changed=changed,
 added=['model/licenses/S174-NOTICE.txt'],unchanged_base_members=len(before)-len(changed),
 enhancement_sha256=hashlib.sha256(enhancement).hexdigest(),release_ready=False)
(H/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))
