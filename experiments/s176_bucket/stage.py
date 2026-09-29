"""Stage only the winning bucket rules on exact S174 and export its patch."""
import difflib
import hashlib
import json
import pathlib
import shutil
import zipfile

H = pathlib.Path(__file__).resolve().parent
D = pathlib.Path('$DATA_DIR')
W = D/'s176_bucket'
R = W/'candidate'
BASE = 'd2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc'
PRED = 'model/stage2/s161/predict.py'
NOTICE = 'model/licenses/S176-NOTICE.txt'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(D/'releases/S174_cov.zip') == BASE
measure = json.loads((H/'measure.json').read_text())
assert measure['n'] == 96 and measure['buckets']['inside_from_start']['eligible'] == 44
for bucket, rule in (('inside_from_start','first_clip'),
                     ('inside_when_first_seen','minus_0p2s')):
    score = measure['buckets'][bucket]['eligible_primary'][rule]
    assert score['candidate'] > score['s174']
for bucket in ('far_inside_when_first_seen','not_in_lane_at_collision'):
    allowed = ['first_inside','minus_0p2s','minus_0p5s','extrapolate']
    assert all(measure['buckets'][bucket]['eligible_primary'][rule]['candidate'] <=
               measure['buckets'][bucket]['eligible_primary'][rule]['s174'] for rule in allowed)

R.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(D/'releases/S174_cov.zip') as z:
    names = z.namelist()
    before = {n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
    source = z.read(PRED)
    for name in names:
        target = R/name
        assert target.resolve().is_relative_to(R.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, target.open('wb') as dest:
            shutil.copyfileobj(src, dest, 1024*1024)

extension = (H/'enhance.py').read_bytes()
assert source.rstrip().endswith(b'except Exception:\n    pass')
prediction = source + b'\n\n' + extension
(R/PRED).write_bytes(prediction)
notice = '''S176 definition-faithful bucket extension on exact S174, no new weights.
Only S161 inside_from_start and inside_when_first_seen abstentions receive
respectively first-clip entry and first-observation minus 0.2 second.
S161/S174 accepted results and all other reasons remain unchanged.
If S174 fails or the S176 rule fails, retain S174's per-clip output.
Original S174 and detector notices/licenses remain bundled.
No private evaluation input. Only entry_frame can change; inherited adapter
clamps entry to collision and maps indices to original source frame IDs.
'''
(R/NOTICE).write_text(notice)

actual = {p.relative_to(R).as_posix(): sha(p) for p in R.rglob('*') if p.is_file()}
changed = [n for n in names if actual[n] != before[n]]
added = sorted(set(actual)-set(before))
assert changed == [PRED] and added == [NOTICE], (changed, added)
for path in R.rglob('*.py'):
    compile(path.read_bytes(), str(path), 'exec')
diff = ''.join(difflib.unified_diff(source.decode().splitlines(keepends=True),
                                    prediction.decode().splitlines(keepends=True),
                                    fromfile='a/'+PRED, tofile='b/'+PRED))
diff += ''.join(difflib.unified_diff([],notice.splitlines(keepends=True),
                                     fromfile='/dev/null',tofile='b/'+NOTICE))
assert diff and diff.count('@@') == 4
(H/'S176_on_S174.patch').write_bytes(diff.encode())
receipt = dict(base_sha256=BASE, members=actual, changed=changed, added=added,
               extension_sha256=hashlib.sha256(extension).hexdigest(),
               measure_sha256=sha(H/'measure.json'),
               patch_sha256=sha(H/'S176_on_S174.patch'),
               unchanged_base_members=len(before)-len(changed), release_ready=False)
(H/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))
