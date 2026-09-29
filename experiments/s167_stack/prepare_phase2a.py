"""Pin S162 and rerun the existing multi-arm builder without altering phase 1."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from candidates.s167_stack.build_stack import sha, stage

DATA = Path('$DATA_DIR')
WORK = DATA / 's167_stack/phase2a'
S162_SHA = '31bff1cfae4ff312bfe4b371906a2055695bcc29b8de48c4e9ca3ead1408f33a'

def main():
    assert sha(DATA / 'releases/S162_acc.zip') == S162_SHA
    args = SimpleNamespace(base=str(DATA / 'releases/S156_casc.zip'),
        arms=[str(DATA / 'releases' / p) for p in ('S163_s1.zip','S162_acc.zip','S164_fast.zip')],
        out=str(WORK / 'candidate'))
    stage(args)
    original = json.loads((DATA / 's167_stack/phase1/build.json').read_text())
    current = json.loads((WORK / 'build.json').read_text())
    assert original['members'].keys() == current['members'].keys()
    changed = [p for p in original['members'] if original['members'][p] != current['members'][p]]
    assert changed == ['model/stage3/s141/predict.py'], changed
    (WORK / 'phase1_delta.json').write_text(json.dumps(dict(changed=changed,
        phase1_build_sha256=sha(DATA / 's167_stack/phase1/build.json'),
        phase2a_build_sha256=sha(WORK / 'build.json'), s162_sha256=S162_SHA), indent=2))
    print('S169 differs from S167 only in', changed, flush=True)

if __name__ == '__main__':
    main()
