"""Refresh the staged S182 runtime after a pre-QA source correction."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = Path('$DATA_DIR/s167_stack/phase6')
TARGET = WORK/'candidate/model/stage2/s182_chain.py'
SOURCE = HERE/'chain_entry.py'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


report = json.loads((WORK/'build.json').read_text())
assert sha(TARGET) == report['members']['model/stage2/s182_chain.py']
assert sha(TARGET) != sha(SOURCE)
shutil.copyfile(SOURCE, TARGET)
report['members']['model/stage2/s182_chain.py'] = sha(TARGET)
report['postprocessor_sha256'] = sha(SOURCE)
(WORK/'build.json').write_text(json.dumps(report, indent=2)+'\n')
print('REFRESHED', sha(TARGET))
