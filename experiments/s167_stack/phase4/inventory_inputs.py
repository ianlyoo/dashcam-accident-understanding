"""Pin S175 input identities and extract small integration sources for review."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

H=Path(__file__).resolve().parent
D=Path('$DATA_DIR/releases')
FILES={'S171_ent.zip':'f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315',
       'S172_vis.zip':'2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549',
       'S174_cov.zip':'d2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    manifests={}
    for fn,expected in FILES.items():
        path=D/fn
        assert sha(path)==expected,fn
        with zipfile.ZipFile(path) as z:
            m={}
            for info in z.infolist():
                if info.is_dir():continue
                h=hashlib.sha256()
                with z.open(info) as f:
                    for block in iter(lambda:f.read(2**20),b''):h.update(block)
                assert info.filename not in m
                m[info.filename]=h.hexdigest()
            manifests[fn]=m
            for name in ('inference.py','model/stage2/s118/adapter.py','model/stage2/s161/predict.py','model/stage2/s172/runtime.py'):
                if name in m:
                    target=H/'source'/fn.removesuffix('.zip')/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(name))
    base=manifests['S171_ent.zip']
    result={}
    for fn in ('S172_vis.zip','S174_cov.zip'):
        m=manifests[fn]
        result[fn]=dict(changed=sorted(k for k in base if m.get(k)!=base[k]),
                        added=sorted(set(m)-set(base)),deleted=sorted(set(base)-set(m)),members=len(m))
    (H/'input_deltas.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
