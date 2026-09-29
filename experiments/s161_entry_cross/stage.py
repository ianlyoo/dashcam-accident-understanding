"""Build the isolated S161 tree from the exact S156 release and record every diff."""
import json
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from candidates.s118_s2rule import build

WORK=build.DATA/'s161_entry_cross'
SOURCE=build.DATA/'stage2_s118/S156_casc'
DEST=WORK/'stage'


def main():
    archive=build.DATA/'releases/S156_casc.zip'
    expected='e7fb70a66ad251d47c180861e2e001da7b270259c2e12ca394c6de3f1c510d70'
    assert build.identity(archive)['sha256']==expected
    old=build.read(SOURCE/'stage.json')
    assert build.tree(SOURCE/'candidate')==old['members']
    if not (DEST/'candidate').exists():
        shutil.copytree(SOURCE/'candidate',DEST/'candidate')
    root=DEST/'candidate'
    config=build.read(root/'model/stage2/s118/rule.json')
    config.update(name='S161_entry',note='Exact S156 collision; observed participant crossing only; S109 fallback; side/evasion unchanged')
    config['entry_frame']=dict(type='track',package='model/stage2/s161',fallback='s109',side=False,min_n=0,budget_seconds=240)
    # budget_seconds is schema compatibility only: execution is bounded per file,
    # never by cumulative time or another input's data.
    adds={'model/stage2/s118/adapter.py':HERE/'adapter.py',
          'model/stage2/s161/predict.py':HERE/'predict.py',
          'model/stage2/s161/params.json':HERE/'params.json',
          'model/licenses/S161-NOTICE.txt':HERE/'NOTICE.txt'}
    for dst,src in adds.items():
        target=root/dst;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
    rule_path=root/'model/stage2/s118/rule.json'
    rule_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    members=build.tree(root)
    changed=sorted(k for k in old['members'] if members.get(k)!=old['members'][k])
    added=sorted(set(members)-set(old['members']))
    assert changed==['model/stage2/s118/adapter.py','model/stage2/s118/rule.json'],changed
    assert added==['model/licenses/S161-NOTICE.txt','model/stage2/s161/params.json','model/stage2/s161/predict.py'],added
    receipt=dict(old,candidate_id='S161',root=str(root),rule=config,members=members,
                 manifest_sha256=build.manifest_sha(members),comparison_base='S156_casc',
                 s161=dict(base_sha256=expected,modified=changed,added=added,
                           changed_output_fields_vs_s156=['stage2.entry_frame']))
    receipt['added_files']=dict(old['added_files'])
    for name in changed+added:
        receipt['added_files'][name]=dict(members[name],source='candidates/s161_entry_cross')
    (DEST/'stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['s161']))


if __name__=='__main__':main()
