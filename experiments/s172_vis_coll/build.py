"""Exact-S171 additive stage; requires an observed >10/750 OOF gain."""
import argparse
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
DATA=Path('$DATA_DIR'); WORK=DATA/'s172_vis_coll'
BASE=DATA/'releases/S171_ent.zip'
SHA='f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315'

APPEND='''
# S172 applies after complete S171 entry reconstruction; only collision and its clamp change.
_S172_BASE_STAGE2 = predict_stage2

def _s172_runtime():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parent / 'model/stage2/s172/runtime.py'
    spec = importlib.util.spec_from_file_location('_s172_runtime', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def predict_stage2(data_dir, model_dir):
    from pathlib import Path
    base = _S172_BASE_STAGE2(data_dir, model_dir)
    try:
        runtime = _s172_runtime()
        return runtime.apply(globals(), base, data_dir,
                             Path(__file__).resolve().parent / 'model/stage2/s172')
    except Exception as exc:
        globals()['_S172_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}
        return base
'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',required=True)
    ap.add_argument('--refine',action='store_true')
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    ap.add_argument('--tag',default='v1')
    ap.add_argument('--epoch',type=int,required=True)
    args=ap.parse_args()
    selection=json.loads(Path(args.selection).read_text())
    selected=selection['best']
    oof_audit=json.loads((HERE/'oof_audit.json').read_text())
    assert oof_audit['passed'] and oof_audit['head_sha256']==sha(HERE/'head.py')
    assert selected['hits']>468,'User stop gate: no >10/750 OOF improvement'
    assert selected['tag']==f'{args.tag}_ep{args.epoch}_s'+''.join(map(str,args.seeds))
    fit=json.loads((WORK/f'fit_{args.tag}_oof.json').read_text())
    expected_ids={row['id'] for row in json.loads((HERE/'inputs.json').read_text())}
    assert sha(BASE)==SHA,'Wrong S171'
    target=WORK/'candidate'; assert not target.exists(),'Preserve existing stage'
    target.mkdir(parents=True)
    before={}
    with zipfile.ZipFile(BASE) as archive:
        for entry in archive.infolist():
            if entry.is_dir(): continue
            name=entry.filename; p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name
            dst=target/name; dst.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(entry) as source,dst.open('wb') as out: shutil.copyfileobj(source,out,2**20)
            before[name]=sha(dst)
    inference=target/'inference.py'
    original_inference=inference.read_bytes()
    inference.write_bytes(original_inference+APPEND.encode())
    p=target/'model/stage2/s144/predict.py'
    original=p.read_text()
    needle="                detail.update(loc_from=int(onset),loc_to=int(new));onset=int(new)"
    assert original.count(needle)==1
    new=needle+"""
                # Capture failure must preserve the already selected S160 onset.
                _s172_capture_start=time.perf_counter()
                try:
                    _fr,_inv=np.unique(FR[M],return_inverse=True)
                    _prob=np.bincount(_inv,weights=_p[M])
                    detail['s172_prior']={'frames':_fr.tolist(),'probs':_prob.tolist()}
                except Exception:
                    pass
                finally:
                    detail['s172_capture_seconds']=time.perf_counter()-_s172_capture_start
"""
    p.write_text(original.replace(needle,new),newline='\n')
    package=target/'model/stage2/s172'; package.mkdir()
    for name in ('runtime.py','head.py','refinement.py'): shutil.copy2(HERE/name,package/name)
    backbone=package/'backbone'; backbone.mkdir()
    for name in ('config.json','model.safetensors','LICENSE.txt','README.md','preprocessor_config.json'):
        shutil.copy2(DATA/'pretrained/dinov2-large-47b73eef'/name,backbone/name)
    import torch
    heads=[]
    for seed in args.seeds:
        record=torch.load(WORK/'heads'/f'{args.tag}_f-1_s{seed}_ep{args.epoch}.pt',map_location='cpu',weights_only=False)
        assert record['fold']==-1 and record['seed']==seed and record['epoch']==args.epoch
        assert record['loss']==fit['args']['loss'] and record['hidden']==fit['args']['hidden']
        assert set(record['training_ids'])==expected_ids and not record['held_out_ids']
        assert record['source_binding']['head.py']==sha(HERE/'head.py')
        assert record['source_binding']['inputs.json']==sha(HERE/'inputs.json')
        name=f'head_{seed}.pt'; heads.append(name)
        torch.save(dict(state=record['state'],hidden=record['hidden']),package/name)
    config=dict(format='s172_visual_v1',heads=heads,kind=selected['kind'],weight=selected['weight'],half=selected['half'],
        refine=args.refine,selected_oof_hits=selected['hits'],baseline_oof_hits=458,coarse_hz=4,refine_hz=10)
    if args.refine:
        config['coarse_schedule']=selection.get('coarse_schedule',selection.get('coarse',selected))
    (package/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    export=HERE/'export'; export.mkdir(exist_ok=False)
    for name in heads+['config.json']: shutil.copy2(package/name,export/name)
    (export/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (target/'model/licenses/S172-NOTICE.txt').write_text(
        'S172 visual collision timing on exact S171. DINOv2-Large by Meta/Facebook Research,\n'
        'https://github.com/facebookresearch/dinov2 and https://huggingface.co/facebook/dinov2-large\n'
        'revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c, Apache-2.0; upstream license\n'
        'and model card are preserved in model/stage2/s172/backbone. Frozen backbone;\n'
        'temporal head fitted on the same 750 public Nexar positive clips documented\n'
        'by the inherited S160 notice. No private evaluation training or extraction.\n'
        'Only collision_frame and entry_frame=min(S171 entry, collision) can change.\n'
        'Per-file failure preserves exact S171. OOF decoder selection reuses 750\n'
        'clips; S160 458 is an incomplete-final-ensemble OOF proxy.\n')
    after={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()}
    changed=[n for n in before if before[n]!=after[n]]
    assert sorted(changed)==['inference.py','model/stage2/s144/predict.py']
    assert inference.read_bytes().startswith(original_inference)
    for p in target.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8-sig'))
    record=dict(base=str(BASE),base_sha256=SHA,changed=changed,added=sorted(set(after)-set(before)),
        unchanged_base_members=len(before)-len(changed),members=after,selection=selection,config=config)
    trials={p.name:json.loads(p.read_text()) for p in sorted(WORK.glob('summary_*.json')) if not p.name.startswith('summary_fine_')}
    record['all_coarse_trials']={name:dict(grid=r['grid'],best=r['best'],all=r['all']) for name,r in trials.items()}
    record['coarse_decoder_selections_tried']=sum(r['grid'] for r in trials.values())
    fine_trials={p.name:json.loads(p.read_text()) for p in sorted(WORK.glob('summary_fine_*.json'))}
    record['fine_decoder_selections_tried']=sum(r['grid'] for r in fine_trials.values())
    record['fine_trials']=fine_trials
    record['initial_fixed_refinement']=json.loads((WORK/'refined_v2mass_ep30_s01.json').read_text())
    record['training_protocols']={p.name:json.loads(p.read_text()) for p in sorted(WORK.glob('fit_*.json'))}
    record['source_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
    (WORK/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(event='staged',changed=changed,members=len(after))),flush=True)

if __name__=='__main__': main()
