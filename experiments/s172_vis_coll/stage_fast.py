"""Stage runtime-safe revision3 from exact, QA-passed revision2 without retraining."""
import ast
import json
from pathlib import Path
import shutil
import zipfile
from build import sha

HERE=Path(__file__).resolve().parent
WORK=Path('$DATA_DIR/s172_vis_coll')
BASE=Path('$DATA_DIR/releases/S171_ent.zip')
SOURCE=WORK/'revision2'
DEST=WORK/'revision3'
NOTICE='model/licenses/S172-NOTICE.txt'
INFERENCE='inference.py'
RUNTIME='model/stage2/s172/runtime.py'
CONFIG='model/stage2/s172/config.json'

def main():
    assert not DEST.exists(),'Preserve any existing revision3'
    result=json.loads((HERE/'rate_2p5.json').read_text())
    assert result['hits']==482 and len(result['picks'])==750 and result['cached_features_exact']
    prior=json.loads((SOURCE/'build.json').read_text())
    assert prior['base_sha256']=='f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315'
    assert sha(BASE)==prior['base_sha256']
    (DEST/'logs').mkdir(parents=True); (DEST/'qa').mkdir()
    root=DEST/'candidate'
    shutil.copytree(SOURCE/'candidate',root)
    for path in root.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8-sig'))
    original=(root/INFERENCE).read_bytes()
    with zipfile.ZipFile(BASE) as archive: carrier=archive.read(INFERENCE)
    assert original.startswith(carrier)
    addition=original[len(carrier):]
    assert addition.startswith(b'\n# S172 applies after complete S171')
    prefix=(b'\n# S172 clocks the visual extension from module initialization.\n'
            b'import time as _s172_time\n_S172_PROCESS_START = _s172_time.perf_counter()\n')
    (root/INFERENCE).write_bytes(carrier+prefix+addition)
    shutil.copy2(HERE/'runtime_fast.py',root/RUNTIME)
    config=json.loads((root/CONFIG).read_text())
    assert config['coarse_hz']==4 and not config['refine']
    config.update(coarse_hz=2.5,batch=16,selected_oof_hits=482,
                  own_limit_seconds=600,process_limit_seconds=2700,
                  oof_rate_record='rate_2p5.json')
    (root/CONFIG).write_text(json.dumps(config,indent=2)+'\n')
    with (root/NOTICE).open('a') as stream:
        stream.write('\nS172 revision3: exact cached-frame 2.5 Hz subset, two-seed 30-epoch\n'
            'temporal heads unchanged; selected five-fold OOF 482/750 versus\n'
            '485/750 at 4 Hz and S160 proxy 458/750. Frozen DINO weights,\n'
            'batched FP16 autocast; 600 s S172 and 2700 s process guards\n'
            'return exact S171 collision and entry for remaining files.\n')
    current={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    changed=[name for name in current if current[name]!=prior['members'][name]]
    assert set(changed)=={INFERENCE,RUNTIME,CONFIG,NOTICE},changed
    assert current['model/stage2/s172/backbone/model.safetensors']==prior['members']['model/stage2/s172/backbone/model.safetensors']
    for path in root.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8-sig'))
    selected=dict(prior['selection']['best'],hits=result['hits'],fold_hits=result['folds'],rate_hz=2.5)
    selection=dict(prior['selection'],best=selected,
                   original_best=prior['selection']['best'],rate_oof_source='rate_2p5.json')
    build=dict(prior,members=current,config=config,selection=selection,
        speed_revision=dict(rate_oof=result,autocast='float16',batch=16,
            guard_s172_seconds=600,guard_process_seconds=2700,
            previous_build_sha256=sha(SOURCE/'build.json'),
            previous_benchmark=json.loads((SOURCE/'qa/benchmark.json').read_text()),
            changed_from_revision2=changed,
            source_sha256={p.name:sha(p) for p in HERE.glob('*.py')}))
    (DEST/'build.json').write_text(json.dumps(build,indent=2)+'\n')
    print(json.dumps(dict(event='revision3_staged',root=str(root),build_sha256=sha(DEST/'build.json'),
                          changed=changed,selected_oof_hits=result['hits'])),flush=True)
if __name__=='__main__': main()
