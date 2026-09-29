"""CPU-only preflight of WSL path translation for the queued CUDA replays."""
import importlib.util,json,pathlib,pickle,zipfile
D=pathlib.Path('$DATA_DIR')
with (D/'s132_s2track/cache_ccd_frcnn_640_0.3.pkl').open('rb') as f:raw=pickle.load(f)
translated={k.replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'):v for k,v in raw.items()}
sample=next(iter(translated));assert pathlib.Path(sample).exists(),sample
row=next(r for r in map(json.loads,(D/'s173_wheel_entry/evaluation.jsonl').read_text().splitlines()) if r['kind']=='ccd')
folder=pathlib.Path(row['folder'].replace(chr(92),'/').replace('$DATA_DIR/','/mnt/d/'))
assert folder.exists(),folder
paths=sorted(folder.glob('*.jpg'));assert paths
candidate=D/'s174_cov/candidate/model/stage2/s161/predict.py'
with zipfile.ZipFile(D/'releases/S171_ent.zip') as z:original=z.read('model/stage2/s161/predict.py')
assert candidate.read_bytes().startswith(original)
spec=importlib.util.spec_from_file_location('_s174_stage_preflight',candidate)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.Tracker.analyze is module._s174_analyze
assert module._s174_original_analyze is not module._s174_analyze
print('translated_cached_frames',len(translated),'sample_exists',sample,'labeled_folder',folder,'frames',len(paths))
