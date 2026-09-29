"""CPU contract/fault test of exported S144 and S164, using synthetic model fixtures.

This does not establish neural output parity or real multiprocessing/CUDA behavior.
It executes the real merged locate, collision adapter and S164 dispatcher.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as N
from unittest.mock import patch
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', S164_DETECT_BATCH='2')
import numpy as np
import pandas as pd
import cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from candidates.s167_stack.build_stack import sha
WORK = Path('$DATA_DIR/s167_stack/phase2b')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    root = WORK / 'candidate'
    pkg = load(root / 'model/stage2/s144/predict.py', '_s170_cpu_pkg')
    base = load(WORK / 'reference_s160/model/stage2/s144/predict.py', '_s160_cpu_pkg')
    runtime = load(root / 'model/runtime_s164/runtime.py', '_s170_cpu_runtime')
    inputs = WORK / 'cpu_inputs'
    for clip in ('019','076'):
        folder = inputs / 'images' / clip
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(40):
            assert cv2.imwrite(str(folder / f'{i:03d}.png'), np.full((4,4,3), i, np.uint8))
    features = {k:np.arange(40, dtype=float) for k in ('ego_speed','ego_theta','warp_diff')}
    def frame_paths(p): return sorted(Path(p).glob('*.png'))
    scope = dict(DEFAULT_MOTION=N(), DEFAULT_DECISION=N(baseline_window=1),
        extract_folder_features=lambda p: {k:v.copy() for k,v in features.items()},
        collision_saliency=lambda f: np.arange(40), _normalized_jolt=lambda x,w:x,
        _h1_candidate_vector=lambda *a:[0.], frame_paths=frame_paths,
        frame_numbers=lambda p:list(range(40)))
    scope['locate_collision'] = lambda f,d=None: 30
    def folder(p):
        f = scope['extract_folder_features'](p)
        return dict(ID=Path(p).name, collision_frame=scope['locate_collision'](f),
                    entry_frame=30, entry_side='LEFT', evasion_space=0)
    scope['predict_folder'] = folder
    ns = dict(_S2_COLLISION_NAMESPACE=scope, __file__=str(root/'inference.py'), pd=pd,
        _S2_CHAMPION_PREDICT_STAGE2=lambda *a:None,
        _s008_predict_sides=lambda *a:{'019':'LEFT','076':'LEFT'},
        _s008_frame_paths=frame_paths, _s008_frame_number=lambda p:int(p.stem),
        _S012_LAST_DIAGNOSTICS={k:{'decisive':True} for k in ('019','076')})
    def baseline(d,m):
        _s012_capture_predict_sides = None  # actual function co_names guard below
        getattr(ns, '_s012_capture_predict_sides', None)
        ns['_s008_predict_sides']()
        return pd.DataFrame([scope['predict_folder'](p) for p in sorted((Path(d)/'images').iterdir())])
    # S164 inspects co_names; global reference mirrors the actual S109 signature.
    exec('def baseline(d,m):\n _s012_capture_predict_sides\n ns["_s008_predict_sides"]()\n return pd.DataFrame([scope["predict_folder"](p) for p in sorted((Path(d)/"images").iterdir())])',
         dict(ns=ns, pd=pd, scope=scope, Path=Path, _s012_capture_predict_sides=None), funcs := {})
    ns['_S118_BASE_PREDICT_STAGE2'] = funcs['baseline']
    fault = [None]
    detections = []
    def choose(*args):
        args[-1].update(votes=np.array([1.]), pcts=[])
        return 0
    def init(self, emb, package_dir):
        self.scope=scope; self.path=Path(package_dir);self.embed_k=0;self.members=[];self.base_names=['base']
        self._candidates=lambda f,n,fps: ([(31,30)], np.arange(40))
        self.box=N(indices=lambda *a:[1,2], vector=lambda *a:[0.])
        self.track=N(_hist=lambda *a:np.zeros(64))
        self.F=N(CONTACT_NAMES=['contact'], contact_vector=lambda *a:[0.], context_matrix=lambda x,n:np.zeros((len(x),0)))
        # base vector has two fields plus one contact = 3 + 106.
        self.base_names=['base','box']
        self.refiner={'format':'s147_refiner_v1'}
        self.R=N(refine=lambda *a:(30,{}))
        self.loc={'n_ranked':1,'k':1,'member_pct_order':[], 'signals':[], 'feature_names':list(range(109))}
        self.loc_error=None;self.nets=[]
        self.L=N(rows=lambda *a:(np.zeros((1,109)),np.ones(1),np.array([29])))
        def localize(*a):
            if fault[0]=='raise' and self.active=='076': raise RuntimeError('CPU injected localizer failure')
            if fault[0]=='invalid' and self.active=='076': return -1,np.array([np.nan])
            return 29,np.ones(1)
        self.L.localize=localize
        def detector(images):
            detections.append(len(images))
            if len(images)>1: raise RuntimeError('CPU injected detector batch OOM')
            return [np.zeros((0,4)) for _ in images]
        self.detector=detector
    original_locate=pkg.Reranker.locate
    def locate(self,f,paths):
        self.active=paths[0].parent.name
        return original_locate(self,f,paths)
    def prefetch(source,stage,paths,original,diag):
        diag.update(prefetched=len(paths), workers=1, synthetic_cpu_prefetch=True)
        return {str(p.resolve()):original(p) for p in paths}
    reports=[]
    def call():
        diag={}
        out=runtime.predict(ns, lambda d,m:pkg.predict_collision_only(ns,d,m,root/'model/stage2/s144',0,diag),inputs,root/'model/stage2',2)
        assert scope['extract_folder_features'] is hooks[0] and scope['locate_collision'] is hooks[1]
        assert scope['predict_folder'] is hooks[2] and cv2.imread is hooks[3]
        assert ns['_S118_BASE_PREDICT_STAGE2'] is hooks[4]
        r=ns['_S164_LAST_DIAGNOSTICS']
        assert r['unused_features']==0 and r['lazy_scene_skipped']==['019','076']
        assert all(v['detector_batch']==2 and v['detector_singleton_retries']==2 for v in diag['s142_clips'].values())
        return out,diag,r
    hooks=(scope['extract_folder_features'],scope['locate_collision'],scope['predict_folder'],cv2.imread,ns['_S118_BASE_PREDICT_STAGE2'])
    with patch.object(pkg.Reranker,'__init__',init), patch.object(pkg.Reranker,'locate',locate), patch.object(pkg.Reranker,'close',lambda s:None), patch.object(pkg,'choose',choose), patch.object(runtime,'prefetch',prefetch), patch.dict(sys.modules, {'torch':N(cuda=N(is_available=lambda:False))}):
        healthy,_,_=call()
        assert healthy.collision_frame.tolist()==[29,29] and healthy.entry_frame.tolist()==[29,29]
        for kind in ('raise','invalid'):
            fault[0]=kind
            failed,diag,r=call()
            assert failed.collision_frame.tolist()==[29,30] and failed.entry_frame.tolist()==[29,30]
            assert failed[['ID','entry_side','evasion_space']].equals(healthy[['ID','entry_side','evasion_space']])
            assert diag['s142_clips']['076']['loc_error'] and 'loc_to' not in diag['s142_clips']['076']
            reports.append(dict(fault=kind, diagnostics=diag, runtime=r))
        fault[0]=None
        recovered,_,_=call()
        assert recovered.equals(healthy)
    report=dict(passed=True, scope='Synthetic CPU model fixtures; actual merged locate, collision adapter, S164 lazy dispatcher and restoration. Real GPU/multiprocess parity remains pending.',
        runtime_sha256=sha(root/'model/runtime_s164/runtime.py'), s144_sha256=sha(root/'model/stage2/s144/predict.py'),
        source_sha256=sha(__file__), build_sha256=sha(WORK/'build.json'), faults=reports,
        next_call_recovers=True, hooks_restored=True, detector_batch2_singleton_retry=True,
        other_clip_preserved=True, entry_clamp_preserved=True, cuda_used=False, wsl_used=False)
    (WORK/'cpu_fault.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
