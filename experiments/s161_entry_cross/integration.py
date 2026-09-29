"""Exact S156/S161 exported long-clip Stage2 output and single-file parity."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import gc
import importlib.util
import json
from pathlib import Path
import time
import torch
import cv2

DATA=Path('$DATA_DIR')
WORK=DATA/'s161_entry_cross'
REPO=Path(__file__).resolve().parents[2]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    original_loader=torch.utils.data.DataLoader
    class SingleProcessLoader(original_loader):
        def __init__(self,*args,**kwargs):
            kwargs['num_workers']=0;kwargs['persistent_workers']=False
            kwargs.pop('prefetch_factor',None)
            super().__init__(*args,**kwargs)
    torch.utils.data.DataLoader=SingleProcessLoader
    torch.cuda.set_per_process_memory_fraction(1.75*1024**3/torch.cuda.get_device_properties(0).total_memory)
    images=WORK/'integration/images';images.mkdir(parents=True,exist_ok=True)
    for vid in ['00060','00208','00533']:
        source=DATA/'s132_s2track/nexar/images'/vid
        if not source.exists():source=WORK/'images'/vid
        dest=images/vid
        if not dest.exists():dest.symlink_to(source,target_is_directory=True)
    frames={p.name:len(list(p.glob('*.jpg'))) for p in images.iterdir()}
    assert all(n>310 for n in frames.values()),frames
    roots=[DATA/'stage2_s118/S156_casc/candidate',WORK/'stage/candidate']
    report=dict(frames=frames,rows={},seconds={},test_only_loader_workers=0)
    mods=[]
    for name,root in zip(['s156','s161'],roots):
        mod=load(root/'inference.py','_s161_integration_'+name);mods.append(mod)
        t=time.perf_counter()
        out=mod.predict_stage2(images.parent,root/'model/stage2')
        report['seconds'][name]=time.perf_counter()-t
        report['rows'][name]=out.to_dict(orient='records')
        out.to_csv(WORK/(name+'_long.csv'),index=False)
        report[name+'_diagnostics']=mod._S118_LAST_DIAGNOSTICS
        assert not mod._S118_LAST_DIAGNOSTICS.get('rule_error'),mod._S118_LAST_DIAGNOSTICS
        print(name,report['seconds'][name],report['rows'][name],flush=True)
        if name=='s156':base=out
        else:actual=out
        gc.collect();torch.cuda.empty_cache()
    protected=[c for c in base.columns if c!='entry_frame']
    assert base[protected].equals(actual[protected]),'non-entry output changed'
    assert (actual.entry_frame<=actual.collision_frame).all()
    # A second tracker from the shipped package independently maps the exact
    # S156 outputs; this checks the final collision, fallback and clamp wiring.
    package=load(roots[1]/'model/stage2/s161/predict.py','_s161_integration_reference')
    tracker=package.Tracker(roots[1]/'model/stage2/s161')
    reference=[]
    for r in base.itertuples(index=False):
        paths=mods[1]._s008_frame_paths(images/r.ID)
        nums=[mods[1]._s008_frame_number(p) for p in paths]
        c=min(range(len(nums)),key=lambda i:abs(nums[i]-r.collision_frame))
        result=tracker.analyze(paths,nums,c)
        pos,_=package.decide(result,'s109')
        entry=min(r.entry_frame if pos is None else nums[pos],r.collision_frame)
        reference.append(entry)
    tracker.close()
    assert actual.entry_frame.tolist()==reference
    single=WORK/'integration_single/images';single.mkdir(parents=True,exist_ok=True)
    dest=single/'00208'
    if not dest.exists():dest.symlink_to(images/'00208',target_is_directory=True)
    repeated=mods[1].predict_stage2(single.parent,roots[1]/'model/stage2')
    assert repeated.reset_index(drop=True).equals(actual[actual.ID=='00208'].reset_index(drop=True))
    report.update(passed=True,changed_rows=int((base.entry_frame!=actual.entry_frame).sum()),
                  independent_reference=True,single_batch_parity=True,
                  measured_wall_delta_seconds=report['seconds']['s161']-report['seconds']['s156'],
                  cuda_peak_bytes=torch.cuda.max_memory_allocated())
    (WORK/'integration.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('PASS',report['changed_rows'],'entry changes; exact protected outputs',flush=True)


if __name__=='__main__':
    main()
    from candidates.s161_entry_cross.finish import main as finish
    finish()
