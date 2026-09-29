"""Contract tests for exported S172 time fallback with a mocked clock."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_s172_fast_test',HERE/'runtime_fast.py')
runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)

class Clock:
    def __init__(self): self.t=0.
    def __call__(self): return self.t

def case(process_start,advance):
    clock=Clock(); positions=[]
    class Engine:
        def __init__(self,root): self.check_budget=lambda:None
        def position(self,paths,prior):
            positions.append(paths[0])
            clock.t=advance[len(positions)-1]
            return 0,dict(coarse_frames=80,fine_frames=0)
        def close(self): pass
    base=pd.DataFrame(dict(ID=['a','b','c'],collision_frame=[10,20,30],entry_frame=[5,7,9],
        evasion_space=[0,1,0],entry_side=['LEFT','RIGHT','LEFT']))
    priors={i:{'s172_prior':dict(frames=[0],probs=[1.])} for i in base.ID}
    ns={'_S118_LAST_DIAGNOSTICS':{'s142_clips':priors},'_S172_PROCESS_START':process_start,
        '_s008_frame_paths':lambda folder:[Path('0.jpg')],
        '_s008_frame_number':lambda path:0}
    with patch.object(runtime,'VisualCollision',Engine),patch.object(torch.cuda,'is_available',lambda:False):
        out=runtime.apply(ns,base,Path('/unused'),Path('/unused'),clock=clock)
    assert out[['ID','evasion_space','entry_side']].equals(base[['ID','evasion_space','entry_side']])
    return base,out,ns['_S172_DIAGNOSTICS'],positions

def main():
    base,out,diag,positions=case(0.,[500.,601.])
    assert out.collision_frame.tolist()==[0,20,30]
    assert out.entry_frame.tolist()==[0,7,9]
    assert diag['budget_triggered']=='s172_cumulative_600s'
    assert diag['covered_clips']==1 and diag['remaining_s171_clips']==2 and len(positions)==2
    base,out,diag,positions=case(-2700.,[])
    assert out.equals(base) and not positions
    assert diag['budget_triggered']=='process_wall_45m'
    assert diag['covered_clips']==0 and diag['remaining_s171_clips']==3
    print('time guard and protected-field fallback passed')

if __name__=='__main__': main()
