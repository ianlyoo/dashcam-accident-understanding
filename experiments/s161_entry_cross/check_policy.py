"""Exercise crossing-only selection, uncertainty, clamp and per-file independence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import predict
import adapter
import pandas as pd
import numpy as np

r=dict(reason='crossing',entry_index=24,bracket=[23,25],fps=30,
       inside_samples=3,outside_samples=3,track_len=8,collision_index=30)
assert predict.decide(r,'s109')==(24,None)
for change in [dict(reason='inside_from_start'),dict(reason='far_inside_when_first_seen'),
               dict(reason='inside_when_first_seen'),dict(reason='frame_budget'),
               dict(bracket=[1,25]),dict(inside_samples=1),dict(outside_samples=1),
               dict(entry_index=31),dict(track_len=2)]:
    assert predict.decide(dict(r,**change),'s109')==(None,None),change

class Tracker:
    device='cpu'
    def __init__(self,*args):pass
    def analyze(self,paths,numbers,anchor):return dict(r,entry_index=24)
    def close(self):pass
class Package:
    Tracker=Tracker
    decide=staticmethod(predict.decide)
original=adapter.load_package
adapter.load_package=lambda *args:Package
ns={'_s008_frame_paths':lambda path:[Path('frame_%06d.jpg'%i) for i in range(40)],
    '_s008_frame_number':lambda path:int(path.stem.split('_')[-1])}
rule=dict(entry_frame=dict(type='track',package='model/stage2/s161',fallback='s109',side=False,min_n=0,budget_seconds=.000001))
base=pd.DataFrame([dict(ID=k,collision_frame=30,entry_frame=10,evasion_space=1,entry_side='LEFT') for k in ['a','b']])
try:
    got=adapter._predict_track(ns,Path('.'),None,rule,base_call=lambda:base)
    reverse=adapter._predict_track(ns,Path('.'),None,rule,base_call=lambda:base.iloc[::-1])
    assert got['entry_frame'].tolist()==[24,24]
    assert got.sort_values('ID').reset_index(drop=True).equals(reverse.sort_values('ID').reset_index(drop=True))
    assert got.drop(columns='entry_frame').equals(base.drop(columns='entry_frame'))
    Package.decide=staticmethod(lambda result,fallback:(39,None))
    clamped=adapter._predict_track(ns,Path('.'),None,rule,base_call=lambda:base)
    assert clamped.entry_frame.tolist()==[30,30]
finally:adapter.load_package=original

# An unambiguous synthetic actor moves across the left ego-lane boundary.
# This exercises tracking, geometry and native-frame crossing selection together.
class SyntheticDetector:
    def load(self,path):
        image=np.zeros((90,160,3),dtype=np.uint8)
        image[0,0,0]=int(path.stem.split('_')[-1])
        return image
    def __call__(self,images):
        return [np.array([[float(im[0,0,0])*.01,.6,float(im[0,0,0])*.01+.2,.8,.99,3]],np.float32) for im in images]
paths=[Path('frame_%06d.jpg'%i) for i in range(50)]
p=predict.Params(grid='absolute',horizon_mode='fixed',inset=.05)
result=predict.analyze(paths,list(range(50)),40,SyntheticDetector(),p)
assert result['reason']=='crossing',result
assert predict.decide(result,'s109')==(9,None),result
clip=predict.Clip(paths,SyntheticDetector(),predict.Params(max_detect_frames=2))
try:clip.ensure([0,1,2])
except predict.FrameBudgetExceeded:pass
else:raise AssertionError('per-file detection bound not enforced')
print('PASS: tracking geometry, confidence, clamp, frame bound, protected fields, order independence')
