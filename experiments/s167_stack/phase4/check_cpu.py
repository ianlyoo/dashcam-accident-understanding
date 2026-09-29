"""Exercise the staged S175 postprocessor's final-anchor and failure policy."""
import importlib.util
import pathlib
import sys
import types
from unittest.mock import patch
import pandas as pd

ROOT=pathlib.Path('$DATA_DIR/s167_stack/phase4/candidate')
SOURCE=ROOT/'model/stage2/s175_entry.py'
spec=importlib.util.spec_from_file_location('_s175_cpu',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
sys.modules['torch']=fake_torch
pkg=types.SimpleNamespace()
class Tracker:
    def __init__(self,*args):pass
    analyze=lambda self,paths,numbers,anchor: {'entry_index':90 if anchor==60 else 65,'reason':'crossing'}
    close=lambda self:None
pkg.Tracker=Tracker;pkg._s174_analyze=Tracker.analyze
pkg.decide=lambda result,_: (result['entry_index'],None)
adapter=types.SimpleNamespace(load_package=lambda *args:pkg,
                              position_of=lambda numbers,value:numbers.index(value))
ns={'_s118_adapter':lambda:adapter,
    '_s008_frame_paths':lambda directory:[pathlib.Path(f'f_{n}.jpg') for n in range(101)],
    '_s008_frame_number':lambda path:int(path.stem.split('_')[1])}
base=pd.DataFrame([('A',60,50,0,'LEFT'),('B',80,70,1,'RIGHT')],
                  columns=['ID','collision_frame','entry_frame','evasion_space','entry_side'])
with patch('ctypes.CDLL', return_value=types.SimpleNamespace(malloc_trim=lambda _:None)):
    out=module.apply(ns,base,'unused','unused')
    assert out.entry_frame.tolist()==[60,65], ns['_S175_DIAGNOSTICS']
    assert out.drop(columns='entry_frame').equals(base.drop(columns='entry_frame'))
    assert [x['anchor_index'] for x in ns['_S175_DIAGNOSTICS']['clips'].values()]==[60,80]
    original=Tracker.analyze
    Tracker.analyze=lambda self,*args: (_ for _ in ()).throw(RuntimeError('injected'))
    pkg._s174_analyze=Tracker.analyze
    try:
        failed=module.apply(ns,base,'unused','unused')
        assert failed.equals(base)
    finally:
        Tracker.analyze=original;pkg._s174_analyze=original
    assert module.apply(ns,base,'unused','unused').equals(out)
    adapter.load_package=lambda *args: (_ for _ in ()).throw(RuntimeError('load'))
    assert module.apply(ns,base,'unused','unused').equals(base)
print('PASS: S172 final collision anchors; S174 clamp; protected columns; per-clip/load fail closed; recovery')
