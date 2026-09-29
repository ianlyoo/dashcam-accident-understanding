"""S174 preserves every accepted S161 decision and fail-closes on extension faults."""
import ast,pathlib,types,unittest,zipfile
H=pathlib.Path(__file__).resolve().parent
import numpy as np
with zipfile.ZipFile('$DATA_DIR/releases/S171_ent.zip') as z:
 source=z.read('model/stage2/s161/predict.py')
def module(base):
 m=types.ModuleType('_s174_test')
 exec(compile(source,'S171:s161/predict.py','exec'),m.__dict__)
 m.Tracker.analyze=lambda self,*args:base
 exec(compile((H/'enhance.py').read_bytes(),'enhance.py','exec'),m.__dict__)
 return m
def crossing():
 return dict(reason='crossing',entry_index=12,bracket=[11,12],fps=10,
             inside_samples=3,outside_samples=3,track_len=6,collision_index=45)
class Policy(unittest.TestCase):
 def test_all_harness_sources_parse(self):
  for path in H.glob('*.py'):ast.parse(path.read_bytes(),filename=str(path))
 def test_original_firing_is_identical_and_no_fallback_runs(self):
  original=crossing();m=module(original)
  def forbidden(*args):raise AssertionError('fallback ran')
  m._s174_fallback=forbidden
  got=m.Tracker.analyze(object(),[],[],45)
  self.assertIs(got,original)
 def test_fallback_error_retains_original(self):
  original=dict(reason='no_anchor',entry_index=None);m=module(original)
  m._s174_fallback=lambda *args:(_ for _ in ()).throw(RuntimeError('test'))
  self.assertIs(m.Tracker.analyze(object(),[],[],45),original)
 def test_supported_secondary_actor_crossing(self):
  original=dict(reason='no_anchor',entry_index=None);m=module(original)
  class FakeClip:
   def __init__(self,paths,detector,params,cache=None):
    self.p=params;self.dets={};self.hists={};self.aspect=.5625
   def ensure(self,indices):
    for i in indices:
     if i in self.dets:continue
     a=[.44,.50,.64,.80,.90,3]
     x=.05+.010*i;b=[x,.50,x+.10,.80,.90,3]
     self.dets[i]=np.asarray([a,b],np.float32)
     self.hists[i]=np.vstack([np.eye(2,64,dtype=np.float32)[0],np.eye(2,64,dtype=np.float32)[1]])
  m.Clip=FakeClip
  fake=types.SimpleNamespace(params=m.Params(),detector=object())
  result=m.Tracker.analyze(fake,[pathlib.Path('frame_%06d.jpg'%i) for i in range(50)],list(range(50)),45)
  self.assertEqual(result['reason'],'crossing')
  self.assertIsNotNone(m.decide(result,'s109')[0])
  self.assertLess(result['entry_index'],45)
if __name__=='__main__':unittest.main()
