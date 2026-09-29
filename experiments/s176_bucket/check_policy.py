"""Focused S176 preservation, bucket scope, and fail-closed tests."""
import ast
import pathlib
import types
import unittest
import zipfile

H = pathlib.Path(__file__).resolve().parent
D = pathlib.Path('$DATA_DIR')
PRED = 'model/stage2/s161/predict.py'
with zipfile.ZipFile(D/'releases/S174_cov.zip') as z:
    BASE = z.read(PRED)

def module(original, fallback=None):
    m = types.ModuleType('_s176_test')
    exec(compile(BASE, 'S174:'+PRED, 'exec'), m.__dict__)
    m._s174_original_analyze = lambda self,*args:original
    m._s174_fallback = fallback or (lambda *args:m.Result(reason='s174_no_supported_crossing',entry_index=None))
    exec(compile((H/'enhance.py').read_bytes(), 'S176:enhance.py', 'exec'), m.__dict__)
    assert m.Tracker.analyze is m._s176_analyze
    return m

def proxy(m):return types.SimpleNamespace(params=m.Params())

class Policy(unittest.TestCase):
    def test_all_sources_parse(self):
        for p in H.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))

    def test_accepted_s161_kept_identical(self):
        original=dict(reason='crossing',entry_index=12,bracket=[11,12],fps=10,
                      inside_samples=3,outside_samples=3,track_len=6,collision_index=45)
        def forbidden(*args):raise AssertionError('S174 fallback ran')
        m=module(original,forbidden);got=m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45)
        self.assertIs(got,original)
        self.assertEqual(m.decide(got,'s109')[0],12)

    def test_accepted_s174_kept_identical(self):
        original=dict(reason='inside_from_start',entry_index=0)
        accepted=m_result=dict(reason='crossing',entry_index=21,bracket=[20,21],fps=10,
                               inside_samples=3,outside_samples=3,track_len=6,collision_index=45)
        m=module(original,lambda *args:accepted)
        got=m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45)
        self.assertIs(got,accepted)
        self.assertEqual(m.decide(got,'s109')[0],21)

    def test_first_clip_only_inside_from_start(self):
        m=module(dict(reason='inside_from_start',entry_index=0))
        got=m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45)
        self.assertEqual(got['reason'],'s176_inside_from_start')
        self.assertEqual(m.decide(got,'s109')[0],0)

    def test_fixed_offset_only_inside_when_first_seen(self):
        m=module(dict(reason='inside_when_first_seen',entry_index=24))
        got=m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45)
        self.assertEqual(got['reason'],'s176_inside_first_minus_0p2s')
        self.assertEqual(m.decide(got,'s109')[0],22)
        longpaths=list(range(m.Params().long_n+1))
        got=m.Tracker.analyze(proxy(m),longpaths,longpaths,30)
        self.assertEqual(m.decide(got,'s109')[0],18)

    def test_zero_gain_buckets_unchanged(self):
        for reason in ('far_inside_when_first_seen','not_in_lane_at_collision'):
            original=dict(reason=reason,entry_index=10)
            m=module(original)
            self.assertIs(m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45),original)

    def test_s174_fault_retains_its_output(self):
        original=dict(reason='inside_from_start',entry_index=0)
        def fault(*args):raise RuntimeError('injected S174 failure')
        m=module(original,fault)
        self.assertIs(m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45),original)

    def test_s176_fault_retains_s174(self):
        original=dict(reason='inside_when_first_seen',entry_index='broken')
        m=module(original)
        self.assertIs(m.Tracker.analyze(proxy(m),list(range(50)),list(range(50)),45),original)

if __name__=='__main__':unittest.main()
