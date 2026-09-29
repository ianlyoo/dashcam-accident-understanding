"""Real Linux/CUDA three-stage QA for an S118-family rule override: exact export, offline read guard."""
import argparse
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import zipfile

from candidates.s118_s2rule.build import BASE, DATA, EXPECTED, HERE, MODEL_DIR, REPO, identity, read, tree, manifest_sha, save_new
from candidates.s108_export.harness_common import (
    THREAD_ENV, framework_setup, offline_readonly_guard, linux_abstract_ipc, host_memory,
)

OUTPUT_COLUMNS = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']

# Pin the harness text for PyTorch worker tracebacks while agents edit this shared file.
_QA_SOURCE_BYTES = Path(__file__).read_bytes()



def load_entry(path):
    name = '_s118_unregistered_actual_entry'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert name not in sys.modules
    return module.__dict__


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_values(rule, numbers, base_collision, base_entry):
    """Independent re-derivation of the rule (does not call the adapter)."""
    n = len(numbers)
    clip = lambda i: min(max(int(i), 0), n - 1)
    def pos(v):
        if v in numbers:
            return numbers.index(v)
        best = min(abs(x - v) for x in numbers)
        return next(i for i, x in enumerate(numbers) if abs(x - v) == best)
    c_spec = rule.get('collision_frame', {'type': 'passthrough'})
    e_spec = rule.get('entry_frame', {'type': 'passthrough'})
    if c_spec['type'] == 'passthrough' or (c_spec['type'] in ('position_prior', 'rerank') and n <= c_spec['min_n']):
        collision, final_pos = base_collision, pos(base_collision)
    elif c_spec['type'] == 'rerank':
        raise ValueError('long rerank needs a feature-level reference')
    elif c_spec['type'] == 'const_position':
        final_pos = clip(c_spec['p'])
        collision = numbers[final_pos]
    elif c_spec['type'] == 'const_fraction':
        final_pos = clip(round(c_spec['f'] * (n - 1)))
        collision = numbers[final_pos]
    elif c_spec['type'] == 'floor_replace':
        final_pos = pos(base_collision) if pos(base_collision) >= clip(c_spec['lo']) else clip(c_spec['c'])
        collision = numbers[final_pos]
    else:
        hi = clip(c_spec['hi'])
        lo = min(clip(c_spec['lo']), hi)
        final_pos = sorted([lo, pos(base_collision), hi])[1]
        collision = numbers[final_pos]
    if e_spec['type'] in ('passthrough', 'track') or (e_spec['type'] == 'motion_lead_scale' and n <= e_spec['min_n']):
        entry = base_entry
    elif e_spec['type'] == 'first_frame':
        entry = numbers[0]
    else:
        anchor = final_pos if e_spec['source'] == 'final' else pos(base_collision)
        lead = e_spec['k']
        if e_spec['type'] == 'collision_minus_grid' and n > 50:
            lead = int(round(e_spec['k'] * n / 50.0))
        entry = numbers[max(anchor - lead, 0)]
    return int(collision), int(entry)


def unit_checks(adapter, workdir):
    """Pure rule logic on synthetic frame lists plus fail-closed behavior; no model calls."""
    import pandas as pd
    lists = {'n50': list(range(50)), 'n37_offset': list(range(100, 137)), 'n1': [7],
             'gappy': [0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135, 152, 170, 189, 209,
                       230, 252, 275, 299, 324, 350, 377, 405, 434, 464, 495, 527, 560, 594, 629, 665],
             'n400': list(range(1, 401))}
    rules = [
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_position', 'p': 33}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_position', 'p': -4}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_fraction', 'f': 0.67}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'clamp_position', 'lo': 30, 'hi': 49}},
        {'schema': 's118-rule-v1', 'entry_frame': {'type': 'collision_minus', 'k': 8, 'source': 's109'}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'floor_replace', 'lo': 30, 'c': 32}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_position', 'p': 33},
         'entry_frame': {'type': 'collision_minus', 'k': 40, 'source': 'final'}},
        {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_position', 'p': 33},
         'entry_frame': {'type': 'collision_minus_grid', 'k': 6, 'source': 'final'}},
        {'schema': 's118-rule-v1', 'entry_frame': {'type': 'collision_minus_grid', 'k': 6, 'source': 's109'}},
        {'schema': 's118-rule-v1', 'entry_frame': {'type': 'first_frame'}},
    ]
    cases = 0
    for rule in rules:
        adapter.validate_rule(rule)
        for numbers in lists.values():
            for base_pos in {0, len(numbers) // 3, len(numbers) - 1}:
                for jitter in (0, 1):  # jitter=1 exercises a base value absent from the list
                    bc, be = numbers[base_pos] + jitter, numbers[max(base_pos - 5, 0)]
                    got = adapter.apply_rule(rule, numbers, bc, be)
                    want = expected_values(rule, numbers, bc, be)
                    assert got == want, (rule, len(numbers), bc, got, want)
                    changed = adapter.changed_fields(rule)
                    if 'collision_frame' in changed:
                        assert got[0] in numbers
                    if 'entry_frame' in changed:
                        assert got[1] in numbers
                    cases += 1
    assert adapter.apply_rule(rules[0], list(range(50)), 35, 18) == (33, 18)
    assert adapter.apply_rule(rules[0], list(range(10)), 5, 1) == (9, 1)
    assert adapter.apply_rule(rules[3], list(range(50)), 29, 21) == (30, 21)
    assert adapter.apply_rule(rules[3], list(range(50)), 42, 38) == (42, 38)
    assert adapter.apply_rule(rules[3], list(range(20)), 5, 1) == (19, 1)
    floor = rules[5]
    assert adapter.apply_rule(floor, list(range(50)), 29, 21) == (32, 21)
    assert adapter.apply_rule(floor, list(range(50)), 30, 21) == (30, 21)
    assert adapter.apply_rule(floor, list(range(50)), 45, 40) == (45, 40)
    assert adapter.apply_rule(floor, list(range(100, 150)), 110, 101) == (132, 101)
    assert adapter.apply_rule(floor, list(range(20)), 5, 1) == (19, 1)
    grid = rules[7]
    assert adapter.apply_rule(grid, list(range(50)), 35, 18) == (33, 27)
    assert adapter.apply_rule(grid, list(range(150)), 99, 18) == (33, 15)
    assert adapter.apply_rule(grid, list(range(125)), 99, 18) == (33, 18)
    assert adapter.apply_rule(grid, list(range(30)), 5, 1) == (29, 23)
    for bad in ({'schema': 'x'}, {'schema': 's118-rule-v1', 'collision_frame': {'type': 'const_position', 'p': True}},
                {'schema': 's118-rule-v1', 'collision_frame': {'type': 'clamp_position', 'lo': 5, 'hi': 4}},
                {'schema': 's118-rule-v1', 'entry_frame': {'type': 'collision_minus', 'k': -1, 'source': 'final'}},
                {'schema': 's118-rule-v1', 'bogus': 1}):
        try:
            adapter.validate_rule(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rule accepted: %r' % bad)
    # Fail-closed with a fake namespace: missing folder keeps its row; bad rule keeps all rows.
    base = pd.DataFrame([{'ID': 'A', 'collision_frame': 35, 'entry_frame': 18, 'evasion_space': 1, 'entry_side': 'LEFT'},
                         {'ID': 'MISSING', 'collision_frame': 20, 'entry_frame': 10, 'evasion_space': 0, 'entry_side': 'RIGHT'}],
                        columns=OUTPUT_COLUMNS)
    data = workdir / 'data'
    (data / 'images' / 'A').mkdir(parents=True)
    for i in range(50):
        (data / 'images' / 'A' / ('%05d.jpg' % (i + 3))).write_bytes(b'')
    import re
    fake_ns = {'_S118_BASE_PREDICT_STAGE2': lambda d, m: base.copy(),
               '_s008_frame_number': lambda p: int(re.search(r'(\d+)$', Path(p).stem).group(1)),
               '_s008_frame_paths': lambda f: sorted(Path(f).iterdir(), key=lambda p: int(p.stem))}
    for label, rule_text in (('good', json.dumps(rules[0])), ('bad', '{"schema": "s118-rule-v1", "collision_frame": 3}')):
        folder = workdir / label
        folder.mkdir()
        shutil.copyfile(adapter.__file__, folder / 'adapter.py')
        (folder / 'rule.json').write_text(rule_text, encoding='utf-8')
        module = load_file_module(folder / 'adapter.py', '_s118_qa_' + label)
        result = module.predict(fake_ns, data, None)
        if label == 'good':
            assert result['collision_frame'].tolist() == [36, 20], result
            assert result.drop(columns='collision_frame').equals(base.drop(columns='collision_frame'))
            assert 'error' in fake_ns['_S118_LAST_DIAGNOSTICS']['clips']['MISSING']
        else:
            assert result.equals(base) and fake_ns['_S118_LAST_DIAGNOSTICS']['rule_error']
    prior_cases = position_prior_checks(adapter) + lead_scale_checks(adapter)
    rerank_cases = rerank_unit_checks(adapter, workdir)
    return {'rule_cases': cases, 'fail_closed_missing_folder': True, 'fail_closed_bad_rule': True,
            'position_prior_cases': prior_cases, 'rerank_cases': rerank_cases}


def lead_scale_checks(adapter):
    """lead_scaled_locator passes scaled leads for long clips and S109 leads otherwise."""
    class P:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    seen = []
    original = lambda f, c, d: seen.append((d.entry_min_lead, d.entry_max_lead, d.entry_default_lead)) or 7
    emb = {'DEFAULT_DECISION': P(entry_min_lead=3, entry_max_lead=20, entry_default_lead=8, onset_ratio=0.5), 'DecisionParams': P}
    spec = {'min_n': 310, 'scale': 3}
    loc = adapter.lead_scaled_locator(emb, original, spec)
    assert loc({'resid_energy': [0] * 1000}, 500) == 7 and seen[-1] == (9, 60, 24)
    assert loc({'resid_energy': [0] * 50}, 30) == 7 and seen[-1] == (3, 20, 8)
    adapter.validate_rule({'schema': 's118-rule-v1', 'entry_frame': {'type': 'motion_lead_scale', 'min_n': 310, 'scale': 3}})
    for bad in ({'type': 'motion_lead_scale', 'min_n': 310, 'scale': 0.5}, {'type': 'motion_lead_scale', 'min_n': 5, 'scale': 3}):
        try:
            adapter.validate_rule({'schema': 's118-rule-v1', 'entry_frame': bad})
        except ValueError:
            continue
        raise AssertionError('bad motion_lead_scale accepted')
    return 4


def position_prior_checks(adapter):
    """prior_locator on synthetic saliency: window argmax, soft prior, short-clip and failure passthrough."""
    import numpy as np

    class D:
        onset_ratio, max_backtrack = 0.5, 4

    def emb_for(score):
        return {'DEFAULT_DECISION': D(), 'collision_saliency': lambda f, d: np.asarray(score, dtype=float),
                '_onset_index': lambda sc, peak, ratio, back: peak}
    original = lambda f, d=None: -1
    hard = {'min_n': 310, 'lo_frac': 0.42, 'hi_frac': 0.58, 'center_frac': 0.5, 'sigma_frac': 0.0}
    score = np.zeros(1000); score[100] = 1.0; score[450] = 0.3; score[700] = 0.9
    assert adapter.prior_locator(emb_for(score), original, hard)({}, D()) == 450
    score2 = np.zeros(300); score2[10] = 1.0
    assert adapter.prior_locator(emb_for(score2), original, hard)({}, D()) == -1  # n <= min_n keeps S109
    soft = dict(hard, lo_frac=0.0, hi_frac=1.0, sigma_frac=0.06)
    score3 = np.zeros(1000); score3[100] = 1.0; score3[520] = 0.2
    assert adapter.prior_locator(emb_for(score3), original, soft)({}, D()) == 520
    broken = {'DEFAULT_DECISION': D(), 'collision_saliency': lambda f, d: 1 / 0, '_onset_index': None}
    assert adapter.prior_locator(broken, original, hard)({}, D()) == -1  # failure keeps S109
    for bad in (dict(hard, lo_frac=0.6), dict(hard, min_n=10), dict(hard, sigma_frac=True)):
        try:
            adapter.validate_rule({'schema': 's118-rule-v1', 'collision_frame': dict(bad, type='position_prior')})
        except ValueError:
            continue
        raise AssertionError('bad position_prior accepted: %r' % bad)
    try:
        adapter.validate_rule({'schema': 's118-rule-v1', 'collision_frame': dict(hard, type='position_prior'),
                               'entry_frame': {'type': 'collision_minus', 'k': 5, 'source': 'final'}})
    except ValueError:
        pass
    else:
        raise AssertionError('position_prior with entry rule accepted')
    return 6


def rerank_unit_checks(adapter, workdir):
    """Synthetic full-call patch, short passthrough, tracker composition, and load failure."""
    import pandas as pd

    collision = {'type': 'rerank', 'package': 'model/stage2/s135', 'min_n': 310}
    track = {'type': 'track', 'package': 'model/stage2/s132', 'fallback': 'first_frame',
             'side': True, 'min_n': 0, 'budget_seconds': 600}
    rule = {'schema': 's118-rule-v1', 'collision_frame': collision, 'entry_frame': track}
    adapter.validate_rule(rule)
    adapter.validate_rule(dict(rule, entry_frame={'type': 'passthrough'}))
    assert set(adapter.changed_fields(rule)) == {'collision_frame', 'entry_frame', 'entry_side', 'evasion_space'}
    assert expected_values(rule, list(range(310)), 10, 7) == (10, 7)
    assert expected_values(dict(rule, entry_frame={'type': 'passthrough'}), list(range(310)), 10, 7) == (10, 7)
    for bad in (dict(collision, min_n=True), dict(collision, min_n=49),
                dict(collision, package='../s135')):
        try:
            adapter.validate_rule(dict(rule, collision_frame=bad))
        except ValueError:
            continue
        raise AssertionError('bad rerank rule accepted: %r' % bad)

    folder = workdir / 'rerank_composition'
    folder.mkdir()
    shutil.copyfile(adapter.__file__, folder / 'adapter.py')
    (folder / 'rule.json').write_text(json.dumps(rule), encoding='utf-8')
    module = load_file_module(folder / 'adapter.py', '_s118_qa_rerank_composition')

    class D:
        pass
    default = D()
    original = lambda features, decision=None: 10
    embedded = {'locate_collision': original, 'DEFAULT_DECISION': default}
    seen = []

    def predict_folder(path):
        n = 310 if Path(path).name == 'SHORT' else 311
        c = embedded['locate_collision']({'ego_speed': [0] * n}, default)
        return {'ID': Path(path).name, 'collision_frame': c, 'entry_frame': c - 3,
                'evasion_space': int(c > 100), 'entry_side': 'LEFT'}
    embedded['predict_folder'] = predict_folder
    embedded['frame_numbers'] = lambda path: list(range(310 if Path(path).name == 'SHORT' else 311))
    calls = []

    def base(data_dir, model_dir):
        calls.append(1)
        return pd.DataFrame([embedded['predict_folder'](Path(data_dir) / key)
                             for key in ('SHORT', 'LONG')], columns=OUTPUT_COLUMNS)

    class FakeTracker:
        device = 'cpu'
        def __init__(self, path):
            pass
        def analyze(self, paths, numbers, collision_position):
            seen.append((len(numbers), collision_position))
            return {'entry_index': max(0, collision_position - 7), 'side': 'RIGHT'}
        def close(self):
            pass

    class FakeReranker:
        @staticmethod
        def make_locator(emb, own, path, min_n, diagnostics):
            return lambda features, decision=None: (200 if len(features['ego_speed']) > min_n
                                                    else own(features, decision))

    class FakeTrackPackage:
        Tracker = FakeTracker
        decide = staticmethod(lambda result, fallback: (result['entry_index'], result['side']))

    module.load_package = lambda root, path: FakeReranker if path.endswith('s135') else FakeTrackPackage
    fake_ns = {'_S2_COLLISION_NAMESPACE': embedded, '_S118_BASE_PREDICT_STAGE2': base,
               '_s008_frame_paths': lambda path: [Path('frame_%06d.jpg' % i)
                                                  for i in range(310 if path.name == 'SHORT' else 311)],
               '_s008_frame_number': lambda path: int(path.stem.split('_')[-1])}
    result = module.predict(fake_ns, folder, None)
    assert result['collision_frame'].tolist() == [10, 200], result
    assert result['entry_frame'].tolist() == [3, 193], result
    assert result['evasion_space'].tolist() == [0, 1], result
    assert result['entry_side'].tolist() == ['RIGHT', 'RIGHT'], result
    assert seen == [(310, 10), (311, 200)], seen
    assert embedded['locate_collision'] is original
    assert fake_ns['_S118_LAST_DIAGNOSTICS']['rerank'] and fake_ns['_S118_LAST_DIAGNOSTICS']['track']

    passthrough = dict(rule, entry_frame={'type': 'passthrough'})
    (folder / 'rule.json').write_text(json.dumps(passthrough), encoding='utf-8')
    result = module.predict(fake_ns, folder, None)
    assert result['collision_frame'].tolist() == [10, 200]
    assert result['entry_frame'].tolist() == [7, 197]
    assert embedded['locate_collision'] is original
    only = dict(passthrough, collision_frame=dict(collision, preserve_s109_fields=True))
    adapter.validate_rule(only)
    assert set(adapter.changed_fields(only)) == {'collision_frame', 'entry_frame'}
    (folder / 'rule.json').write_text(json.dumps(only), encoding='utf-8')
    count_before = len(calls)
    result = module.predict(fake_ns, folder, None)
    assert len(calls) == count_before + 1, 'collision-only must call S109 exactly once'
    assert embedded['predict_folder'] is predict_folder and embedded['locate_collision'] is original
    assert fake_ns['_S118_LAST_DIAGNOSTICS']['single_base_call']
    assert result['collision_frame'].tolist() == [10, 200]
    assert result['entry_frame'].tolist() == [7, 7]
    assert result['evasion_space'].tolist() == [0, 0]
    assert result['entry_side'].tolist() == ['LEFT', 'LEFT']
    earlier = result.copy()
    earlier['collision_frame'] = [3, 200]
    diagnostic = {}
    clamped = module.collision_only_output(base(folder, None), earlier, diagnostic)
    assert clamped['entry_frame'].tolist() == [3, 7]
    assert diagnostic['entry_validity_clamps'] == ['SHORT']
    assert base(folder, None)['entry_frame'].tolist() == [7, 7]
    class FakeCollisionOnlyPackage:
        @staticmethod
        def predict_collision_only(ns, data_dir, model_dir, package_dir, min_n, diagnostics):
            assert min_n == 310
            kept = ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
            revised = kept.copy()
            revised.loc[revised['ID'] == 'LONG', 'collision_frame'] = 2
            diagnostics['single_base_call'] = True
            return module.collision_only_output(kept, revised, diagnostics)
    module.load_package = lambda root, path: FakeCollisionOnlyPackage
    count_before = len(calls)
    result = module.predict(fake_ns, folder, None)
    assert len(calls) == count_before + 1
    assert result['collision_frame'].tolist() == [10, 2]
    assert result['entry_frame'].tolist() == [7, 2]
    assert result['entry_side'].tolist() == ['LEFT', 'LEFT']
    assert result['evasion_space'].tolist() == [0, 0]
    first = dict(only, entry_frame={'type': 'first_frame'})  # S145: first-frame entry on rerank
    adapter.validate_rule(first)
    assert set(adapter.changed_fields(first)) == {'collision_frame', 'entry_frame'}
    try:
        adapter.validate_rule(dict(only, entry_frame={'type': 'collision_minus', 'k': 3, 'source': 'final'}))
    except ValueError:
        pass
    else:
        raise AssertionError('collision-only rerank accepted a collision_minus entry')
    (folder / 'rule.json').write_text(json.dumps(first), encoding='utf-8')
    count_before = len(calls)
    result = module.predict(fake_ns, folder, None)
    assert len(calls) == count_before + 1
    assert fake_ns['_S118_LAST_DIAGNOSTICS']['rule_error'] is None
    assert result['collision_frame'].tolist() == [10, 2]
    assert result['entry_frame'].tolist() == [0, 0]
    assert result['entry_side'].tolist() == ['LEFT', 'LEFT']
    assert result['evasion_space'].tolist() == [0, 0]
    (folder / 'rule.json').write_text(json.dumps(passthrough), encoding='utf-8')
    module.load_package = lambda root, path: (_ for _ in ()).throw(FileNotFoundError(path))
    result = module.predict(fake_ns, folder, None)
    assert result.equals(base(folder, None))
    assert fake_ns['_S118_LAST_DIAGNOSTICS']['rerank_error'].startswith('FileNotFoundError')
    assert embedded['locate_collision'] is original
    return {'short_passthrough': True, 'track_reanchored': True, 'passthrough_entry': True,
            'first_frame_entry': True, 'missing_package_s109': True}


W4 = REPO / 'candidates/s120_ccd_loc'
W4_HEAD = W4 / 'weights/stage2_s120/head.pt'


def track_rule(rule):
    return bool(rule) and rule.get('entry_frame', {}).get('type') == 'track'


def rerank_rule(rule):
    return bool(rule) and rule.get('collision_frame', {}).get('type') == 'rerank'


def track_reference(candidate, rule, source, baseline, entry_ns):
    """S132: independent second tracker instance from the staged package bytes -> {ID: (entry, side)}."""
    spec = rule['entry_frame']
    module = load_file_module(candidate / spec['package'] / 'predict.py', '_s118_qa_track_reference')
    tracker = module.Tracker(candidate / spec['package'])
    out = {}
    try:
        for row in baseline.itertuples(index=False):
            paths = list(entry_ns['_s008_frame_paths'](source / 'images' / str(row.ID)))
            numbers = [int(entry_ns['_s008_frame_number'](p)) for p in paths]
            entry, side = int(row.entry_frame), row.entry_side
            if len(numbers) > spec['min_n']:
                c = min(range(len(numbers)), key=lambda i: (abs(numbers[i] - int(row.collision_frame)), i))
                position, new_side = module.decide(tracker.analyze(paths, numbers, c), spec['fallback'])
                if position is not None:
                    entry = numbers[min(max(int(position), 0), len(numbers) - 1)]
                if spec['side'] and new_side in ('LEFT', 'RIGHT'):
                    side = new_side
            out[str(row.ID)] = (min(int(entry), int(row.collision_frame)), side)
    finally:
        tracker.close()
    return out


def learned_rule(rule):
    return bool(rule) and rule.get('collision_frame', {}).get('type') == 'learned'


def learned_static_checks(candidate, rule):
    """Packaged S120 files equal W4's sources; W4's CV decode is the one shipped."""
    pkg = candidate / rule['collision_frame']['package']
    assert identity(pkg / 'head.pt') == identity(W4_HEAD), 'packaged head differs from W4 head'
    assert (pkg / 'features.py').read_bytes() == (W4 / 'features.py').read_bytes(), 'features.py differs from W4'
    train_src = (W4 / 'train.py').read_text(encoding='utf-8')
    head_src = (pkg / 'head_model.py').read_text(encoding='utf-8')
    for marker in ('class Head(nn.Module):', 'def decode(p, mode):', 'def build_input(d, feats: list[str]) -> np.ndarray:'):
        block = head_src[head_src.index(marker):].split(chr(10) * 3)[0].rstrip()
        assert block in train_src, 'head_model block not verbatim from W4 train.py: ' + marker
    assert 'T = 50' in head_src and 'LO, HI = 29, 49' in head_src
    assert not any('backbone' in q.name or q.suffix == '.safetensors' for q in pkg.rglob('*')), 'duplicate backbone shipped'
    return {'head_sha256': identity(W4_HEAD)['sha256'], 'features_verbatim': True, 'head_model_verbatim': True,
            'package_files': sorted(q.relative_to(pkg).as_posix() for q in pkg.rglob('*') if q.is_file())}


def learned_unit_checks(adapter, workdir):
    import pandas as pd
    ok = {'schema': 's118-rule-v1', 'collision_frame': {'type': 'learned', 'package': 'model/stage2/s120'}}
    adapter.validate_rule(ok)
    for bad in ('../x', 'model/stage2', 'model/stage2/../s1', '$LOCAL_ROOT/x', 3):
        try:
            adapter.validate_rule({'schema': 's118-rule-v1', 'collision_frame': {'type': 'learned', 'package': bad}})
        except ValueError:
            continue
        raise AssertionError('bad learned package accepted: %r' % (bad,))
    assert adapter.apply_rule(ok, list(range(100, 150)), 129, 110, learned_position=41) == (141, 110)
    assert adapter.apply_rule(ok, list(range(10)), 5, 1, learned_position=41) == (9, 1)
    try:
        adapter.apply_rule(ok, list(range(50)), 29, 21)
    except ValueError:
        pass
    else:
        raise AssertionError('learned rule without a position accepted')
    # A missing package fails closed to S109 for every clip.
    base = pd.DataFrame([{'ID': 'A', 'collision_frame': 35, 'entry_frame': 18, 'evasion_space': 1, 'entry_side': 'LEFT'}],
                        columns=OUTPUT_COLUMNS)
    folder = workdir / 'learned_missing'
    folder.mkdir()
    shutil.copyfile(adapter.__file__, folder / 'adapter.py')
    missing = dict(ok, collision_frame={'type': 'learned', 'package': 'model/stage2/s120_absent'})
    (folder / 'rule.json').write_text(json.dumps(missing), encoding='utf-8')
    module = load_file_module(folder / 'adapter.py', '_s118_qa_learned_missing')
    fake_ns = {'_S118_BASE_PREDICT_STAGE2': lambda d, m: base.copy()}
    assert module.predict(fake_ns, workdir, None).equals(base)
    assert fake_ns['_S118_LAST_DIAGNOSTICS']['rule_error'].startswith('learned load')
    return {'learned_rule_cases': 9, 'learned_fail_closed_missing_package': True}


def w4_reference(candidate, rule, source, report):
    """Independent recompute with W4's own predict.py/train.py over the same backbone."""
    import numpy as np
    import torch
    w4 = load_file_module(W4 / 'predict.py', '_s118_qa_w4_predict')
    assert Path(w4.fx.__file__).resolve() == (W4 / 'features.py').resolve()
    train = sys.modules['train']
    assert Path(train.__file__).resolve() == (W4 / 'train.py').resolve()
    positions, deployed, clip_paths = {}, {}, {}
    for folder in sorted((source / 'images').iterdir()):
        paths = w4.fx.frame_paths_of(folder)
        p = w4.collision_probs(paths, candidate / 'model', W4_HEAD)
        positions[folder.name] = int(train.decode(p[None], 'c_win3')[0])
        deployed[folder.name] = int(w4.predict_collision_positions(paths, candidate / 'model', W4_HEAD))
        clip_paths[folder.name] = (paths, p)
    w4._CACHE.clear()
    # Packaged localiser alone: warm per-clip time, CUDA peak and probability agreement with W4.
    pkg = load_file_module(candidate / rule['collision_frame']['package'] / 'predict.py', '_s118_qa_s120_pkg')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loc = pkg.Localiser(candidate / 'model')
    load_seconds = time.perf_counter() - start
    seconds, max_diff = [], 0.0
    for name, (paths, p) in clip_paths.items():
        start = time.perf_counter()
        q = loc.probs(paths)
        position = int(pkg.hm.decode(q[None], 'c_win3')[0])
        seconds.append(time.perf_counter() - start)
        max_diff = max(max_diff, float(np.abs(q - p).max()))
        assert position == positions[name], (name, position, positions[name])
    peak = torch.cuda.max_memory_allocated()
    loc.close()
    report['learned_reference'] = {
        'w4_c_win3_positions': positions, 'w4_predict_py_positions': deployed,
        'package_vs_w4_max_prob_diff': max_diff, 'package_load_seconds': load_seconds,
        'package_warm_seconds_per_clip': seconds, 'package_cuda_peak_bytes': peak}
    return positions


def stage1_unit_checks(adapter, workdir):
    import pandas as pd
    rule = {'schema': 's118-rule1-v1', 'encoder_signature':
            {'original_fourcc': 'FMP4', 'original_muxer': 'Lavf58.12.100', 'mode': 'override'}}
    adapter.validate_rule1(rule)
    cases = [
        ('FMP4', b'header Lavf58.12.100 end', 'ORIGINAL'),
        ('FMP4', b'Lavf58.12.100 x264 - core 165', 'RERECORDED'),
        ('FMP4', b'other muxer', 'RERECORDED'),
        ('h264', b'Lavf58.12.100', 'RERECORDED'),
        ('avc1', b'Lavf63.1.100 x264 - core 165', 'RERECORDED'),
    ]
    for fourcc, payload, expected in cases:
        assert adapter.classify_encoder_signature(fourcc, payload, rule) == expected
    for bad in ({'schema': 'wrong', 'encoder_signature': rule['encoder_signature']},
                {'schema': 's118-rule1-v1', 'encoder_signature': dict(rule['encoder_signature'], mode='append')},
                {'schema': 's118-rule1-v1', 'encoder_signature': dict(rule['encoder_signature'], original_fourcc='FM')}):
        try:
            adapter.validate_rule1(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid Stage1 rule accepted')
    videos = workdir / 'stage1_synthetic' / 'videos'
    videos.mkdir(parents=True)
    (videos / 'A.mp4').write_bytes(b'a')
    (videos / 'B.mp4').write_bytes(b'b')
    base = pd.DataFrame({'ID': ['A', 'B'], 'answer': ['RERECORDED', 'ORIGINAL'],
                         'probability': [0.8, 0.1]})
    original = adapter._file_signature
    def fake(path, _rule):
        if path.stem == 'B':
            raise OSError('synthetic read error')
        return 'ORIGINAL', {'fourcc': 'FMP4'}
    adapter._file_signature = fake
    try:
        diag = {}
        got = adapter.apply_stage1(base, videos.parent, rule, diag)
    finally:
        adapter._file_signature = original
    assert got['answer'].tolist() == ['ORIGINAL', 'ORIGINAL']
    assert got['probability'].tolist() == [0.0, 0.1]
    assert base['answer'].tolist() == ['RERECORDED', 'ORIGINAL']
    assert diag['B']['error'].startswith('OSError:') and not diag['B']['changed']
    return {'stage1_signature_cases': len(cases), 'stage1_invalid_rules': 3,
            'stage1_per_file_fallback': True}


def stage3_unit_checks(adapter, workdir):
    import pandas as pd
    base = pd.DataFrame({'ID': ['a', 'a', 'b'], 'sample_index': [0, 1, 0],
                         'accel_label': ['X', 'Y', 'Z'], 'steer_label': ['LEFT', 'RIGHT', 'STRAIGHT']})
    good, bad = workdir / 'rule3_good.json', workdir / 'rule3_bad.json'
    adapter.validate_rule3({'schema': 's118-rule3-v1', 'steer_regression_package': 'model/stage3/s141'})
    for invalid in ({'steer_regression_package': '../bad'},
                    {'steer_regression_package': 'model/stage3/s141', 'accel_log_bias': [0,0,0,0]}):
        try:
            adapter.validate_rule3(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid regression rule accepted')
    good.write_text('{"steer_label_const": "STRAIGHT"}', encoding='utf-8')
    bad.write_text('{"steer_label_const": "straight"}', encoding='utf-8')
    out = adapter.apply_stage3(base, good)
    assert out['steer_label'].tolist() == ['STRAIGHT'] * 3
    assert out.drop(columns='steer_label').equals(base.drop(columns='steer_label'))
    assert base['steer_label'].tolist() == ['LEFT', 'RIGHT', 'STRAIGHT'], 'input mutated'
    try:
        adapter.apply_stage3(base, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid Stage3 rule accepted')
    adapter.validate_rule3({'schema': 's118-rule3-v1', 'steer_video_mix':
                            {'package': 'model/stage3/s124', 'w': 0.25}})
    adapter.validate_rule3({'schema': 's118-rule3-v1', 'steer_video_mix':
                            {'package': 'model/stage3/s124', 'w': 0.25},
                            'accel_log_bias': [-0.25, -0.25, 0, -0.75]})
    for invalid in ([0, 0, 0], [0, 0, 0, float('nan')], [0, 0, 0, 'x']):
        try:
            adapter.validate_rule3({'accel_log_bias': invalid})
        except ValueError:
            pass
        else:
            raise AssertionError('invalid acceleration bias accepted')
    return {'stage3_rule_cases': 7}


def s124_static_checks(candidate, receipt):
    pkg = candidate / 'model/stage3/s124'
    source = HERE / 's124_pkg/predict.py'
    artifact = REPO / 'candidates/s124_vjepa_steer/artifacts/steer_head.pt'
    final = DATA / 's124/steer_head_final.pt'
    assert identity(pkg / 'predict.py') == identity(source), 'S124 package source differs'
    head = identity(pkg / 'steer_head.pt')
    assert head == identity(artifact), 'packaged S124 head differs from versioned artifact'
    import torch
    left = torch.load(pkg / 'steer_head.pt', map_location='cpu', weights_only=True)
    right = torch.load(final, map_location='cpu', weights_only=True)
    assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left), 'S124 final head tensors differ'
    assert receipt['added_files']['model/stage3/s124/steer_head.pt']['sha256'] == head['sha256']
    training = json.loads((REPO / 'candidates/s124_vjepa_steer/artifacts/final_training.json').read_text())
    assert training['arm'] == 'final' and training['n_train'] == 390
    return {'head': head, 'data_copy': identity(final), 'tensors_identical_to_data_copy': True,
            'source': identity(source), 'training':
            {'arm': training['arm'], 'n_train': training['n_train'], 'seed': training['seed']}}


def s130_static_checks(candidate, receipt):
    source = REPO / 'candidates/s130_s3label'
    result = {}
    for filename in ('predict.py', 'stage3_accel_robust.npz', 'head.pt'):
        name = 'model/stage3/s130/' + filename
        wanted = identity(source / filename)
        assert identity(candidate / name) == wanted, 'S130 staged file differs: ' + name
        assert receipt['added_files'][name]['sha256'] == wanted['sha256']
        result[filename] = wanted
    return result


def protected_member_check(candidate, receipt):
    """Every S109 member byte-identical; inference.py == S109 bytes + dispatch. Returns S109 inference bytes."""
    reference = None
    with zipfile.ZipFile(BASE) as z:
        infos = z.infolist()
        for info in infos:
            data = z.read(info)
            actual = (candidate / info.filename).read_bytes()
            if info.filename == 'inference.py':
                dispatch = chr(10).encode() * 2 + (HERE / 'dispatch.py').read_bytes()
                assert actual == data + dispatch, 'inference.py is not S109 + dispatch'
                assert hashlib.sha256(data).hexdigest() == receipt['base_inference']['sha256']
                reference = data
            else:
                assert actual == data, 'protected member differs: ' + info.filename
    return len(infos) - 1, reference


def load_reference(candidate, source_bytes):
    """Unmodified S109 inference.py source bound to the same (byte-identical) model tree."""
    module = types.ModuleType('_s118_s109_reference')
    module.__file__ = str(candidate / 'inference.py')
    exec(compile(source_bytes, str(candidate / 'inference.py'), 'exec'), module.__dict__)
    return module.__dict__


def log(message):
    print(time.strftime('%H:%M:%S'), message, file=sys.stderr, flush=True)


def timed(report, key, fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    report.setdefault('seconds', {})[key] = time.perf_counter() - start
    return value


def run(stage, output, skip_stage1=False):
    run_started = time.perf_counter()
    stage, output = Path(stage), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = read(stage / 'stage.json')
    candidate = stage / 'candidate'
    original_members = tree(candidate)
    if original_members != receipt['members'] or manifest_sha(original_members) != receipt['manifest_sha256']:
        raise ValueError('candidate differs from stage binding')
    rule_file, rule1_file, rule3_file = (candidate / MODEL_DIR / 'rule.json',
        candidate / 'model/stage1/s118/rule1.json', candidate / 'model/stage3/s118/rule3.json')
    rule = json.loads(rule_file.read_text(encoding='utf-8')) if rule_file.is_file() else None
    rule1 = json.loads(rule1_file.read_text(encoding='utf-8')) if rule1_file.is_file() else None
    rule3 = json.loads(rule3_file.read_text(encoding='utf-8')) if rule3_file.is_file() else None
    assert rule == receipt['rule'] and rule1 == receipt.get('rule1') and rule3 == receipt['rule3']
    stage2_prior_qa = stage3_prior_qa = None
    stage2_prior_name = stage3_prior_name = None
    if rule1 is not None:
        if rule.get('name') == 'S132_trk':
            stage2_prior_name = stage3_prior_name = 'S134b_trks3'
        elif rule.get('name') == 'S135_rrtrk':
            stage2_prior_name = 'S135_rrtrk'
            stage3_prior_name = 'S136_s1sig' if 'accel_package' in (rule3 or {}) else 'S135_rrtrk'
        else:
            raise ValueError('Stage1 rule has no approved Stage2/3 QA reference')
        stage2_prior_stage = read(DATA / 'stage2_s118' / stage2_prior_name / 'stage.json')
        stage3_prior_stage = read(DATA / 'stage2_s118' / stage3_prior_name / 'stage.json')
        stage2_prior_qa = read(DATA / 'stage2_s118' / (stage2_prior_name + '_qa') / 'qa.json')
        stage3_prior_qa = read(DATA / 'stage2_s118' / (stage3_prior_name + '_qa') / 'qa.json')
        assert stage2_prior_qa['passed'] is True and stage3_prior_qa['passed'] is True
        assert stage2_prior_stage['rule'] == rule and stage3_prior_stage['rule3'] == rule3
        expected_adds = {k: v for k, v in stage2_prior_stage['added_files'].items()
                         if not k.startswith('model/stage3/')}
        expected_adds.update({k: v for k, v in stage3_prior_stage['added_files'].items()
                              if k.startswith('model/stage3/')})
        assert expected_adds == receipt['added_files'], 'Stage2/3 added assets differ from approved references'
    samples = REPO / 'Baseline/sample_evaluation_data'
    s1, source, s3 = samples / 'stage1', samples / 'stage2', samples / 'stage3'
    input_identity = {name: tree(samples / name) for name in ('stage1', 'stage2', 'stage3')}
    if stage2_prior_qa is not None:
        assert stage2_prior_qa['inputs'] == input_identity and stage3_prior_qa['inputs'] == input_identity, 'reference QA inputs differ'
    import pandas as pd
    public_labels = pd.read_csv(REPO / 'Baseline/data/stage3/labels.csv')
    single = output / 'single'
    first = sorted((source / 'images').iterdir())[0]
    (single / 'images').mkdir(parents=True)
    shutil.copytree(first, single / 'images' / first.name)
    ipc = output / 'ipc'
    ipc.mkdir()
    unit_dir = output / 'unit'
    unit_dir.mkdir()
    tempfile.tempdir = str(ipc)
    os.environ.update(THREAD_ENV)
    os.environ['TMPDIR'] = str(ipc)
    multiprocessing.set_start_method('fork', force=True)
    report = {'schema': 's118-actual-qa-v2', 'passed': False, 'candidate_id': receipt['candidate_id'],
              'candidate_manifest_sha256': receipt['manifest_sha256'], 'mock_models': False,
              'rule': rule, 'rule1': rule1, 'rule3': rule3, 'source': {'bytes': len(_QA_SOURCE_BYTES), 'sha256': hashlib.sha256(_QA_SOURCE_BYTES).hexdigest()}, 'inputs': input_identity}
    try:
        if identity(BASE) != EXPECTED:
            raise ValueError('S109 base identity mismatch')
        log('protected member check')
        report['protected_members_byte_identical'], s109_source = protected_member_check(candidate, receipt)
        # Pure rule unit checks write scratch files, so they run before the read-only guard,
        # on the staged adapter bytes (the same file the dispatch loads).
        report['unit'] = {**unit_checks(load_file_module(candidate / MODEL_DIR / 'adapter.py', '_s118_qa_staged'), unit_dir),
                          **stage3_unit_checks(load_file_module(candidate / MODEL_DIR / 'adapter.py', '_s118_qa_staged3'), unit_dir)}
        if rule1 is not None:
            report['unit'].update(stage1_unit_checks(load_file_module(candidate / MODEL_DIR / 'adapter.py', '_s118_qa_staged1'), unit_dir))
        if learned_rule(rule):
            report['learned_static'] = learned_static_checks(candidate, rule)
            report['unit'].update(learned_unit_checks(load_file_module(candidate / MODEL_DIR / 'adapter.py', '_s118_qa_staged4'), unit_dir))
        if rule3 and 'steer_video_mix' in rule3:
            report['s124_identity'] = s124_static_checks(candidate, receipt)
        if rule3 and 'accel_package' in rule3:
            report['s130_identity'] = s130_static_checks(candidate, receipt)
        log('framework setup')
        environment = framework_setup(DATA / 'stage3_s107/vjepa/deps/site', gpu=True)
        report['environment'] = environment
        import torch
        original_loader = torch.utils.data.DataLoader
        class SingleProcessLoader(original_loader):
            def __init__(self, *args, **kwargs):
                kwargs['num_workers'] = 0
                kwargs['persistent_workers'] = False
                kwargs.pop('prefetch_factor', None)
                super().__init__(*args, **kwargs)
        torch.utils.data.DataLoader = SingleProcessLoader
        report['test_only_loader_workers'] = 0
        if receipt['candidate_id'].startswith(('S161', 'S137', 'S141', 'S142', 'S144', 'S145', 'S146', 'S147', 'S154', 'S156', 'S159')):
            import torch
            # Leave CUDA context/library headroom inside the shared 2 GiB budget.
            gpu_limit = int(1.75 * 1024 ** 3)
            torch.cuda.set_per_process_memory_fraction(gpu_limit / torch.cuda.get_device_properties(0).total_memory)
            report['qa_cuda_allocator_limit_bytes'] = gpu_limit
        sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != REPO]
        s156_root = DATA / 'stage2_s118/S156_casc/candidate'
        roots = [candidate, s156_root, samples, output, Path(sys.prefix), Path(sys.base_prefix)] + ([W4] if learned_rule(rule) else [])
        roots += [Path(p) for p in sys.path if p and Path(p).is_dir()]
        roots += [Path(p) for p in ('/proc', '/dev', '/sys', '/etc', '/usr')]
        import linecache
        source_name = str(Path(__file__).resolve())
        linecache.cache[source_name] = (len(_QA_SOURCE_BYTES), None,
            _QA_SOURCE_BYTES.decode('utf-8').splitlines(keepends=True), source_name)
        with linux_abstract_ipc(ipc), offline_readonly_guard(roots, strict=True, ipc_root=ipc) as guard:
            entry = load_entry(candidate / 'inference.py')
            ref = load_reference(candidate, s109_source)
            assert entry['_S118_BASE_PREDICT_STAGE3'].__code__.co_code == ref['predict_stage3'].__code__.co_code
            assert entry['_S118_BASE_PREDICT_STAGE1'].__code__.co_code == ref['predict_stage1'].__code__.co_code
            aliases = (entry['_s008_tensor'], entry['_s008_track_origin'], entry['_s008_predict_sides'])
            module = entry['_s118_adapter']()
            m1, m2, m3 = (candidate / 'model' / n for n in ('stage1', 'stage2', 'stage3'))

            log('entry + reference loaded')
            # Stage1 always runs the original S109 predictor before any per-file override.
            if skip_stage1:
                assert rule1 is None, 'Stage1 signature QA cannot be skipped'
                report['stage1'] = {'skipped': True, 'reason': 'function unchanged; members byte-identical'}
            else:
                ref1 = timed(report, 'stage1_s109', ref['predict_stage1'], s1, m1)
                act1 = timed(report, 'stage1_candidate', entry['predict_stage1'], s1, m1)
                assert list(act1.columns) == list(ref1.columns) == ['ID', 'answer']
                assert act1['ID'].tolist() == ref1['ID'].tolist() and len(act1) == 10
                if rule1 is None:
                    assert ref1.equals(act1), 'Stage1 outputs differ without a rule'
                else:
                    diag1 = copy.deepcopy(entry['_S118_STAGE1_DIAGNOSTICS'])
                    assert diag1['rule_error'] is None and len(diag1['videos']) == 10, diag1
                    assert all(not row.get('error') for row in diag1['videos'].values()), diag1
                    expected1 = {str(row.ID): ('ORIGINAL' if '_O_' in str(row.ID) else 'RERECORDED')
                                 for row in act1.itertuples(index=False)}
                    assert sum(v == 'ORIGINAL' for v in expected1.values()) == 5
                    assert sum(v == 'RERECORDED' for v in expected1.values()) == 5
                    assert dict(zip(act1['ID'], act1['answer'])) == expected1
                    report['stage1_signature_diagnostics'] = diag1
                report['stage1'] = {'rows': len(act1), 'identical_to_s109': bool(ref1.equals(act1)),
                                    'changed_rows': int((ref1['answer'] != act1['answer']).sum()),
                                    'columns': list(act1.columns), 'candidate': act1.to_dict(orient='records')}

            log('stage1 done')
            # S156 Stage1 is byte-identical to the S109 function checked above.
            if not skip_stage1:
                act1.to_csv(ipc / 's161_stage1.csv', index=False)
                ref1.to_csv(ipc / 's156_stage1.csv', index=False)
            if receipt['candidate_id'].startswith(('S161', 'S137', 'S141', 'S142', 'S144', 'S145', 'S146', 'S147', 'S154', 'S156', 'S159')):
                import gc
                import torch
                report['stage1_cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
                for namespace in (entry, ref):
                    namespace.get('_S071_PREDICTORS', {}).clear()
                gc.collect()
                torch.cuda.empty_cache()
            # Stage2
            baseline = timed(report, 'stage2_s109', ref['predict_stage2'], source, m2)
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            actual = timed(report, 'stage2_candidate', entry['predict_stage2'], source, m2)
            report['stage2_candidate_cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
            if receipt['candidate_id'] == 'S137d':
                ratio = report['seconds']['stage2_candidate'] / report['seconds']['stage2_s109']
                report['stage2_runtime_ratio'] = ratio
                assert ratio <= 1.10, 'S137d exceeds requested +10% Stage2 runtime: %.4f' % ratio

            report['diagnostics'] = copy.deepcopy(entry['_S118_LAST_DIAGNOSTICS'])
            learned_positions = w4_reference(candidate, rule, source, report) if learned_rule(rule) else {}
            reranked_base = (module._rerank_base(entry, source, m2, rule, {'clips': {}})
                             if rerank_rule(rule) else baseline)
            if rerank_rule(rule):
                assert not report['diagnostics'].get('rerank_error'), report['diagnostics']
                assert not report['diagnostics'].get('rerank_fallbacks'), report['diagnostics']
                assert not report['diagnostics'].get('s142_load_error'), report['diagnostics']
                assert not report['diagnostics'].get('s142_fallbacks'), report['diagnostics']
                assert not report['diagnostics'].get('s142_capture_errors'), report['diagnostics']
                if rule['collision_frame'].get('preserve_s109_fields'):
                    assert report['diagnostics'].get('single_base_call'), report['diagnostics']
                    assert not report['diagnostics'].get('capture_errors'), report['diagnostics']
                if rule['collision_frame'].get('reanchor_s109_fields'):  # S143
                    assert report['diagnostics'].get('single_base_call'), report['diagnostics']
                    reanchor = report['diagnostics'].get('reanchor', {})
                    assert not reanchor.get('errors') and not reanchor.get('side_error'), reanchor
            reranked_rows = {str(row.ID): row for row in reranked_base.itertuples(index=False)}
            tracked = track_reference(candidate, rule, source, reranked_base, entry) if track_rule(rule) else {}
            repeated = entry['predict_stage2'](single, m2)
            assert aliases == (entry['_s008_tensor'], entry['_s008_track_origin'], entry['_s008_predict_sides'])
            assert list(actual.columns) == OUTPUT_COLUMNS and list(baseline.columns) == OUTPUT_COLUMNS
            assert len(actual) == len(baseline) == len(list((source / 'images').iterdir()))
            assert actual['ID'].tolist() == baseline['ID'].tolist()
            for column in ('collision_frame', 'entry_frame', 'evasion_space'):
                assert str(actual[column].dtype).startswith('int'), (column, actual[column].dtype)
            s2_rule = rule or {'schema': 's118-rule-v1'}
            if rule is not None:
                assert report['diagnostics'].get('rule_error') is None, report['diagnostics']
            changed = module.changed_fields(s2_rule)
            protected = [c for c in OUTPUT_COLUMNS if c not in changed]
            assert baseline[protected].equals(actual[protected]), 'protected Stage2 outputs changed'
            diff = []
            for base_row, row in zip(baseline.itertuples(index=False), actual.itertuples(index=False)):
                numbers = [entry['_s008_frame_number'](p) for p in entry['_s008_frame_paths'](source / 'images' / str(row.ID))]
                anchor_row = reranked_rows[str(row.ID)]
                clip_rule = s2_rule
                if learned_rule(rule):
                    assert not report['diagnostics']['clips'][str(row.ID)].get('error'), report['diagnostics']['clips'][str(row.ID)]
                    clip_rule = dict(s2_rule, collision_frame={'type': 'const_position', 'p': learned_positions[str(row.ID)]})
                if track_rule(rule):
                    record = report['diagnostics']['clips'][str(row.ID)]
                    assert not record.get('error') and not record.get('skipped'), record
                    want = (int(anchor_row.collision_frame), tracked[str(row.ID)][0])
                    assert row.entry_side == tracked[str(row.ID)][1], (row.ID, tracked[str(row.ID)])
                elif rerank_rule(rule):
                    want_entry = (min(int(base_row.entry_frame), int(anchor_row.collision_frame))
                                  if rule['collision_frame'].get('preserve_s109_fields') else int(anchor_row.entry_frame))
                    if rule.get('entry_frame', {}).get('type') == 'first_frame':  # S145
                        want_entry = int(numbers[0])
                    want = (int(anchor_row.collision_frame), want_entry)
                    assert row.entry_side == anchor_row.entry_side
                else:
                    want = expected_values(clip_rule, numbers, int(base_row.collision_frame), int(base_row.entry_frame))
                assert (int(row.collision_frame), int(row.entry_frame)) == want, (row.ID, want)
                if rerank_rule(rule):
                    assert row.evasion_space == anchor_row.evasion_space, (row.ID, anchor_row)
                    if len(numbers) <= s2_rule['collision_frame']['min_n']:
                        assert anchor_row == base_row, 'short rerank path changed S109'
                ids = set(numbers)
                assert int(row.entry_frame) in ids and int(row.collision_frame) in ids
                assert row.entry_side in {'LEFT', 'RIGHT'} and row.evasion_space in {0, 1}
                diff.append({'ID': str(row.ID), 'frames': len(numbers),
                             's109_collision': int(base_row.collision_frame), 'collision': int(row.collision_frame),
                             's109_entry': int(base_row.entry_frame), 'entry': int(row.entry_frame),
                             'entry_before_collision': int(row.entry_frame) < int(row.collision_frame)})
            expected = actual[actual['ID'] == first.name].reset_index(drop=True)
            assert expected.equals(repeated.reset_index(drop=True)), 'input independence failed'
            report['stage2'] = {'s109': baseline.to_dict(orient='records'), 'candidate': actual.to_dict(orient='records'),
                                'single_repeat': repeated.to_dict(orient='records'), 'diff_vs_s109': diff,
                                'changed_rows': {'collision_frame': sum(d['s109_collision'] != d['collision'] for d in diff),
                                                 'entry_frame': sum(d['s109_entry'] != d['entry'] for d in diff)}}

            if stage2_prior_qa is not None:
                assert report['stage2']['candidate'] == stage2_prior_qa['stage2']['candidate'], 'Stage2 differs from ' + stage2_prior_name
                report['stage2_reference_identical'] = stage2_prior_name
            log('stage2 done')
            assert actual.drop(columns='entry_frame').equals(reranked_base.drop(columns='entry_frame'))
            actual.to_csv(ipc / 's161_stage2.csv', index=False)
            reranked_base.to_csv(ipc / 's156_stage2.csv', index=False)
            # Stage3
            ref3 = timed(report, 'stage3_s109', ref['predict_stage3'], s3, m3)
            if rule3 and 'accel_log_bias' in rule3:
                assert entry['ACCEL'] == ['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED']
                assert tuple(entry['_S019_ACCEL_LOG_BIAS']) == (-0.25, -0.5, 0.0, -0.75)
                equal3 = timed(report, 'stage3_equal_bias', module.predict_stage3_bias,
                               entry, s3, m3, {'accel_log_bias': list(entry['_S019_ACCEL_LOG_BIAS'])})
                assert ref3.equals(equal3), 'equal acceleration bias differs from S109 output'
                report['stage3_equal_bias_byte_identical'] = True
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            act3 = timed(report, 'stage3_candidate', entry['predict_stage3'], s3, m3)
            report['stage3_candidate_cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
            report['stage3_runtime_delta_seconds'] = report['seconds']['stage3_candidate'] - report['seconds']['stage3_s109']
            assert list(act3.columns) == list(ref3.columns) and len(act3) == len(ref3)
            changed3 = set()
            if rule3 and ('steer_label_const' in rule3 or 'steer_video_mix' in rule3 or 'steer_regression_package' in rule3):
                changed3.add('steer_label')
            if rule3 and ('accel_log_bias' in rule3 or 'accel_package' in rule3):
                changed3.add('accel_label')
            keep3 = [c for c in ref3.columns if c not in changed3]
            assert ref3[keep3].equals(act3[keep3]), 'Stage3 unselected outputs/order changed'
            if rule3 is None:
                assert ref3.equals(act3), 'Stage3 changed without a rule'
            elif 'steer_label_const' in rule3:
                assert set(act3['steer_label']) == {rule3['steer_label_const']}
            else:
                diag3 = copy.deepcopy(entry['_S118_STAGE3_DIAGNOSTICS'])
                assert diag3['rule_error'] is None, diag3
                if 'steer_video_mix' in rule3:
                    assert diag3['w0_parity_with_s109'], diag3
                    assert len(diag3['videos']) == int(ref3['ID'].nunique())
                    assert all(v.get('mixed') for v in diag3['videos'].values()), diag3
                if 'steer_regression_package' in rule3:
                    assert len(diag3['videos']) == int(ref3['ID'].nunique()), diag3
                    assert all(v.get('regressed') for v in diag3['videos'].values()), diag3
                report['stage3_mix_diagnostics'] = diag3
            if rule3 and 'accel_log_bias' not in rule3 and 'accel_package' not in rule3:
                assert ref3['accel_label'].equals(act3['accel_label'])
            if rule3 and 'steer_video_mix' not in rule3 and 'steer_label_const' not in rule3 and 'steer_regression_package' not in rule3:
                assert ref3['steer_label'].equals(act3['steer_label'])
            order = entry['ACCEL']
            share = lambda series: {k: float((series == k).mean()) for k in order}
            labeled = ref3.merge(public_labels[['ID', 'sample_index', 'accel_label']],
                                 on=['ID', 'sample_index'], how='inner', suffixes=('_s109', '_truth'))
            assert len(labeled) == len(public_labels) == 50
            report['stage3_public_accel'] = {'s109_all_rows_share': share(ref3['accel_label']),
                's109_sparse_rows_share': share(labeled['accel_label_s109']),
                'sparse_truth_share': share(labeled['accel_label_truth']),
                'sparse_rows': len(labeled),
                's109_sparse_correct': int((labeled['accel_label_s109'] == labeled['accel_label_truth']).sum()),
                'candidate_all_rows_share': share(act3['accel_label'])}
            report['stage3'] = {'rows': len(act3), 'videos': int(act3['ID'].nunique()),
                                'columns': list(act3.columns),
                                'steer_counts_s109': {str(k): int(v) for k, v in ref3['steer_label'].value_counts().items()},
                                'steer_counts_candidate': {str(k): int(v) for k, v in act3['steer_label'].value_counts().items()},
                                'steer_rows_changed': int((ref3['steer_label'] != act3['steer_label']).sum()),
                                'accel_identical': bool(ref3['accel_label'].equals(act3['accel_label'])),
                                'accel_rows_changed': int((ref3['accel_label'] != act3['accel_label']).sum()),
                                'accel_counts_s109': {str(k): int(v) for k, v in ref3['accel_label'].value_counts().items()},
                                'accel_counts_candidate': {str(k): int(v) for k, v in act3['accel_label'].value_counts().items()}}
            if stage3_prior_qa is not None:
                assert report['stage3'] == stage3_prior_qa['stage3'], 'Stage3 differs from ' + stage3_prior_name
                report['stage3_reference_identical'] = stage3_prior_name
            log('stage3 done')
            # Execute S156's unchanged Stage3 adapter against the already-loaded
            # exact base namespace and shared model bytes, then diff full CSVs.
            s156_adapter = load_file_module(s156_root / MODEL_DIR / 'adapter.py', '_s161_s156_stage3')
            s156_3 = timed(report, 'stage3_s156', s156_adapter.predict_stage3_regression,
                          entry, s3, m3, rule3)
            assert s156_3.equals(act3), 'Stage3 differs from S156'
            act3.to_csv(ipc / 's161_stage3.csv', index=False)
            s156_3.to_csv(ipc / 's156_stage3.csv', index=False)
            report['s156_output_parity'] = {'stage1': True, 'stage2_except_entry': True, 'stage3': True}
            report['checks'] = {'stage1_identical_to_s109_reference': bool(report['stage1'].get('identical_to_s109')),
                                'stage2_protected_fields': True, 'stage2_rule_values_match_independent_derivation': True,
                                'stage3_only_selected_fields_changed_rows_order_identical': True,
                                'independent_single_after_batch': True, 'original_filename_ids': True,
                                'aliases_restored': True, 'unregistered_loader': True, 'actual_linux_cuda': True}
        report['guard'] = {k: v for k, v in guard.items() if k != 'active'}
        assert not guard['network_attempts'] and not guard['filesystem_attempts']
        assert tree(candidate) == original_members
        assert {name: tree(samples / name) for name in ('stage1', 'stage2', 'stage3')} == input_identity
        try:
            import torch
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        except Exception:
            report['cuda_peak_allocated_bytes'] = None
        report['host_memory'] = host_memory()
        report['passed'] = True
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        if 'guard' in locals():
            report['guard'] = {k: v for k, v in guard.items() if k != 'active'}
        import traceback
        report['traceback'] = traceback.format_exc()
    report['wall_seconds'] = time.perf_counter() - run_started
    save_new(output / 'qa.json', report)
    summary = {k: report[k] for k in ('passed', 'error', 'seconds') if k in report}
    if 'stage1' in report:
        summary['stage1'] = {k: v for k, v in report['stage1'].items() if k != 'candidate'}
    if 'stage2' in report:
        summary['stage2'] = {'changed_rows': report['stage2']['changed_rows'], 'diff_vs_s109': report['stage2']['diff_vs_s109']}
    if 'stage3' in report:
        summary['stage3'] = {k: v for k, v in report['stage3'].items() if k != 'columns'}
    summary['unit'] = report.get('unit')
    print(json.dumps(summary, indent=1), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-stage1', action='store_true')
    args = parser.parse_args()
    result = run(args.stage, args.output, args.skip_stage1)
    raise SystemExit(0 if result['passed'] else 1)
