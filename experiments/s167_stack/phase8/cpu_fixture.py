"""Rule-gate fixtures for the no-S176 S185 extension."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

source = Path('$DATA_DIR/s167_stack/phase8/candidate/model/stage2/s185_predict.py')
spec = importlib.util.spec_from_file_location('_s185_cpu_fixture', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.Tracker.analyze is module._s185_analyze
assert not hasattr(module, '_s176_analyze')
tracker = object.__new__(module.Tracker)
tracker.params = SimpleNamespace(long_n=300)
numbers = list(range(30))
paths = [str(i) for i in numbers]


def crossing(side):
    return module.Result(reason='crossing', entry_index=7, collision_index=20,
                         bracket=(6, 7), fps=10.0, inside_samples=2,
                         outside_samples=2, track_len=4, side=side)


def check(original, final, completed):
    def prior(self, _paths, _numbers, _collision_index):
        self._s185_s161_result = original
        self._s185_s174_completed = completed
        return final
    module._s185_previous_analyze = prior
    return module._s185_analyze(tracker, paths, numbers, 20)


result = check(crossing('LEFT'), crossing('LEFT'), False)
assert result['s185_source'] == 'S161' and result['s185_side'] == 'LEFT'
assert 's185_gap_collision_index' not in result

result = check(module.Result(reason='short_track'), crossing('RIGHT'), True)
assert result['s185_source'] == 'S174' and result['s185_side'] is None
assert 's185_gap_collision_index' not in result

for reason in ('short_track', 'not_in_lane_at_collision',
               'far_inside_when_first_seen'):
    result = check(module.Result(reason=reason), module.Result(reason=reason), True)
    assert result['s185_source'] == 'S109' and result['s185_gap_collision_index'] == 20
    assert result['s185_gap_fps'] == 10.0

result = check(module.Result(reason='short_track'),
               module.Result(reason='short_track'), False)
assert 's185_gap_collision_index' not in result

for reason in ('inside_from_start', 'inside_when_first_seen', 'no_anchor',
               'far_crossing', 'frame_budget'):
    result = check(module.Result(reason=reason), module.Result(reason=reason), True)
    assert result['s185_source'] == 'S109'
    assert result['s185_side'] is None
    assert 's185_gap_collision_index' not in result

print('S185_NO_BUCKET_CPU_FIXTURE_PASS')
