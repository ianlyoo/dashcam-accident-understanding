"""CPU rule fixtures for the exact S181 ZIP's chained decisions."""
import importlib.util
from pathlib import Path

SOURCE = Path('$DATA_DIR/s167_stack/phase6/candidate/model/stage2/s181_predict.py')
spec = importlib.util.spec_from_file_location('_s181_fixture', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.Tracker.analyze is module._s181_analyze
assert module._s181_previous_analyze is module._s178_analyze

# An accepted original S161 crossing keeps its position; the S178 side result
# is independent of the entry decision.
crossing = module.Result(reason='crossing', entry_index=7, collision_index=20,
                         bracket=(6, 7), fps=10.0, inside_samples=2,
                         outside_samples=2, track_len=4, side='LEFT')
assert module._s180_previous_decide(crossing, 's109')[0] == 7
assert module.decide(crossing, 's109')[0] == 7

# S174-only crossing has no S178 side and receives exactly +0.2 seconds.
crossing['side'] = None
assert module.decide(crossing, 's109')[0] == 9
crossing['collision_index'] = 8
assert module.decide(crossing, 's109')[0] == 8

# S181 gap is legal only with a completed fallback marker; here the marker
# appears only after its analyze wrapper, so decide sees it in the result.
gap = module.Result(reason='short_track', s181_collision_index=20, s181_fps=10.0)
assert module._s181_previous_decide(gap, 's109')[0] is None
assert module.decide(gap, 's109')[0] == 12
assert module.decide(module.Result(reason='short_track'), 's109')[0] is None
assert module.decide(module.Result(reason='no_anchor', s181_collision_index=20,
                                   s181_fps=10.0), 's109')[0] is None
assert module.decide(module.Result(reason='s176_inside_from_start',
                                   entry_index=0, collision_index=20), 's109')[0] == 0
print('S181_CHAIN_CPU_FIXTURE_PASS')
