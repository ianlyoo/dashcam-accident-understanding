"""S118 rule-driven Stage2 leaf override around the unchanged S109 predictor.

S109 predict_stage2 runs exactly as released. Afterwards each output row may
have collision_frame and/or entry_frame replaced according to rule.json, which
sits next to this file (model/stage2/s118/rule.json). All other fields, and
any field the rule does not mention, pass through byte-for-byte.

Positions are 0-based indices into the clip's frame files sorted by trailing
filename integer (the same order S109 uses for numbers[index]); every output
is mapped back to that clip's actual filename frame number. Positions are
clipped to [0, n-1] so any frame count works.

Rule schema (s118-rule-v1):
  collision_frame: {"type": "passthrough"}
                 | {"type": "const_position", "p": int}
                 | {"type": "const_fraction", "f": float in [0, 1]}  (round(f*(n-1)))
                 | {"type": "clamp_position", "lo": int, "hi": int}   (clamps S109's position)
                 | {"type": "floor_replace", "lo": int, "c": int}      (S109 position if >= lo else c)
                 | {"type": "learned", "package": "model/stage2/<pkg>"}
                   (position from <pkg>/predict.py Localiser(model_root).position(frame_paths);
                    the package is loaded once per predict call and released afterwards)
  entry_frame:     {"type": "passthrough"}
                 | {"type": "collision_minus", "k": int >= 0, "source": "final" | "s109"}
                 | {"type": "collision_minus_grid", "k": int >= 0, "source": "final" | "s109"}
                   (k counts 10 Hz grid frames: the position offset is round(k * n / 50)
                    for n > 50 frames, i.e. a fixed time lead on a 5 s clip, and k otherwise)
Omitted fields are passthrough.

Optional Stage3 rule (model/stage3/s118/rule3.json, schema s118-rule3-v1):
  {"steer_label_const": "LEFT" | "STRAIGHT" | "RIGHT"} sets steer_label on
  every row of the full S109 predict_stage3 output. Without the file the
  dispatch returns S109's Stage3 result untouched.

Fail-closed: an invalid/missing rule keeps every S109 row; any exception on a
clip keeps that clip's S109 values. No state is shared across clips.
"""
import json
from pathlib import Path

RULE_SCHEMA = 's118-rule-v1'
RULE_KEYS = {'schema', 'name', 'note', 'collision_frame', 'entry_frame'}
COLLISION_TYPES = {'passthrough': set(), 'const_position': {'p'}, 'const_fraction': {'f'},
                   'clamp_position': {'lo', 'hi'}, 'floor_replace': {'lo', 'c'}, 'learned': {'package'},
                   'position_prior': {'min_n', 'lo_frac', 'hi_frac', 'center_frac', 'sigma_frac'},
                   'rerank': {'package', 'min_n'}}
ENTRY_TYPES = {'passthrough': set(), 'collision_minus': {'k', 'source'}, 'collision_minus_grid': {'k', 'source'},
               'motion_lead_scale': {'min_n', 'scale'}, 'first_frame': set(),
               'track': {'package', 'fallback', 'side', 'min_n', 'budget_seconds'}}
ENTRY_SOURCES = {'final', 's109'}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


RULE1_SCHEMA = 's118-rule1-v1'


def validate_rule1(rule):
    """Validate the optional per-file encoder signature override."""
    if not isinstance(rule, dict) or rule.get('schema') != RULE1_SCHEMA or set(rule) - {'schema', 'name', 'note', 'encoder_signature'}:
        raise ValueError('bad Stage1 rule schema or keys')
    spec = rule.get('encoder_signature')
    if not isinstance(spec, dict) or set(spec) != {'original_fourcc', 'original_muxer', 'mode'}:
        raise ValueError('bad encoder_signature keys')
    if spec['mode'] != 'override':
        raise ValueError('encoder_signature mode must be override')
    fourcc, muxer = spec['original_fourcc'], spec['original_muxer']
    if not isinstance(fourcc, str) or len(fourcc) != 4 or not fourcc.isascii():
        raise ValueError('original_fourcc must be four ASCII characters')
    if not isinstance(muxer, str) or not muxer or len(muxer) > 64 or not muxer.isascii():
        raise ValueError('original_muxer must be 1..64 ASCII characters')
    return rule


def load_rule1(path):
    return validate_rule1(json.loads(Path(path).read_text(encoding='utf-8')))


def classify_encoder_signature(fourcc, payload, rule):
    """Pure classifier also used for synthetic byte-string QA."""
    spec = validate_rule1(rule)['encoder_signature']
    if not isinstance(payload, bytes):
        raise TypeError('payload must be bytes')
    original = (fourcc == spec['original_fourcc']
                and spec['original_muxer'].encode('ascii') in payload
                and b'x264 - core' not in payload)
    return 'ORIGINAL' if original else 'RERECORDED'


def _file_signature(path, rule):
    """Scan all bytes in bounded chunks, retaining overlap for split markers."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError('cannot open video for FOURCC')
        code = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join(chr((code >> (8 * i)) & 255) for i in range(4))
    finally:
        capture.release()
    muxer = rule['encoder_signature']['original_muxer'].encode('ascii')
    x264 = b'x264 - core'
    found_muxer = found_x264 = False
    overlap = max(len(muxer), len(x264)) - 1
    carry = b''
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            scan = carry + chunk
            found_muxer |= muxer in scan
            found_x264 |= x264 in scan
            carry = scan[-overlap:] if overlap else b''
    original = fourcc == rule['encoder_signature']['original_fourcc'] and found_muxer and not found_x264
    return ('ORIGINAL' if original else 'RERECORDED',
            {'fourcc': fourcc, 'original_muxer_found': bool(found_muxer), 'x264_sei_found': bool(found_x264)})


def apply_stage1(base, data_dir, rule, diagnostics=None):
    """Override each readable video's answer; keep S109 on per-file errors."""
    validate_rule1(rule)
    if 'ID' not in base.columns or 'answer' not in base.columns:
        raise ValueError('S109 Stage1 output lacks ID/answer')
    out = base.copy()
    root = Path(data_dir) / 'videos'
    paths = {}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.3gp', '.3gpp', '.wmv'}:
            paths[path.stem] = path if path.stem not in paths else None
    for index, row in out.iterrows():
        key = str(row['ID'])
        record = {}
        try:
            path = paths.get(key)
            if path is None:
                raise FileNotFoundError('Stage1 video missing or ID ambiguous')
            answer, signature = _file_signature(path, rule)
            record.update(signature)
            record['answer'] = answer
            record['changed'] = bool(answer != row['answer'])
            out.at[index, 'answer'] = answer
            if record['changed']:
                for column in ('probability', 'rerecorded_probability'):
                    if column in out.columns:
                        out.at[index, column] = float(answer == 'RERECORDED')
        except Exception as exc:
            record['error'] = '%s: %s' % (type(exc).__name__, exc)
            record['answer'] = row['answer']
            record['changed'] = False
        if diagnostics is not None:
            diagnostics[key] = record
    return out


def validate_rule(rule):
    if not isinstance(rule, dict) or rule.get('schema') != RULE_SCHEMA:
        raise ValueError('rule schema must be ' + RULE_SCHEMA)
    extra = set(rule) - RULE_KEYS
    if extra:
        raise ValueError('unknown rule keys: ' + ', '.join(sorted(extra)))
    for field, types in (('collision_frame', COLLISION_TYPES), ('entry_frame', ENTRY_TYPES)):
        spec = rule.get(field, {'type': 'passthrough'})
        if not isinstance(spec, dict) or spec.get('type') not in types:
            raise ValueError('bad %s rule type' % field)
        optional = ({'preserve_s109_fields', 'reanchor_s109_fields'}
                    if field == 'collision_frame' and spec['type'] == 'rerank' else set())
        if 'preserve_s109_fields' in spec and spec['preserve_s109_fields'] is not True:
            raise ValueError('preserve_s109_fields must be true')
        if 'reanchor_s109_fields' in spec and spec['reanchor_s109_fields'] is not True:
            raise ValueError('reanchor_s109_fields must be true')
        if 'preserve_s109_fields' in spec and 'reanchor_s109_fields' in spec:
            raise ValueError('preserve and reanchor S109 fields are exclusive')
        if set(spec) - {'type'} - optional != types[spec['type']]:
            raise ValueError('bad %s rule parameters' % field)
    collision = rule.get('collision_frame', {'type': 'passthrough'})
    kind = collision['type']
    if kind == 'const_position' and not _is_int(collision['p']):
        raise ValueError('p must be an integer')
    if kind == 'const_fraction':
        f = collision['f']
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0.0 <= float(f) <= 1.0:
            raise ValueError('f must be a number in [0, 1]')
    if kind == 'clamp_position':
        if not (_is_int(collision['lo']) and _is_int(collision['hi'])) or collision['lo'] > collision['hi']:
            raise ValueError('lo/hi must be integers with lo <= hi')
    if kind == 'floor_replace' and not (_is_int(collision['lo']) and _is_int(collision['c'])):
        raise ValueError('lo/c must be integers')
    if kind in ('learned', 'rerank'):
        package = collision['package']
        parts = package.split('/') if isinstance(package, str) else []
        if len(parts) < 3 or parts[0] != 'model' or any(x in ('', '.', '..') or chr(92) in x or ':' in x for x in parts):
            raise ValueError('package must be a relative path under model/')
    if kind == 'rerank' and (not _is_int(collision['min_n']) or collision['min_n'] < 50):
        raise ValueError('rerank min_n must be an integer >= 50')
    if kind == 'position_prior':
        fracs = [collision[k] for k in ('lo_frac', 'hi_frac', 'center_frac', 'sigma_frac')]
        if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0.0 <= float(f) <= 1.0 for f in fracs):
            raise ValueError('position_prior fractions must be numbers in [0, 1]')
        if not collision['lo_frac'] < collision['hi_frac']:
            raise ValueError('lo_frac must be below hi_frac')
        if not _is_int(collision['min_n']) or collision['min_n'] < 50:
            raise ValueError('min_n must be an integer >= 50')
        if rule.get('entry_frame', {'type': 'passthrough'})['type'] != 'passthrough':
            raise ValueError('position_prior re-anchors S109 entry itself; entry rule must be passthrough')
    entry = rule.get('entry_frame', {'type': 'passthrough'})
    if collision.get('preserve_s109_fields'):
        if entry['type'] not in ('passthrough', 'first_frame', 'track'):
            raise ValueError('collision-only rerank requires passthrough, first_frame or entry-only track')
        if entry['type'] == 'track' and entry['side']:
            raise ValueError('collision-only rerank requires track side=false')
    if collision.get('reanchor_s109_fields') and entry['type'] != 'passthrough':
        raise ValueError('S143 re-anchored rerank requires passthrough entry')
    if entry['type'] == 'motion_lead_scale':
        scale = entry['scale']
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 1.0 <= float(scale) <= 10.0:
            raise ValueError('scale must be a number in [1, 10]')
        if not _is_int(entry['min_n']) or entry['min_n'] < 50:
            raise ValueError('min_n must be an integer >= 50')
        if kind != 'passthrough':
            raise ValueError('motion_lead_scale needs a passthrough collision rule')
    if entry['type'] == 'track':  # S132 participant tracking; see _predict_track
        package = entry['package']
        parts = package.split('/') if isinstance(package, str) else []
        if len(parts) < 3 or parts[0] != 'model' or any(x in ('', '.', '..') or chr(92) in x or ':' in x for x in parts):
            raise ValueError('package must be a relative path under model/')
        if entry['fallback'] not in ('s109', 'first_frame') or not isinstance(entry['side'], bool):
            raise ValueError('track fallback must be s109|first_frame and side a bool')
        if not _is_int(entry['min_n']) or entry['min_n'] < 0:
            raise ValueError('min_n must be a non-negative integer')
        budget = entry['budget_seconds']
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < float(budget) <= 3600:
            raise ValueError('budget_seconds must be in (0, 3600]')
        if kind not in ('passthrough', 'rerank'):
            raise ValueError('track needs a passthrough or rerank collision rule')
    if entry['type'] in ('collision_minus', 'collision_minus_grid'):
        if not _is_int(entry['k']) or entry['k'] < 0:
            raise ValueError('k must be a non-negative integer')
        if entry['source'] not in ENTRY_SOURCES:
            raise ValueError('source must be final or s109')
    return rule


def load_rule(path):
    return validate_rule(json.loads(Path(path).read_text(encoding='utf-8')))


def changed_fields(rule):
    fields = [f for f in ('collision_frame', 'entry_frame')
              if rule.get(f, {'type': 'passthrough'})['type'] != 'passthrough']
    if rule.get('collision_frame', {}).get('type') == 'rerank':
        # S109 computes these fields after locate_collision inside the same base call.
        downstream = ('entry_frame',) if rule['collision_frame'].get('preserve_s109_fields') else ('entry_frame', 'entry_side', 'evasion_space')
        fields += [f for f in downstream if f not in fields]
    entry = rule.get('entry_frame', {'type': 'passthrough'})
    if entry['type'] == 'track' and entry['side'] and 'entry_side' not in fields:
        fields.append('entry_side')
    return fields


def _clip(position, n):
    return max(0, min(int(position), n - 1))


def position_of(numbers, value):
    """Index of a filename frame number; nearest (earliest on ties) if absent."""
    value = int(value)
    try:
        return numbers.index(value)
    except ValueError:
        return min(range(len(numbers)), key=lambda i: (abs(numbers[i] - value), i))


def apply_rule(rule, numbers, base_collision, base_entry, learned_position=None):
    """Pure per-clip rule: sorted filename numbers + S109 values -> new values."""
    numbers = [int(n) for n in numbers]
    n = len(numbers)
    if n == 0:
        raise ValueError('clip has no frames')
    collision_spec = rule.get('collision_frame', {'type': 'passthrough'})
    entry_spec = rule.get('entry_frame', {'type': 'passthrough'})
    kind = collision_spec['type']
    base_position = None
    if kind != 'passthrough' or entry_spec['type'] != 'passthrough':
        base_position = position_of(numbers, base_collision)
    if kind in ('passthrough', 'rerank'):
        collision, final_position = int(base_collision), base_position
    else:
        if kind == 'const_position':
            final_position = _clip(collision_spec['p'], n)
        elif kind == 'const_fraction':
            final_position = _clip(round(float(collision_spec['f']) * (n - 1)), n)
        elif kind == 'learned':
            if learned_position is None:
                raise ValueError('learned rule needs a model position')
            final_position = _clip(learned_position, n)
        elif kind == 'floor_replace':
            lo = _clip(collision_spec['lo'], n)
            final_position = base_position if base_position >= lo else _clip(collision_spec['c'], n)
        else:
            hi = _clip(collision_spec['hi'], n)
            lo = min(_clip(collision_spec['lo'], n), hi)
            final_position = max(lo, min(base_position, hi))
        collision = numbers[final_position]
    if entry_spec['type'] == 'passthrough':
        entry = int(base_entry)
    elif entry_spec['type'] == 'first_frame':
        entry = numbers[0]
    else:
        anchor = final_position if entry_spec['source'] == 'final' else base_position
        lead = int(entry_spec['k'])
        if entry_spec['type'] == 'collision_minus_grid' and n > 50:
            lead = int(round(lead * n / 50.0))
        entry = numbers[max(0, anchor - lead)]
    return int(collision), int(entry)


RULE3_KEYS = {'schema', 'name', 'note', 'steer_label_const', 'steer_video_mix', 'accel_log_bias', 'accel_package', 'steer_regression_package'}
STEER_LABELS = {'LEFT', 'STRAIGHT', 'RIGHT'}


def validate_rule3(rule):
    if not isinstance(rule, dict) or set(rule) - RULE3_KEYS:
        raise ValueError('bad Stage3 rule keys')
    if rule.get('schema', 's118-rule3-v1') != 's118-rule3-v1':
        raise ValueError('Stage3 rule schema must be s118-rule3-v1')
    if 'steer_regression_package' in rule:
        if rule['steer_regression_package'] != 'model/stage3/s141':
            raise ValueError('unexpected regression package')
        if set(rule) & {'steer_label_const', 'steer_video_mix', 'accel_log_bias', 'accel_package'}:
            raise ValueError('regression rule changes steering alone')
    const, mix = rule.get('steer_label_const'), rule.get('steer_video_mix')
    bias = rule.get('accel_log_bias')
    if const is not None and mix is not None:
        raise ValueError('constant and video steering cannot be combined')
    if const is None and mix is None and bias is None and 'accel_package' not in rule and 'steer_regression_package' not in rule:
        raise ValueError('Stage3 rule changes no field')
    if const is not None and const not in STEER_LABELS:
        raise ValueError('steer_label_const must be one of LEFT/STRAIGHT/RIGHT')
    if mix is not None:
        if not isinstance(mix, dict) or set(mix) != {'package', 'w'}:
            raise ValueError('bad steer_video_mix keys')
        if mix['package'] != 'model/stage3/s124':
            raise ValueError('unexpected steering package')
        if type(mix['w']) not in (int, float) or not 0 <= mix['w'] <= 1:
            raise ValueError('steering mixture weight must be in [0,1]')
    if bias is not None:
        import math
        if (not isinstance(bias, list) or len(bias) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bias)):
            raise ValueError('accel_log_bias must be four finite numbers in ACC/DEC/CONST/STOP order')
    if 'accel_package' in rule and rule['accel_package'] != 'model/stage3/s130':
        raise ValueError('unexpected acceleration package')
    if 'accel_package' in rule and mix is None:
        raise ValueError('S130 acceleration requires the shared S124/S109 Stage3 pass')
    return rule


def predict_stage3_regression(ns, data_dir, model_dir, rule):
    root = Path(__file__).resolve().parents[3]
    try:
        package = load_package(root, rule['steer_regression_package'])
    except Exception as exc:
        ns['_S118_STAGE3_DIAGNOSTICS'] = {'rule_error': str(exc), 'videos': {}}
        return ns['_S118_BASE_PREDICT_STAGE3'](data_dir, model_dir)
    return package.predict(ns, data_dir, model_dir, root / rule['steer_regression_package'])


def load_rule3(path):
    return validate_rule3(json.loads(Path(path).read_text(encoding='utf-8')))


def apply_stage3(base, rule_path):
    """Set steer_label to a constant on every row; rows, order and accel_label untouched."""
    rule = load_rule3(rule_path)
    out = base.copy()
    if 'steer_label_const' in rule:
        out['steer_label'] = [rule['steer_label_const']] * len(out)
    return out


def _install_accel_bias(ns, s109, bias, active, records):
    """Patch only S109's final 75/25 decoder, while its call holds _S108_LOCK."""
    import numpy as np
    if tuple(ns['ACCEL']) != ('ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED'):
        raise ValueError('S109 acceleration class order differs')
    if tuple(ns['_S019_ACCEL_LOG_BIAS']) != tuple(s109.EXPECTED_BIAS):
        raise ValueError('S109 acceleration bias differs')
    original = s109.combine_probabilities

    def combine(base, video, smooth, window, incumbent_bias):
        incumbent = original(base, video, smooth, window, incumbent_bias)
        if tuple(bias) == tuple(incumbent_bias):
            result = incumbent
        else:
            try:
                mixture = 0.75 * np.asarray(base, dtype=np.float64) + 0.25 * np.asarray(video, dtype=np.float64)
                smoothed = smooth(mixture, window)
                if smoothed.shape != mixture.shape or not np.isfinite(smoothed).all():
                    raise ValueError('invalid candidate smoothed probabilities')
                result = (np.log(np.maximum(smoothed, 1e-9)) + np.asarray(bias, dtype=np.float64)).argmax(1).astype(np.int64)
            except Exception as exc:
                result = incumbent
                records.setdefault(active[0], {})['accel_error'] = '%s: %s' % (type(exc).__name__, exc)
        records.setdefault(active[0], {}).update(accel_s109=incumbent, accel_candidate=result)
        return result

    s109.combine_probabilities = combine
    return original


def predict_stage3_bias(ns, data_dir, model_dir, rule):
    """Bias-only rule: preserve S109's loading, encoder, motion graph and steering."""
    from pathlib import Path
    import numpy as np
    diag = {'rule_error': None, 'videos': {}, 'accel_log_bias': rule['accel_log_bias']}
    ns['_S118_STAGE3_DIAGNOSTICS'] = diag
    with ns['_S108_LOCK']:
        s109 = ns['_s109_adapter']()
        active, records = [None], {}
        old_extract = ns['extract_video_features']

        def extract(path):
            active[0] = str(path.stem)
            records[active[0]] = {}
            return old_extract(path)

        try:
            original = _install_accel_bias(ns, s109, rule['accel_log_bias'], active, records)
        except Exception as exc:
            diag['rule_error'] = '%s: %s' % (type(exc).__name__, exc)
            return ns['_S118_BASE_PREDICT_STAGE3'](data_dir, model_dir)
        ns['extract_video_features'] = extract
        try:
            result = ns['_S118_BASE_PREDICT_STAGE3'](data_dir, model_dir)
        finally:
            s109.combine_probabilities = original
            ns['extract_video_features'] = old_extract
        for video_id, indices in result.groupby('ID', sort=False).indices.items():
            record = records.get(str(video_id), {})
            diag['videos'][str(video_id)] = {'rows': len(indices),
                'changed': int(np.count_nonzero(record['accel_s109'] != record['accel_candidate']))
                if 'accel_candidate' in record else 0,
                'error': record.get('accel_error')}
        return result


def predict_stage3_mix(ns, data_dir, model_dir, rule):
    """Use S109's one shared decode/encoder pass; keep failures local to each file."""
    from pathlib import Path
    import time
    import numpy as np
    w = float(rule['steer_video_mix']['w'])
    base_predict = ns['_S118_BASE_PREDICT_STAGE3']
    diag = {'rule_error': None, 'videos': {}, 'w': w}
    ns['_S118_STAGE3_DIAGNOSTICS'] = diag
    if w == 0 and 'accel_log_bias' not in rule:
        return base_predict(data_dir, model_dir)
    if w == 0:
        return predict_stage3_bias(ns, data_dir, model_dir, rule)
    root = Path(__file__).resolve().parents[3]
    try:
        package = load_package(root, rule['steer_video_mix']['package'])
        runtime = ns['_s108_runtime']()
        folder = Path(model_dir) if model_dir is not None else root / 'model/stage3'
        models = runtime.load_models(folder)
        steer_head = package.load(folder / 's124/steer_head.pt', runtime)
        accel_package = load_package(root, rule['accel_package']) if 'accel_package' in rule else None
        if accel_package is not None:
            accel_package.install_head(folder, models)
            probe = accel_package.load_motion(folder)
            probe.close()
    except Exception as exc:
        diag['rule_error'] = 'package load %s: %s' % (type(exc).__name__, exc)
        return base_predict(data_dir, model_dir)

    raw_fn = ns['_s015_steer_probabilities']
    extract_fn = ns['extract_video_features']
    motion_loader = ns['_s022_load_accel_weights']
    active = [None]
    records = {}

    def extract(path):
        active[0] = str(path.stem)
        records[active[0]] = {'started': time.perf_counter()}
        return extract_fn(path)

    def raw(*args, **kwargs):
        value = raw_fn(*args, **kwargs)
        if active[0] is not None:
            records[active[0]]['raw'] = value
        return value

    class Models:
        def __init__(self, base):
            self.base = base
            self.head = self.run_heads

        def encode(self, crops):
            return self.base.encode(crops)

        def run_heads(self, tensor):
            result = self.base.head(tensor)
            record = records.get(active[0])
            if record is not None:
                try:
                    n = len(record['raw'])
                    record['video'] = package.probabilities_tensor(steer_head, tensor, n)
                except Exception as exc:
                    record['error'] = 'head %s: %s' % (type(exc).__name__, exc)
            return result

    ns['extract_video_features'] = extract
    ns['_s015_steer_probabilities'] = raw
    if accel_package is not None:
        ns['_s022_load_accel_weights'] = lambda _folder: accel_package.load_motion(folder)
    base = None
    setup_error = None
    try:
        # The S109 adapter owns the exact acceleration path and per-file fallback.
        with ns['_S108_LOCK']:
            s109 = ns['_s109_adapter']()
            original = None
            try:
                if 'accel_log_bias' in rule:
                    original = _install_accel_bias(ns, s109, rule['accel_log_bias'], active, records)
                base = s109.predict(ns, data_dir, folder, runtime=runtime, models=Models(models))
            except ValueError as exc:
                setup_error = exc
            finally:
                if original is not None:
                    s109.combine_probabilities = original
    finally:
        ns['extract_video_features'] = extract_fn
        ns['_s015_steer_probabilities'] = raw_fn
        ns['_s022_load_accel_weights'] = motion_loader
    if setup_error is not None:
        diag['rule_error'] = '%s: %s' % (type(setup_error).__name__, setup_error)
        return base_predict(data_dir, model_dir)
    out = base.copy()
    labels = ns['STEER']
    for video_id, indices in out.groupby('ID', sort=False).indices.items():
        record = records.get(str(video_id), {})
        try:
            if record.get('accel_error'):
                raise ValueError(record['accel_error'])
            raw_probs, video_probs = record['raw'], record['video']
            if len(raw_probs) != len(indices) or len(video_probs) != len(indices):
                raise ValueError('S109 rows and probability rows differ')
            predicted = package.combine_steer(raw_probs, video_probs,
                        ns['_s015_smooth_probabilities'],
                        int(round(ns['_S015_STEER_WINDOW_SECONDS'] / 0.1)),
                        ns['_S015_STEER_LOG_BIAS'], w)
            # A capture/decoder mismatch means fallback to S109 for this video.
            incumbent = package.combine_steer(raw_probs, video_probs,
                        ns['_s015_smooth_probabilities'],
                        int(round(ns['_S015_STEER_WINDOW_SECONDS'] / 0.1)),
                        ns['_S015_STEER_LOG_BIAS'], 0.0)
            if [labels[int(x)] for x in incumbent] != out.iloc[indices]['steer_label'].tolist():
                raise ValueError('captured S015 steering differs from S109')
            out.iloc[indices, out.columns.get_loc('steer_label')] = [labels[int(x)] for x in predicted]
            diag['videos'][str(video_id)] = {'rows': len(indices), 'mixed': True,
                'changed': int(np.count_nonzero(predicted != incumbent)),
                'seconds': time.perf_counter() - record['started']}
        except Exception as exc:
            if 'accel_log_bias' in rule and 'accel_s109' in record and len(record['accel_s109']) == len(indices):
                out.iloc[indices, out.columns.get_loc('accel_label')] = [ns['ACCEL'][int(x)] for x in record['accel_s109']]
            diag['videos'][str(video_id)] = {'rows': len(indices), 'mixed': False,
                                            'error': '%s: %s' % (type(exc).__name__, exc)}
    diag['w0_parity_with_s109'] = all(v.get('mixed') for v in diag['videos'].values())
    return out


def load_package(root, package):
    import hashlib
    import importlib.util
    import sys
    path = (Path(root) / package / 'predict.py').resolve()
    name = '_s118_pkg_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def prior_locator(emb, original, spec):
    """S109 collision saliency searched inside a clip-position prior (clips with n > min_n).

    Only the peak search changes: same saliency, same onset backtrack. Positions are
    fractions of the clip length, so no frame rate is assumed. Any failure, or a clip
    with n <= min_n, uses S109's own locate_collision.
    """
    import numpy as np

    def locate_collision(features, decision=None):
        decision = emb['DEFAULT_DECISION'] if decision is None else decision
        try:
            score = np.asarray(emb['collision_saliency'](features, decision), dtype=np.float64)
            n = len(score)
            if n <= spec['min_n']:
                return original(features, decision)
            lo = max(0, int(round(spec['lo_frac'] * n)))
            hi = min(n - 1, int(round(spec['hi_frac'] * n)))
            if hi <= lo:
                return original(features, decision)
            weighted = score.copy()
            if spec['sigma_frac'] > 0:
                x = np.arange(n) / float(n) - spec['center_frac']
                weighted = weighted * np.exp(-0.5 * (x / spec['sigma_frac']) ** 2)
            peak = lo + int(np.argmax(weighted[lo:hi + 1]))
            return int(emb['_onset_index'](score, peak, decision.onset_ratio, decision.max_backtrack))
        except Exception:  # noqa: BLE001 - never let the prior trigger the middle-frame fallback
            return original(features, decision)
    return locate_collision


def lead_scaled_locator(emb, original, spec):
    """S109 motion entry with its frame leads (min/max/default) multiplied by scale for n > min_n.

    Same residual-mover onset rule; only the search window is widened in frames, i.e.
    the S109 10 fps-tuned leads are expressed for higher native frame rates. Any
    failure, or n <= min_n, uses S109's own locate_entry.
    """
    def locate_entry(features, collision_index, decision=None):
        decision = emb['DEFAULT_DECISION'] if decision is None else decision
        try:
            n = len(features['resid_energy'])
            if n <= spec['min_n']:
                return original(features, collision_index, decision)
            values = dict(vars(decision))
            for key in ('entry_min_lead', 'entry_max_lead', 'entry_default_lead'):
                values[key] = int(round(values[key] * float(spec['scale'])))
            return original(features, collision_index, emb['DecisionParams'](**values))
        except Exception:  # noqa: BLE001 - never trigger the middle-frame fallback
            return original(features, collision_index, decision)
    return locate_entry


def _predict_lead_scale(ns, data_dir, model_dir, rule, diagnostics):
    emb = ns['_S2_COLLISION_NAMESPACE']
    original = emb['locate_entry']
    emb['locate_entry'] = lead_scaled_locator(emb, original, rule['entry_frame'])
    try:
        return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    finally:
        emb['locate_entry'] = original


def _predict_position_prior(ns, data_dir, model_dir, rule, diagnostics):
    emb = ns['_S2_COLLISION_NAMESPACE']
    original = emb['locate_collision']
    emb['locate_collision'] = prior_locator(emb, original, rule['collision_frame'])
    try:
        return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    finally:
        emb['locate_collision'] = original


def _rerank_base(ns, data_dir, model_dir, rule, diagnostics):
    """Patch S109 collision only while its full Stage2 base call is running."""
    emb = ns['_S2_COLLISION_NAMESPACE']
    original = emb['locate_collision']
    spec = rule['collision_frame']
    try:
        root = Path(__file__).resolve().parents[3]
        package = load_package(root, spec['package'])
        if spec.get('reanchor_s109_fields'):
            if not hasattr(package, 'predict_collision_only'):
                raise ValueError('reanchor needs a collision-only package')
            return _rerank_reanchor(ns, data_dir, model_dir, package, root / spec['package'],
                                    spec['min_n'], diagnostics)
        if spec.get('preserve_s109_fields') and hasattr(package, 'predict_collision_only'):
            return package.predict_collision_only(ns, data_dir, model_dir,
                                                  root / spec['package'], spec['min_n'], diagnostics)
        locator = package.make_locator(emb, original, root / spec['package'],
                                       spec['min_n'], diagnostics)
    except Exception as exc:  # package failure: use S109's own collision locator
        diagnostics['rerank_error'] = '%s: %s' % (type(exc).__name__, exc)
        return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    if spec.get('preserve_s109_fields'):
        return _rerank_collision_capture(ns, data_dir, model_dir, locator, diagnostics)
    emb['locate_collision'] = locator
    try:
        return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    finally:
        emb['locate_collision'] = original


def _rerank_collision_capture(ns, data_dir, model_dir, rerank, diagnostics):
    """One S109 base pass; captured alternatives never reach its downstream code."""
    emb = ns['_S2_COLLISION_NAMESPACE']
    own_collision, own_folder = emb['locate_collision'], emb['predict_folder']
    active = [None]
    replacements = {}

    def locate(features, decision=None):
        original_index = own_collision(features, decision)
        current = active[0]
        if current is not None:
            try:
                current['index'] = int(rerank(features, decision))
            except Exception as exc:
                current['error'] = '%s: %s' % (type(exc).__name__, exc)
        return original_index

    def folder(path, *args, **kwargs):
        current = {}
        active[0] = current
        try:
            row = own_folder(path, *args, **kwargs)
            if 'index' in current:
                try:
                    numbers = emb['frame_numbers'](path)
                    if not 0 <= current['index'] < len(numbers):
                        raise ValueError('captured rerank index outside current file')
                    replacements[str(row['ID'])] = int(numbers[current['index']])
                except Exception as exc:
                    current['error'] = '%s: %s' % (type(exc).__name__, exc)
            if 'error' in current:
                diagnostics.setdefault('capture_errors', {})[str(row['ID'])] = current['error']
            return row
        finally:
            active[0] = None

    emb['locate_collision'], emb['predict_folder'] = locate, folder
    try:
        baseline = ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    finally:
        emb['locate_collision'], emb['predict_folder'] = own_collision, own_folder
        active[0] = None
    reranked = baseline.copy()
    reranked['collision_frame'] = [replacements.get(str(key), int(old))
                                  for key, old in zip(baseline['ID'], baseline['collision_frame'])]
    diagnostics['single_base_call'] = True
    diagnostics['captured_collision_frames'] = replacements
    return collision_only_output(baseline, reranked, diagnostics)


def _rerank_reanchor(ns, data_dir, model_dir, package, package_dir, min_n, diagnostics):
    """S143: collision-only rerank, then S109's own entry/side/evasion logic at the new collision.

    Same S109 derivations as the released chain, re-run only for clips whose collision moved:
    entry = embedded motion locate_entry(features, new index) (S010); side = S008 track origin
    on its window for (entry = collision - 9 frame numbers as in S007, new collision); evasion =
    S012 clearance on those detections. A None side / non-decisive evasion keeps the S109
    value (S109's scene-head fallback is not recomputed). Any failure keeps that clip's
    collision-only row (S109 fields, entry clamped).
    """
    import pandas as pd
    emb = ns['_S2_COLLISION_NAMESPACE']
    own_collision, own_folder = emb['locate_collision'], emb['predict_folder']
    active = [None]
    captured = {}

    def locate(features, decision=None):
        index = own_collision(features, decision)
        current = active[0]
        if current is not None and len(features['resid_energy']) > int(min_n):
            current.update(features=features, s109_index=int(index))
        return index

    def folder(path, *args, **kwargs):
        current = {'path': Path(path)}
        active[0] = current
        try:
            row = own_folder(path, *args, **kwargs)
            if 'features' in current:
                captured[str(row['ID'])] = current
            return row
        finally:
            active[0] = None

    emb['locate_collision'], emb['predict_folder'] = locate, folder
    try:
        out = package.predict_collision_only(ns, data_dir, model_dir, package_dir, min_n, diagnostics)
    finally:
        emb['locate_collision'], emb['predict_folder'] = own_collision, own_folder
        active[0] = None
    report = diagnostics.setdefault('reanchor', {'clips': {}, 'errors': {}})
    moved = {}
    for row_index, row in out.iterrows():
        ident = str(row['ID'])
        record = captured.pop(ident, None)
        if record is None:
            continue
        try:
            numbers = [int(v) for v in emb['frame_numbers'](record['path'])]
            if len(numbers) != len(record['features']['resid_energy']):
                raise ValueError('frame count mismatch')
            new_index = position_of(numbers, row['collision_frame'])
            if numbers[new_index] != int(row['collision_frame']) or new_index == record['s109_index']:
                continue
            entry_index = min(int(emb['locate_entry'](record['features'], new_index)), new_index)
            target = int(row['collision_frame']) - 9
            s007_entry = min(numbers, key=lambda v: abs(v - target))
            moved[ident] = {'row': row_index, 'entry': int(numbers[entry_index]),
                            's007_entry': int(s007_entry), 'collision': int(row['collision_frame'])}
        except Exception as exc:  # noqa: BLE001 - keep the collision-only row
            report['errors'][ident] = '%s: %s' % (type(exc).__name__, exc)
    captured.clear()
    if not moved:
        return out
    sides, evidence = {}, {}
    try:
        window = pd.DataFrame([{'ID': k, 'entry_frame': v['s007_entry'], 'collision_frame': v['collision']}
                               for k, v in moved.items()])
        capture = ns['_s012_capture_predict_sides'](ns['_s008_predict_sides'], evidence)
        sides = capture(data_dir, model_dir, window)
    except Exception as exc:  # noqa: BLE001 - entries still re-anchor; side/evasion stay S109
        report['side_error'] = '%s: %s' % (type(exc).__name__, exc)
        sides, evidence = {}, {}
    for ident, v in moved.items():
        i = v['row']
        side = sides.get(ident)
        ev = evidence.get(ident, {})
        record = {'s109': [int(out.at[i, 'entry_frame']), out.at[i, 'entry_side'], int(out.at[i, 'evasion_space'])]}
        out.at[i, 'entry_frame'] = v['entry']
        if side in ('LEFT', 'RIGHT'):
            out.at[i, 'entry_side'] = side
        if ev.get('decisive') and int(ev.get('prediction', -1)) in (0, 1):
            out.at[i, 'evasion_space'] = int(ev['prediction'])
        record['final'] = [int(out.at[i, 'entry_frame']), out.at[i, 'entry_side'], int(out.at[i, 'evasion_space'])]
        report['clips'][ident] = record
    out['entry_frame'] = out['entry_frame'].astype('int64')
    out['evasion_space'] = out['evasion_space'].astype('int64')
    return out


def collision_only_output(baseline, reranked, diagnostics):
    """Preserve S109 downstream fields, with only an entry<=collision clamp."""
    if baseline['ID'].tolist() != reranked['ID'].tolist():
        raise ValueError('rerank output IDs/order differ from S109')
    out = baseline.copy()
    out['collision_frame'] = reranked['collision_frame'].to_numpy()
    clamp = out['entry_frame'] > out['collision_frame']
    out.loc[clamp, 'entry_frame'] = out.loc[clamp, 'collision_frame']
    diagnostics['entry_validity_clamps'] = [str(v) for v in out.loc[clamp, 'ID']]
    return out


def predict(ns, data_dir, model_dir):
    try:
        early = load_rule(Path(__file__).resolve().parent / 'rule.json')
    except Exception:  # noqa: BLE001 - reported by the main path below
        early = None
    if early is not None and early.get('collision_frame', {}).get('type') == 'position_prior':
        diagnostics = {'rule_error': None, 'clips': {}, 'position_prior': True}
        ns['_S118_LAST_DIAGNOSTICS'] = diagnostics
        try:
            return _predict_position_prior(ns, data_dir, model_dir, early, diagnostics)
        except Exception as exc:  # noqa: BLE001 - whole-run failure keeps S109
            diagnostics['rule_error'] = 'position_prior %s: %s' % (type(exc).__name__, exc)
            return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    if early is not None and early.get('collision_frame', {}).get('type') == 'rerank':
        diagnostics = {'rule_error': None, 'clips': {}, 'rerank': True}
        ns['_S118_LAST_DIAGNOSTICS'] = diagnostics
        try:
            if early.get('entry_frame', {}).get('type') == 'track':
                return _predict_track(ns, data_dir, model_dir, early,
                                      base_call=lambda: _rerank_base(ns, data_dir, model_dir, early, diagnostics),
                                      diagnostics=diagnostics)
            if early.get('entry_frame', {}).get('type') == 'first_frame':
                # S145: S130_ef entry (first frame) on the rerank output; other fields unchanged.
                out = _rerank_base(ns, data_dir, model_dir, early, diagnostics)
                return _apply_rows(ns, out, early, ['entry_frame'], data_dir, diagnostics, None)
            return _rerank_base(ns, data_dir, model_dir, early, diagnostics)
        except Exception as exc:  # whole-run failure keeps S109
            diagnostics['rule_error'] = 'rerank %s: %s' % (type(exc).__name__, exc)
            return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    if early is not None and early.get('entry_frame', {}).get('type') == 'motion_lead_scale':
        diagnostics = {'rule_error': None, 'clips': {}, 'motion_lead_scale': True}
        ns['_S118_LAST_DIAGNOSTICS'] = diagnostics
        try:
            return _predict_lead_scale(ns, data_dir, model_dir, early, diagnostics)
        except Exception as exc:  # noqa: BLE001 - whole-run failure keeps S109
            diagnostics['rule_error'] = 'motion_lead_scale %s: %s' % (type(exc).__name__, exc)
            return ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    if early is not None and early.get('entry_frame', {}).get('type') == 'track':
        return _predict_track(ns, data_dir, model_dir, early)
    base = ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    diagnostics = {'rule_error': None, 'clips': {}}
    ns['_S118_LAST_DIAGNOSTICS'] = diagnostics
    try:
        rule = load_rule(Path(__file__).resolve().parent / 'rule.json')
        fields = changed_fields(rule)
    except Exception as exc:  # noqa: BLE001 - fail closed to S109
        diagnostics['rule_error'] = '%s: %s' % (type(exc).__name__, exc)
        return base
    if not fields or len(base) == 0:
        return base
    import time
    localiser = None
    if rule.get('collision_frame', {'type': 'passthrough'})['type'] == 'learned':
        start = time.perf_counter()
        try:
            root = Path(__file__).resolve().parents[3]
            localiser = load_package(root, rule['collision_frame']['package']).Localiser(root / 'model')
        except Exception as exc:  # noqa: BLE001 - no model: every clip keeps S109
            diagnostics['rule_error'] = 'learned load %s: %s' % (type(exc).__name__, exc)
            return base
        diagnostics['learned_load_seconds'] = time.perf_counter() - start
        diagnostics['learned_device'] = str(localiser.device)
    try:
        return _apply_rows(ns, base, rule, fields, data_dir, diagnostics, localiser)
    finally:
        if localiser is not None:
            localiser.close()


def _apply_rows(ns, base, rule, fields, data_dir, diagnostics, localiser):
    import time
    out = base.copy()
    image_root = Path(data_dir) / 'images'
    collisions, entries = [], []
    for sample_id, old_collision, old_entry in zip(out['ID'], out['collision_frame'], out['entry_frame']):
        record = {'s109': [int(old_collision), int(old_entry)]}
        try:
            folder = image_root / str(sample_id)
            paths = list(ns['_s008_frame_paths'](folder))
            numbers = [int(ns['_s008_frame_number'](p)) for p in paths]
            learned = None
            if localiser is not None:
                start = time.perf_counter()
                learned = localiser.position(paths)
                record.update(learned_position=learned, learned_seconds=time.perf_counter() - start)
            collision, entry = apply_rule(rule, numbers, old_collision, old_entry, learned)
            record.update(frames=len(numbers), final=[collision, entry])
        except Exception as exc:  # noqa: BLE001 - one bad clip keeps its S109 row
            collision, entry = int(old_collision), int(old_entry)
            record['error'] = '%s: %s' % (type(exc).__name__, exc)
        collisions.append(collision)
        entries.append(entry)
        diagnostics['clips'][str(sample_id)] = record
    if 'collision_frame' in fields:
        out['collision_frame'] = collisions
    if 'entry_frame' in fields:
        out['entry_frame'] = entries
    return out


def track_values(decision, numbers, base_entry, base_side, spec):
    """Pure S132 policy mapping: package decide() output -> (entry frame number, side)."""
    position, side = decision
    entry = int(base_entry) if position is None else int(numbers[_clip(position, len(numbers))])
    side = side if spec['side'] and side in ('LEFT', 'RIGHT') else base_side
    return entry, side


def _predict_track(ns, data_dir, model_dir, rule, base_call=None, diagnostics=None):
    """S132: S109 row, then per clip the tracked participant's entry frame / side.

    Collision, evasion and every S109 field the tracker does not decide pass through.
    Load failure keeps the supplied base rows; a clip error keeps its base values.
    S161 bounds detector frames independently per clip. The legacy budget_seconds
    schema field does not control predictions, so input order cannot affect entry.
    """
    import time
    spec = rule['entry_frame']
    if diagnostics is None:
        diagnostics = {'rule_error': None, 'clips': {}, 'track': True}
    else:
        diagnostics['track'] = True
    ns['_S118_LAST_DIAGNOSTICS'] = diagnostics
    base = base_call() if base_call is not None else ns['_S118_BASE_PREDICT_STAGE2'](data_dir, model_dir)
    if len(base) == 0:
        return base
    start = time.perf_counter()
    try:
        root = Path(__file__).resolve().parents[3]
        package = load_package(root, spec['package'])
        tracker = package.Tracker(root / spec['package'])
    except Exception as exc:  # noqa: BLE001 - no tracker: every clip keeps S109
        diagnostics['rule_error'] = 'track load %s: %s' % (type(exc).__name__, exc)
        return base
    diagnostics['load_seconds'] = time.perf_counter() - start
    diagnostics['device'] = str(tracker.device)
    out = base.copy()
    entries, sides = [], []
    spent = 0.0
    try:
        for row in base.itertuples(index=False):
            record = {'s109': [int(row.collision_frame), int(row.entry_frame), row.entry_side]}
            entry, side = int(row.entry_frame), row.entry_side
            try:
                paths = list(ns['_s008_frame_paths'](Path(data_dir) / 'images' / str(row.ID)))
                numbers = [int(ns['_s008_frame_number'](p)) for p in paths]
                record['frames'] = len(numbers)
                # S161: deterministic per-file frame cap in the tracker. A cumulative
                # wall-time fallback would make answers depend on input order.
                if len(numbers) > spec['min_n']:
                    t = time.perf_counter()
                    result = tracker.analyze(paths, numbers, position_of(numbers, row.collision_frame))
                    record['seconds'] = time.perf_counter() - t
                    spent += record['seconds']
                    record['result'] = {k: result.get(k) for k in ('reason', 'entry_index', 'side', 'track_len',
                                                                     'horizon', 'detected_frames', 'bracket')}
                    entry, side = track_values(package.decide(result, spec['fallback']), numbers,
                                               row.entry_frame, row.entry_side, spec)
            except Exception as exc:  # noqa: BLE001 - one bad clip keeps its S109 row
                entry, side = int(row.entry_frame), row.entry_side
                record['error'] = '%s: %s' % (type(exc).__name__, exc)
            entry = min(int(entry), int(row.collision_frame))
            record['final'] = [entry, side]
            entries.append(int(entry))
            sides.append(side)
            diagnostics['clips'][str(row.ID)] = record
    finally:
        tracker.close()
    diagnostics['track_seconds'] = spent
    out['entry_frame'] = entries
    if spec['side']:
        out['entry_side'] = sides
    return out


# S178: consume only the S161/S176 side decision from the tracker diagnostics.
# The S118 rule keeps side=false because its collision-preservation validator
# requires it. This wrapper updates side after the exact S176 track call.
try:
    _s178_previous_predict_track = _predict_track

    def _predict_track(ns, data_dir, model_dir, rule, base_call=None, diagnostics=None):
        out = _s178_previous_predict_track(ns, data_dir, model_dir, rule,
                                           base_call=base_call, diagnostics=diagnostics)
        try:
            if rule.get('entry_frame', {}).get('package') != 'model/stage2/s161':
                return out
            records = ns.get('_S118_LAST_DIAGNOSTICS', {}).get('clips', {})
            if not records or 'entry_side' not in out:
                return out
            revised = out.copy()
            for index, row in out.iterrows():
                record = records.get(str(row['ID']), {})
                if record.get('error'):
                    continue
                result = record.get('result') or {}
                if result.get('reason') not in ('crossing', 's176_inside_from_start',
                                                's176_inside_first_minus_0p2s'):
                    continue
                side = result.get('side')
                if side in ('LEFT', 'RIGHT'):
                    revised.at[index, 'entry_side'] = side
            return revised
        except Exception:
            # Any side-application fault returns the exact S176 output.
            return out
except Exception:
    pass
