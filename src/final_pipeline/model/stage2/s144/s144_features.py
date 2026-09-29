"""S144 candidate features on top of S142 (H1 motion 34 + box 30), shared by training and a future package.

All features are candidate-local or relative to the other candidates of the same clip; no clip
position/length. Inputs are the S142 cache rows (candidates[k]['vector'], ['box_vector'], ['peak'],
detections[str(frame)] = {'boxes': [[x1,y1,x2,y2,score,label]...], 'hist': [...]}).
"""
import numpy as np

OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)
CONTACT_NAMES = ('max_area_pre', 'max_area_peak', 'max_area_post', 'max_area_jump', 'max_area_accel',
                 'bottom_touch_pre', 'bottom_touch_post', 'bottom_touch_gain', 'lower_cover_pre', 'lower_cover_post',
                 'lower_cover_gain', 'count_drop', 'central_area_gain', 'central_bottom_gain')
# features whose within-clip rank / deviation is added by the context transform
CONTEXT_KEYS = ('saliency_peak', 'saliency_prominence', 'jolt_speed', 'jolt_theta', 'jolt_warp',
                'resid_energy_rise', 'box_max_impact', 'box_area_growth', 'box_area_log_ratio', 'box_scale_jump',
                'box_anchor_area', 'box_anchor_bottom', 'box_overlap_growth', 'box_disappeared_after')


def schedule(peak, n, fps):
    return [max(0, min(n - 1, int(peak) + int(round(t * fps)))) for t in OFFSETS]


def _area(b):
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _cover(boxes, y_min=0.6):
    spans = sorted((max(0.0, b[0]), min(1.0, b[2])) for b in boxes if b[3] >= y_min)
    total, cur = 0.0, None
    for a, b in spans:
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return total + (cur[1] - cur[0] if cur else 0.0)


def contact_vector(frames):
    """frames: five box lists (normalized [x1,y1,x2,y2,score,label]) at OFFSETS."""
    mx = [max([_area(b) for b in f], default=0.0) for f in frames]
    touch = [sum(1 for b in f if b[3] > 0.97) for f in frames]
    cover = [_cover(f) for f in frames]
    count = [len(f) for f in frames]
    central = [max([_area(b) for b in f if b[0] < 0.5 < b[2]], default=0.0) for f in frames]
    cbottom = [max([b[3] for b in f if b[0] < 0.5 < b[2]], default=0.0) for f in frames]
    pre = lambda v: float(np.mean(v[:2]))
    post = lambda v: float(np.mean(v[3:]))
    return [pre(mx), float(mx[2]), post(mx), float(mx[2] - mx[1]), float(mx[3] - 2 * mx[2] + mx[1]),
            pre(touch), post(touch), post(touch) - pre(touch), pre(cover), post(cover), post(cover) - pre(cover),
            pre(count) - post(count), post(central) - pre(central), post(cbottom) - pre(cbottom)]


def base_matrix(row):
    return np.asarray([c['vector'] + c['box_vector'] for c in row['candidates']], dtype=np.float64)


def contact_matrix(row, fps=None):
    fps = fps or (30.0 if row['n'] > 310 else 10.0)  # deployment assumption (S142): fixed 30 on long clips
    det = row['detections']
    out = []
    for c in row['candidates']:
        frames = [[b for b in det.get(str(i), {'boxes': []})['boxes']] for i in schedule(c['peak'], row['n'], fps)]
        out.append(contact_vector(frames))
    return np.asarray(out, dtype=np.float64)


def context_matrix(x, names):
    """Within-clip deviation from the median and normalized rank for CONTEXT_KEYS."""
    cols = [names.index(k) for k in CONTEXT_KEYS if k in names]
    sub = x[:, cols]
    dev = sub - np.median(sub, axis=0, keepdims=True)
    k = max(1, len(x) - 1)
    rank = np.argsort(np.argsort(-sub, axis=0, kind='stable'), axis=0, kind='stable') / float(k)
    return np.concatenate([dev, rank], axis=1)


def context_names(names):
    keys = [k for k in CONTEXT_KEYS if k in names]
    return ['ctxdev_' + k for k in keys] + ['ctxrank_' + k for k in keys]
