"""S132: collision-participant tracking for Stage2 entry_frame / entry_side.

Self-contained (numpy, OpenCV, torch, torchvision); shipped as model/stage2/s132/track.py.

Per clip, given the sorted frame paths and S109's collision index c:
  1. detect COCO road vehicles on sampled frames, backward from c (dense near c,
     sparse further back), lazily in small batches;
  2. anchor the participant = the most impact-like vehicle near c;
  3. track it backward with a motion-predicted IoU/size/appearance association;
  4. ego-lane geometry = flat-road trapezoid centred on the image column. The
     horizon row comes from the clip's own vehicle boxes (bottom - k * height);
     half-width(y) = lane_k * (y - horizon) * H / W (pinhole, lane/camera ratio);
  5. entry = first frame of the participant's final in-lane run: if the run
     reaches the clip start (or the participant was a small, distant in-lane
     vehicle when first seen) -> first frame; else the first native frame whose
     box edge crosses the boundary, refined between the bracketing samples;
  6. side = boundary crossed (or, for already-inside tracks, lateral origin).
Returns None fields when evidence is insufficient; the caller keeps S109.
Frame rate is unknown: time spans use fps_guess (30 for long clips, 10 otherwise).
"""
import math

import cv2
import numpy as np
import torch

VEHICLE_LABELS = (3, 4, 6, 8)  # COCO car, motorcycle, bus, truck


class Params:
    def __init__(self, **kw):
        self.dense_seconds = 4.0
        self.dense_step_seconds = 0.1
        self.sparse_max_samples = 40
        self.batch = 8
        self.score = 0.35
        self.anchor_seconds = 0.2
        self.max_misses = 2
        self.lane_k = 1.35
        self.horizon_k = 0.87
        self.horizon_default = 0.5
        self.horizon_mode = 'est'
        self.horizon_lo = 0.4
        self.horizon_hi = 0.6
        self.max_cross_seconds = 1e9
        self.fallback = 's109'
        self.inset = 0.05
        self.start_seconds = 1.0
        self.far_height = 0.12
        self.min_track = 3
        self.side_mode = 'crossing_or_origin'
        self.long_n = 310
        self.grid = 'relative'
        self.flicker = 0
        for key, value in kw.items():
            if not hasattr(self, key):
                raise ValueError('unknown param ' + key)
            setattr(self, key, value)


class Detector:
    """torchvision COCO detector; returns per-image arrays [x1,y1,x2,y2,score,label] normalized."""

    def __init__(self, kind, weight_path, device=None, max_side=640, score=0.3):
        import torchvision.models.detection as det
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.kind = kind
        self.score = score
        if kind == 'ssdlite':
            model = det.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
        elif kind == 'frcnn':
            model = det.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91,
                                                   min_size=int(max_side * 9 / 16), max_size=max_side,
                                                   box_score_thresh=score)
        else:
            raise ValueError(kind)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.max_side = max_side
        self.half = self.device.type == 'cuda'

    def load(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if image is None:
            return None
        h, w = image.shape[:2]
        scale = self.max_side / float(max(h, w))
        if scale < 1:
            image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return image

    def __call__(self, images):
        tensors = [torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)
                   for im in images]
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.half):
            outputs = self.model([t.to(self.device) for t in tensors])
        result = []
        for image, out in zip(images, outputs):
            h, w = image.shape[:2]
            boxes = out['boxes'].float().cpu().numpy()
            scores = out['scores'].float().cpu().numpy()
            labels = out['labels'].cpu().numpy()
            keep = (scores >= self.score) & np.isin(labels, VEHICLE_LABELS)
            b = boxes[keep] / np.array([w, h, w, h], dtype=np.float32)
            rows = np.concatenate([b, scores[keep, None], labels[keep, None].astype(np.float32)], 1)
            rows = rows[((rows[:, 2] - rows[:, 0]) > 0.01) & ((rows[:, 3] - rows[:, 1]) > 0.01)]
            result.append(_nms(rows))
        return result


def _nms(rows, thr=0.7):
    if len(rows) < 2:
        return rows
    order = np.argsort(-rows[:, 4])
    keep = []
    for i in order:
        if all(_iou(rows[i], rows[j]) < thr for j in keep):
            keep.append(i)
    return rows[keep]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 1e-9 else 0.0


def _hist(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in (box[0] * w, box[1] * h, box[2] * w, box[3] * h)]
    crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if crop.size == 0:
        return np.zeros(64, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
    s = float(hist.sum())
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


class Clip:
    """Lazy detections over one clip's frames (indices into sorted paths)."""

    def __init__(self, paths, detector, params, cache=None):
        self.paths = paths
        self.detector = detector
        self.p = params
        self.dets = {}
        self.hists = {}
        self.aspect = None
        self.cache = cache

    def ensure(self, indices):
        todo = [i for i in sorted(set(indices)) if i not in self.dets]
        cache = self.cache
        if cache is not None:
            for i in list(todo):
                hit = cache.get(str(self.paths[i]))
                if hit is not None:
                    self.dets[i], self.hists[i], aspect = hit
                    self.aspect = self.aspect or aspect
                    todo.remove(i)
        for start in range(0, len(todo), self.p.batch):
            chunk = todo[start:start + self.p.batch]
            imgs, kept = [], []
            for i in chunk:
                im = self.detector.load(self.paths[i])
                if im is None:
                    self.dets[i] = np.zeros((0, 6), np.float32)
                    self.hists[i] = np.zeros((0, 64), np.float32)
                    continue
                imgs.append(im)
                kept.append(i)
            if not imgs:
                continue
            if self.aspect is None:
                self.aspect = imgs[0].shape[0] / float(imgs[0].shape[1])
            for i, im, rows in zip(kept, imgs, self.detector(imgs)):
                self.dets[i] = rows
                self.hists[i] = np.stack([_hist(im, r) for r in rows]) if len(rows) else np.zeros((0, 64), np.float32)
                if cache is not None:
                    cache[str(self.paths[i])] = (rows, self.hists[i], im.shape[0] / float(im.shape[1]))

    def release(self):
        self.hists.clear()


def _impact(row, time_distance):
    x1, y1, x2, y2 = row[:4]
    cx = 0.5 * (x1 + x2)
    width = x2 - x1
    centrality = max(0.0, 1.0 - 1.4 * abs(cx - 0.5)) if not (x1 < 0.5 < x2) else 1.0
    proximity = 0.35 + 0.65 * y2
    size = min(1.0, 3.0 * math.sqrt(max(width * (y2 - y1), 0.0)))
    return row[4] * (0.25 + centrality) * proximity * (0.25 + size) / (1.0 + 3.0 * abs(time_distance))


def _assoc(candidate, predicted, current, hist_c, hist_ref):
    iou = _iou(candidate, predicted)
    pw, ph = predicted[2] - predicted[0], predicted[3] - predicted[1]
    cw, ch = candidate[2] - candidate[0], candidate[3] - candidate[1]
    dx = abs(0.5 * (candidate[0] + candidate[2]) - 0.5 * (predicted[0] + predicted[2])) / max(pw, 0.03)
    dy = abs(candidate[3] - predicted[3]) / max(ph, 0.03)
    ratio = min(cw * ch, pw * ph) / max(cw * ch, pw * ph, 1e-9)
    if iou < 0.05 and (dx > 1.0 or dy > 0.8):
        return -1e9
    if ratio < 0.25:
        return -1e9
    app = float(np.minimum(hist_c, hist_ref).sum())
    return 2.0 * iou + 0.8 * (1.0 - min(dx, 1.0)) + 0.5 * ratio + 1.2 * app


def lane_bounds(y, horizon, aspect, lane_k):
    hw = lane_k * (y - horizon) * aspect
    return 0.5 - hw, 0.5 + hw, hw


def estimate_horizon(clip, params):
    values = []
    if params.horizon_mode == 'fixed':
        return params.horizon_default, 0
    for rows in clip.dets.values():
        for r in rows:
            h = r[3] - r[1]
            if r[4] >= 0.5 and 0.04 < h < 0.5 and r[3] < 0.97 and int(r[5]) == 3:
                values.append(r[3] - params.horizon_k * h)
    if len(values) < 5:
        return params.horizon_default, len(values)
    return float(np.clip(np.median(values), params.horizon_lo, params.horizon_hi)), len(values)


def _state(box, horizon, aspect, params):
    """Lane relation of a box bottom edge: (in_lane, side_outside or None, penetration)."""
    x1, y1, x2, y2 = box[:4]
    if y2 <= horizon + 0.02:
        return None
    left, right, hw = lane_bounds(y2, horizon, aspect, params.lane_k)
    inset = params.inset * (x2 - x1)
    a, b = x1 + inset, x2 - inset
    if b <= left:
        return False, 'LEFT', (b - left) / hw
    if a >= right:
        return False, 'RIGHT', (right - a) / hw
    return True, None, min(b - left, right - a) / hw


class Result(dict):
    pass


def analyze(paths, numbers, collision_index, detector, params=None, fps_guess=None, cache=None):
    p = params or Params()
    n = len(paths)
    res = Result(entry_index=None, side=None, reason=None, detected_frames=0)
    if n < 3:
        res['reason'] = 'short'
        return res
    fps = fps_guess or (30.0 if n > p.long_n else 10.0)
    c = max(0, min(n - 1, int(collision_index)))
    sd = max(1, int(round(p.dense_step_seconds * fps)))
    dense_lo = max(0, c - int(round(p.dense_seconds * fps)))
    if p.grid == 'absolute':
        # sample grid fixed to frame positions (multiples of the step), so a small shift
        # of the collision anchor does not move every sampled frame
        top = min(n - 1, c + sd)
        dense = [i for i in range(top, dense_lo - 1, -1) if i % sd == 0] or [c]
        if dense_lo > 0:
            ss = max(sd, int(math.ceil(n / float(p.sparse_max_samples))))
            sparse = [i for i in range(dense[-1] - 1, -1, -1) if i % ss == 0]
            if not sparse or sparse[-1] != 0:
                sparse.append(0)
        else:
            sparse = [] if dense[-1] == 0 else [0]
    elif dense_lo > 0:
        dense = list(range(min(n - 1, c + sd), dense_lo - 1, -sd))
        ss = max(sd, int(math.ceil(dense_lo / float(p.sparse_max_samples))))
        sparse = list(range(dense[-1] - ss, -1, -ss))
        if not sparse or sparse[-1] != 0:
            sparse.append(0)
    else:
        dense = list(range(min(n - 1, c + sd), dense_lo - 1, -sd))
        sparse = [] if dense[-1] == 0 else [0]
    order = dense + sparse
    clip = Clip(paths, detector, p, cache)
    # anchor: most impact-like vehicle within anchor_seconds of c
    near = [i for i in order[:max(1, int(round(p.anchor_seconds * fps / sd)) * 2 + 2)]]
    clip.ensure(near)
    best, best_v = None, 0.0
    for i in near:
        for k2, r in enumerate(clip.dets.get(i, [])):
            v = _impact(r, (i - c) / fps)
            if v > best_v:
                best, best_v = (i, r, clip.hists[i][k2]), v
    if best is None:
        res['reason'] = 'no_anchor'
        res['detected_frames'] = len(clip.dets)
        return res
    ai, ar, ref_hist = best
    res['evasion'] = evasion(ar, clip.dets.get(ai, []))
    track = [(ai, ar)]
    velocity = np.zeros(4, np.float32)
    misses = 0
    pos = order.index(ai) if ai in order else 0
    k = pos + 1
    while k < len(order):
        chunk = order[k:k + p.batch]
        clip.ensure(chunk)
        stop = False
        for i in chunk:
            last_i, last = track[-1]
            gap = (last_i - i)
            predicted = last[:4] - velocity * min(gap, 2 * sd)
            rows = clip.dets.get(i, np.zeros((0, 6), np.float32))
            scored = [(_assoc(r, predicted, last, clip.hists[i][k2], ref_hist), r) for k2, r in enumerate(rows)]
            scored = [s for s in scored if s[0] > 1.2]
            if not scored:
                misses += 1
                if misses > p.max_misses:
                    stop = True
                    break
                continue
            value, r = max(scored, key=lambda s: s[0])
            if len(track) >= 2:
                velocity = 0.5 * velocity + 0.5 * (last[:4] - r[:4]) / max(gap, 1)
            else:
                velocity = (last[:4] - r[:4]) / max(gap, 1)
            track.append((i, r))
            misses = 0
        k += len(chunk)
        if stop:
            break
    res['detected_frames'] = len(clip.dets)
    horizon, hn = estimate_horizon(clip, p)
    aspect = clip.aspect or 0.5625
    res.update(horizon=horizon, horizon_n=hn, track_len=len(track), anchor=int(ai),
               anchor_box=[round(float(v), 4) for v in ar[:4]])
    if len(track) < p.min_track:
        res['reason'] = 'short_track'
        clip.release()
        return res
    track.sort(key=lambda t: t[0])
    states = [(i, r, _state(r, horizon, aspect, p)) for i, r in track]
    states = [s for s in states if s[2] is not None]
    res['trace'] = [[int(i), round(float(s[2]), 3), s[1]] for i, _, s in states]
    inside = [s[2][0] for s in states]
    if p.flicker and len(inside) >= 2 and not inside[-1] and inside[-2]:
        states, inside = states[:-1], inside[:-1]  # one out-of-lane sample at impact = flicker
    if len(states) < p.min_track or not inside[-1]:
        # participant not inside the ego lane at the collision end of its track
        res['reason'] = 'not_in_lane_at_collision'
        res['side'] = _origin_side(track)
        clip.release()
        return res
    j = len(states) - 1
    while j > 0:
        if inside[j - 1]:
            j -= 1
        elif p.flicker and j >= 2 and inside[j - 2]:
            j -= 2  # bridge a single out-of-lane sample inside an in-lane run
        else:
            break
    first_i, first_r, _ = states[j]
    if j == 0:
        earliest = track[0][0]
        h0 = first_r[3] - first_r[1]
        if earliest <= int(round(p.start_seconds * fps)):
            res.update(entry_index=0, reason='inside_from_start')
        elif h0 <= p.far_height:
            res.update(entry_index=0, reason='far_inside_when_first_seen')
        else:
            res.update(entry_index=int(first_i), reason='inside_when_first_seen')
        res['side'] = _origin_side(track)
        clip.release()
        return res
    out_i, out_r, out_s = states[j - 1]
    side = out_s[1]
    entry = _refine(clip, out_i, out_r, first_i, first_r, horizon, aspect, p)
    reason = 'crossing' if (c - entry) / fps <= p.max_cross_seconds else 'far_crossing'
    res.update(entry_index=int(entry), side=side, reason=reason, bracket=[int(out_i), int(first_i)])
    clip.release()
    return res


def _refine(clip, out_i, out_r, in_i, in_r, horizon, aspect, p):
    if in_i - out_i <= 1:
        return in_i
    idx = list(range(out_i + 1, in_i))
    if len(idx) > 12:
        idx = sorted(set(int(round(v)) for v in np.linspace(out_i + 1, in_i - 1, 12)))
    clip.ensure(idx)
    for i in idx:
        t = (i - out_i) / float(in_i - out_i)
        predicted = (1 - t) * out_r[:4] + t * in_r[:4]
        rows = clip.dets.get(i, [])
        best, bv = None, 0.0
        for r in rows:
            v = _iou(r, predicted)
            if v > bv:
                best, bv = r, v
        if best is None or bv < 0.3:
            continue
        s = _state(best, horizon, aspect, p)
        if s is not None and s[0]:
            return i
    return in_i


def _origin_side(track):
    ordered = sorted(track, key=lambda t: t[0])
    x0 = 0.5 * (ordered[0][1][0] + ordered[0][1][2])
    x1 = 0.5 * (ordered[-1][1][0] + ordered[-1][1][2])
    if x0 < 0.45:
        return 'LEFT'
    if x0 > 0.55:
        return 'RIGHT'
    if x1 - x0 > 0.05:
        return 'LEFT'
    if x0 - x1 > 0.05:
        return 'RIGHT'
    return None


def evasion(target, rows, road=(0.05, 0.95), pad=0.03, min_margin=0.035):
    """S012-style lateral clearance at the anchor frame: 1 = a vehicle-width corridor remains."""
    x1, y1, x2, y2, score = [float(v) for v in target[:5]]
    if score < 0.25 or y2 < 0.45:
        return None
    intervals = [(x1 - pad, x2 + pad)]
    for r in rows:
        if _iou(r, target) >= 0.5 or float(r[4]) < max(0.25, 0.5 * score) or float(r[3]) < max(0.45, y1 + 0.04):
            continue
        intervals.append((float(r[0]) - pad, float(r[2]) + pad))
    merged = []
    for a, b in sorted(intervals):
        a, b = max(road[0], a), min(road[1], b)
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cursor, gaps = road[0], []
    for a, b in merged:
        if a > cursor:
            gaps.append(a - cursor)
        cursor = max(cursor, b)
    if cursor < road[1]:
        gaps.append(road[1] - cursor)
    required = min(0.32, max(0.22, 0.08 + 0.55 * (x2 - x1)))
    margin = (max(gaps) if gaps else 0.0) - required
    return int(margin >= 0) if abs(margin) >= min_margin else None


WEIGHT_NAME = 'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'


class Tracker:
    """Deployment wrapper: package dir holds this file (predict.py), the COCO weights and optional params.json."""

    def __init__(self, root, device=None):
        import json
        from pathlib import Path
        root = Path(root)
        cfg = json.loads((root / 'params.json').read_text(encoding='utf-8')) if (root / 'params.json').is_file() else {}
        self.params = Params(**cfg.get('params', {}))
        self.detector = Detector(cfg.get('detector', 'frcnn'), root / cfg.get('weights', WEIGHT_NAME), device=device,
                                 max_side=int(cfg.get('max_side', 640)), score=float(cfg.get('score', 0.3)))
        self.device = self.detector.device

    def analyze(self, paths, numbers, collision_index):
        return analyze(paths, numbers, collision_index, self.detector, self.params)

    def close(self):
        self.detector.model = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


INSIDE_REASONS = ('inside_from_start', 'far_inside_when_first_seen', 'inside_when_first_seen')


def decide(result, fallback):
    """Policy: (entry position or None=keep S109, side or None=keep S109)."""
    reason = result.get('reason')
    if reason in INSIDE_REASONS:
        return 0, None
    if reason == 'crossing' and result.get('entry_index') is not None:
        return int(result['entry_index']), result.get('side')
    return (0 if fallback == 'first_frame' else None), None
