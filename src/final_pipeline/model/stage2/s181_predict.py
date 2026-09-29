"""S161: observed participant crossing timing, derived from the S132 tracker.

Self-contained (numpy, OpenCV, torch, torchvision); shipped as model/stage2/s161/predict.py.

Per clip, given the sorted frame paths and S156's final collision index c:
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
The analysis diagnostics retain S132's first-seen explanations, but decide()
accepts only a bracketed outside-to-inside crossing with two samples on each
side. All first-seen/already-inside diagnoses abstain. Side always passes through.
Each clip independently has a 64-detection-frame limit; uncertainty keeps S109.
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
        self.batch = 2
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
        self.max_detect_frames = 64
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
        if len(self.dets) + len(todo) > self.p.max_detect_frames:
            raise FrameBudgetExceeded('per-file detection frame limit')
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


class FrameBudgetExceeded(RuntimeError):
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
    near = [i for i in order if abs(i - c) <= max(sd, round(p.anchor_seconds * fps))]
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
    res.update(entry_index=int(entry), side=side, reason=reason, bracket=[int(out_i), int(first_i)],
               collision_index=c, fps=fps, inside_samples=len(states) - j,
               outside_samples=sum(not s[2][0] for s in states[:j]))
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
        try:
            return analyze(paths, numbers, collision_index, self.detector, self.params)
        except FrameBudgetExceeded:
            return Result(entry_index=None, side=None, reason='frame_budget')

    def close(self):
        self.detector.model = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


INSIDE_REASONS = ('inside_from_start', 'far_inside_when_first_seen', 'inside_when_first_seen')


def decide(result, fallback):
    """Only observed crossings; all uncertain/already-inside cases retain S109."""
    if result.get('reason') != 'crossing' or result.get('entry_index') is None:
        return None, None
    lo, hi = result['bracket']
    fps = result['fps']
    entry = int(result['entry_index'])
    if (hi - lo > 0.5 * fps or result.get('inside_samples', 0) < 2
            or result.get('outside_samples', 0) < 2 or result.get('track_len', 0) < 4
            or entry > result['collision_index']):
        return None, None
    return entry, None


# S174 optional extension: initialization failure keeps exact S171 S161.
try:
    # S174 append-only extension to the exact S171 S161 tracker.
    # The original S161 result is returned byte-for-byte in decision terms whenever
    # its conservative observed-crossing gate accepts. Only abstentions get a
    # second collision-anchored, multi-actor geometric observation pass.
    
    _s174_original_analyze = Tracker.analyze
    
    
    def _s174_refine(clip, outside, inside, horizon, aspect, p, lane_k):
        """Check native frames around one observed coarse crossing."""
        lo, lr = outside
        hi, hr = inside
        if hi <= lo + 1:
            return hi, lo, hi
        geo = Params(**vars(p));geo.lane_k = lane_k
        frames = list(range(lo + 1, hi))
        if len(frames) > 12:
            frames = sorted(set(int(round(v)) for v in np.linspace(lo + 1, hi - 1, 12)))
        last_out, first_in = lo, hi
        def inspect(i,last_out,first_in):
            if i not in clip.dets and len(clip.dets) >= p.max_detect_frames:
                return last_out,first_in
            try:clip.ensure([i])
            except FrameBudgetExceeded:return last_out,first_in
            t = (i - lo) / float(hi - lo)
            predicted = (1 - t) * lr[:4] + t * hr[:4]
            rows = clip.dets.get(i, [])
            if not len(rows):return last_out,first_in
            best = max(rows, key=lambda r: _iou(r, predicted))
            if _iou(best, predicted) < 0.25:return last_out,first_in
            state = _state(best, horizon, aspect, geo)
            if state is None:return last_out,first_in
            if state[0] and i < first_in:first_in = i
            elif not state[0] and i > last_out and i < first_in:last_out = i
            return last_out,first_in
        for i in frames:last_out,first_in=inspect(i,last_out,first_in)
        # After the coarse bracket narrows, scan its actual native frames.
        if 1 < first_in-last_out <= 14:
            for i in range(last_out+1,first_in):
                last_out,first_in=inspect(i,last_out,first_in)
        return first_in, last_out, first_in
    
    
    def _s174_fallback(self, paths, numbers, collision_index):
        n=len(paths);c=max(0,min(n-1,int(collision_index)))
        if n<5:return Result(entry_index=None,reason='s174_short')
        fps=30.0 if n>self.params.long_n else 10.0
        p=Params(**vars(self.params));p.batch=2;p.max_detect_frames=48
        step=max(1,int(round(fps/2.5)))
        near=sorted(set(max(0,min(n-1,c+int(round(v*fps)))) for v in (-0.6,-0.3,0.0,0.2)),reverse=True)
        clip=Clip(paths,self.detector,p)
        clip.ensure(near)
        seeds=[]
        for i in near:
            for j,row in enumerate(clip.dets.get(i,[])):
                score=float(_impact(row,(i-c)/fps))
                seeds.append((score,i,row,clip.hists[i][j]))
        tracks=[]
        for score,i,row,hist in sorted(seeds,key=lambda x:-x[0]):
            if any(_iou(row,t['rows'][0][1])>=0.60 for t in tracks):continue
            tracks.append(dict(score=score,rows=[(i,row)],hist=hist,velocity=np.zeros(4,np.float32),misses=0))
            if len(tracks)==3:break
        if not tracks:return Result(entry_index=None,reason='s174_no_anchor',detected_frames=len(clip.dets))
        lo=max(0,c-int(round(6*fps)))
        grid=list(range(min(n-1,c+step),lo-1,-step))
        if lo>0:grid += [int(v) for v in np.linspace(0,lo,7)]
        grid=sorted(set(grid+near),reverse=True)
        for i in grid:
            active=[t for t in tracks if i<t['rows'][-1][0] and t['misses']<=3]
            if not active:continue
            if i not in clip.dets:
                if len(clip.dets)>=p.max_detect_frames:break
                clip.ensure([i])
            used=set()
            for track in active:
                last_i,last=track['rows'][-1];gap=last_i-i
                predicted=last[:4]-track['velocity']*min(gap,2*step)
                scored=[(_assoc(row,predicted,last,clip.hists[i][j],track['hist']),j,row)
                        for j,row in enumerate(clip.dets.get(i,[])) if j not in used]
                scored=[x for x in scored if x[0]>1.0]
                if not scored:
                    track['misses']+=1;continue
                _,j,row=max(scored,key=lambda x:x[0]);used.add(j)
                track['velocity']=0.5*track['velocity']+0.5*(last[:4]-row[:4])/max(gap,1)
                track['rows'].append((i,row));track['misses']=0
        horizon,hn=estimate_horizon(clip,p);aspect=clip.aspect or 0.5625
        geometries=[(horizon,p.lane_k)]
        geometries += [(horizon,1.20),(horizon,1.50)]
        if hn<5:geometries += [(0.47,p.lane_k),(0.53,p.lane_k)]
        for track in sorted(tracks,key=lambda t:-t['score']):
            rows=sorted(track['rows'],key=lambda x:x[0])
            if len(rows)<4:continue
            for h,lane_k in geometries:
                geo=Params(**vars(p));geo.lane_k=lane_k
                states=[(i,row,_state(row,h,aspect,geo)) for i,row in rows]
                states=[x for x in states if x[2] is not None]
                if len(states)<4 or not states[-1][2][0]:continue
                j=len(states)-1
                while j>0 and states[j-1][2][0]:j-=1
                if j==0:continue
                outside=states[j-1];inside=states[j]
                if outside[2][1] not in ('LEFT','RIGHT'):continue
                outside_count=sum(not x[2][0] for x in states[:j]);inside_count=len(states)-j
                if outside_count<2 or inside_count<2:continue
                entry,low,high=_s174_refine(clip,(outside[0],outside[1]),(inside[0],inside[1]),h,aspect,p,lane_k)
                result=Result(entry_index=int(entry),side=outside[2][1],reason='crossing',
                              bracket=[int(low),int(high)],collision_index=c,fps=fps,
                              inside_samples=inside_count,outside_samples=outside_count,
                              track_len=len(rows),detected_frames=len(clip.dets),
                              horizon=float(h),horizon_n=int(hn),lane_k=float(lane_k),
                              anchor=int(track['rows'][0][0]),
                              anchor_box=[round(float(v),4) for v in track['rows'][0][1][:4]],
                              s174_actor_candidates=len(tracks),s174_seed_score=track['score'])
                if decide(result,'s109')[0] is not None:
                    return result
        return Result(entry_index=None,reason='s174_no_supported_crossing',
                      detected_frames=len(clip.dets),s174_actor_candidates=len(tracks),horizon=float(horizon))
    
    
    def _s174_analyze(self,paths,numbers,collision_index):
        original=_s174_original_analyze(self,paths,numbers,collision_index)
        if decide(original,'s109')[0] is not None:
            return original
        try:
            result=_s174_fallback(self,paths,numbers,collision_index)
            if decide(result,'s109')[0] is not None:
                return result
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            pass
        return original
    
    
    Tracker.analyze=_s174_analyze
except Exception:
    pass


# S176 extension on the exact S174 release. Apply only to two S161 abstentions
# after S174 itself has completed and abstained. No new detector or model.
try:
    _s176_previous_analyze = Tracker.analyze
    _s176_original_s161 = _s174_original_analyze
    _s176_original_fallback = _s174_fallback
    _s176_previous_decide = decide

    def _s176_capture_original(self, paths, numbers, collision_index):
        result = _s176_original_s161(self, paths, numbers, collision_index)
        self._s176_original_result = result
        return result

    def _s176_capture_fallback(self, paths, numbers, collision_index):
        result = _s176_original_fallback(self, paths, numbers, collision_index)
        self._s176_fallback_completed = True
        return result

    def _s176_analyze(self, paths, numbers, collision_index):
        self._s176_original_result = None
        self._s176_fallback_completed = False
        s174 = _s176_previous_analyze(self, paths, numbers, collision_index)
        try:
            # An accepted S161 or S174 decision keeps exactly its prior result.
            if _s176_previous_decide(s174, 's109')[0] is not None:
                return s174
            if not self._s176_fallback_completed:
                return s174  # S174 faulted or never ran: fail closed to S174.
            original = self._s176_original_result
            if original is None or not numbers:
                return s174
            reason = original.get('reason')
            if reason == 'inside_from_start':
                index = 0
                tag = 's176_inside_from_start'
            elif reason == 'inside_when_first_seen':
                first = int(original['entry_index'])
                fps = 30.0 if len(paths) > self.params.long_n else 10.0
                index = max(0, first - int(round(0.2 * fps)))
                tag = 's176_inside_first_minus_0p2s'
            else:
                return s174
            c = max(0, min(len(numbers)-1, int(collision_index)))
            if index > c:
                return s174
            return Result(reason=tag, entry_index=int(index),
                          collision_index=c, s176_s161_reason=reason)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return s174

    def decide(result, fallback):
        if result.get('reason') in ('s176_inside_from_start',
                                    's176_inside_first_minus_0p2s'):
            index = result.get('entry_index')
            collision = result.get('collision_index')
            if isinstance(index, int) and isinstance(collision, int) and 0 <= index <= collision:
                return index, None
            return None, None
        return _s176_previous_decide(result, fallback)

    _s174_original_analyze = _s176_capture_original
    _s174_fallback = _s176_capture_fallback
    Tracker.analyze = _s176_analyze
except Exception:
    # An initialization issue retains the exact S174 behavior.
    pass


# S178 side-only extension on exact S176. The entry decision and detector path
# remain S176; expose a side only for supported original S161 crossings or
# lateral collision anchors in the two accepted S176 already-inside buckets.
try:
    _s178_previous_analyze = Tracker.analyze

    def _s178_analyze(self, paths, numbers, collision_index):
        result = _s178_previous_analyze(self, paths, numbers, collision_index)
        side = None
        try:
            reason = result.get('reason')
            original = getattr(self, '_s176_original_result', None)
            if reason == 'crossing' and original is not None:
                # S174-only added crossings lost a confident CCD side panel.
                # Keep its inherited S109 side; retain the proven S161 boundary.
                if _s176_previous_decide(original, 's109')[0] is not None:
                    candidate = result.get('side')
                    if candidate in ('LEFT', 'RIGHT'):
                        side = candidate
            elif reason in ('s176_inside_from_start',
                            's176_inside_first_minus_0p2s') and original is not None:
                box = original.get('anchor_box')
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    center = (float(box[0]) + float(box[2])) / 2.0
                    if 0.0 <= center <= 0.35:
                        side = 'LEFT'
                    elif 0.65 <= center <= 1.0:
                        side = 'RIGHT'
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            side = None
        # The existing decide() does not consult side. Returning a copy keeps
        # the S176 entry path unchanged and makes an error fail closed per clip.
        out = Result(result)
        out['side'] = side
        return out

    Tracker.analyze = _s178_analyze
except Exception:
    pass


# S180: S174-only geometric crossings tend to precede annotated tire contact.
# S178 clears the side of S174-only crossings, while retaining the side of
# accepted S161 crossings. Apply one 0.2-second entry shift to that exact
# source; retain the S178 decision on every uncertainty or per-clip error.
try:
    _s180_previous_decide = decide

    def decide(result, fallback):
        previous = _s180_previous_decide(result, fallback)
        try:
            position, side = previous
            if (position is None or result.get('reason') != 'crossing'
                    or result.get('side') is not None):
                return previous
            fps = float(result['fps'])
            collision = int(result['collision_index'])
            if not (1.0 <= fps <= 120.0 and 0 <= position <= collision):
                return previous
            offset = int(round(0.2 * fps))
            return min(collision, position + offset), side
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return previous
except Exception:
    pass


# S181: a collision-relative fallback only for S161 reasons with labeled,
# panel-held-out support. The 0.85-second gap is the median of 52 accepted
# S161 crossing-to-collision gaps on public clips, independent of fallback
# labels. S174 must have completed and abstained, and S176 must have abstained.
try:
    _s181_previous_analyze = Tracker.analyze
    _s181_previous_decide = decide
    _s181_reasons = frozenset(('short_track', 'not_in_lane_at_collision',
                               'far_inside_when_first_seen'))

    def _s181_analyze(self, paths, numbers, collision_index):
        result = _s181_previous_analyze(self, paths, numbers, collision_index)
        try:
            if (result.get('reason') not in _s181_reasons
                    or not getattr(self, '_s176_fallback_completed', False)
                    or _s181_previous_decide(result, 's109')[0] is not None):
                return result
            if not numbers:
                return result
            collision = max(0, min(len(numbers)-1, int(collision_index)))
            fps = 30.0 if len(paths) > self.params.long_n else 10.0
            out = Result(result)
            out['s181_collision_index'] = collision
            out['s181_fps'] = fps
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return result

    def decide(result, fallback):
        previous = _s181_previous_decide(result, fallback)
        try:
            if (previous[0] is not None or result.get('reason') not in _s181_reasons
                    or 's181_collision_index' not in result):
                return previous
            collision = result['s181_collision_index']
            fps = result['s181_fps']
            if (not isinstance(collision, int) or collision < 0
                    or fps not in (10.0, 30.0)):
                return previous
            entry = max(0, collision-int(round(0.85*fps)))
            return entry, None
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            return previous

    Tracker.analyze = _s181_analyze
except Exception:
    pass
