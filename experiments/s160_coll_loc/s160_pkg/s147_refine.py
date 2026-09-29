"""S147/S156 candidate-offset collision refiner, v1, v2 and the S156 top-K cascade (package side; NumPy only).

S156 (format s156_cascade_v1): v2 frame features in the windows of the top-K E10_base candidates plus candidate-level
features (rank, rank-mean score, gap to the best, member percentiles); binary shortlist of 8 rows, pairwise Copeland
winner, moved from the top-1 onset only if its binary margin beats it by delta. Mirrors s156_cascade.py.

v2 (format s147_refiner_v2): v1 features plus multi-scale motion; the binary frame GBDT shortlists its top-8
frames and a pairwise GBDT on feature differences picks the Copeland winner (moved only if its binary margin
beats the original onset by delta). Mirrors candidates/s143_s144_overnight/s147_refiner_v2.py.

Given the S109 motion arrays of a long clip, the chosen collision onset o and its saliency peak p, score every
frame within o-45..o+45 with a portable GBDT and move to the best frame if its margin beats o's by delta.
Feature code mirrors candidates/s143_s144_overnight/s147_refiner.py (parity-tested on all 750 Nexar clips).
"""
import numpy as np

HALF = 45


def clip_signals(features, signals, saliency):
    feats = {k: np.asarray(features[k], np.float32).astype(np.float64) for k in signals}
    sal = np.asarray(saliency(feats), np.float64)
    out = {}
    for k in list(signals) + ['saliency']:
        x = sal if k == 'saliency' else feats[k]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        out[k] = (x - med) / mad
    return out, sal


def frame_matrix(sig, sal, n, o, p, order_names):
    frames = np.arange(max(0, o - HALF), min(n, o + HALF + 1))
    csum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in sig.items()}

    def mean(k, a, b):
        a, b = max(0, a), min(n, b)
        return (csum[k][b] - csum[k][a]) / max(b - a, 1)
    wmax = max(float(sal[frames].max()), 1e-9)
    order = np.argsort(np.argsort(-sal[frames], kind='stable'), kind='stable')
    rows = []
    for idx, j in enumerate(frames):
        r = []
        for k in order_names:
            pre, post = mean(k, j - 4, j), mean(k, j + 1, j + 5)
            r += [sig[k][j], pre, post, post - pre]
        prev = sal[j - 1] if j > 0 else sal[j]
        nxt = sal[j + 1] if j + 1 < n else sal[j]
        r += [sal[j] / wmax, float(order[idx]) / max(len(frames) - 1, 1), sal[j] - prev,
              float(sal[j] >= prev and sal[j] >= nxt), (j - o) / 30.0, (j - p) / 30.0, abs(j - o) / 30.0,
              float(j == o)]
        rows.append(r)
    return frames, np.asarray(rows, np.float64).astype(np.float32).astype(np.float64)


def margin(x, model):
    out = np.full(len(x), float(model['base_logit']))
    rate = float(model['learning_rate'])
    for tree in model['trees']:
        left, right, feature, threshold, value = tree['left'], tree['right'], tree['feature'], tree['threshold'], tree['value']
        for r in range(len(x)):
            node = 0
            while left[node] >= 0:
                node = left[node] if x[r, feature[node]] <= threshold[node] else right[node]
            out[r] += rate * value[node]
    return out


SCALES = (2, 8, 15)


def frame_matrix_v2(sig, sal, n, o, p, order_names):
    frames, x1 = frame_matrix(sig, sal, n, o, p, order_names)
    csum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in sig.items()}

    def mean(k, a, b):
        a, b = max(0, a), min(n, b)
        return (csum[k][b] - csum[k][a]) / max(b - a, 1)
    lo, hi = int(frames[0]), int(frames[-1]) + 1
    wsal = sal[lo:hi]
    local = [i for i in range(len(wsal)) if (i == 0 or wsal[i] >= wsal[i - 1]) and (i + 1 == len(wsal) or wsal[i] >= wsal[i + 1])]
    com = float(np.sum(np.arange(lo, hi) * np.maximum(wsal, 0)) / max(np.sum(np.maximum(wsal, 0)), 1e-9))
    wargmax = lo + int(np.argmax(wsal))
    rows = []
    for j in frames:
        r = []
        for k in order_names:
            for s in SCALES:
                r.append(mean(k, j + 1, j + 1 + s) - mean(k, j - s, j))
            r.append(float(np.max(sig[k][max(0, j - 8):min(n, j + 9)])))
        d2 = (sal[j + 1] if j + 1 < n else sal[j]) - 2 * sal[j] + (sal[j - 1] if j > 0 else sal[j])
        near = min(abs(j - lo - i) for i in local) if local else 0
        r += [d2, near / 30.0, (j - com) / 30.0, (j - wargmax) / 30.0]
        rows.append(r)
    x = np.concatenate([x1, np.asarray(rows, np.float64)], 1)
    return frames, x.astype(np.float32).astype(np.float64)


def refine(features, n, onset, peak, config, saliency):
    """Return (index, detail). config = refiner.json (s147_refiner_v1 or s147_refiner_v2)."""
    signals = config['signals']
    sig, sal = clip_signals(features, signals, saliency)
    if len(sal) != n:
        raise ValueError('refiner feature length mismatch')
    v2 = config['format'] == 's147_refiner_v2'
    build = frame_matrix_v2 if v2 else frame_matrix
    frames, x = build(sig, sal, n, int(onset), int(peak), signals + ['saliency'])
    s = margin(x, config['model'])
    at_o = int(np.where(frames == int(onset))[0][0])
    if v2:
        top = np.argsort(-s)[:int(config['topk'])]  # same call as training
        d = (x[top][:, None, :] - x[top][None, :, :]).reshape(-1, x.shape[1])
        d = d.astype(np.float32).astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-margin(d, config['pair_model']))).reshape(len(top), len(top))
        np.fill_diagonal(p, 0.0)
        best = int(top[int(np.argmax(p.sum(1)))])
    else:
        best = int(np.argmax(s))
    j = best if s[best] - s[at_o] > float(config['delta']) else at_o
    return int(frames[j]), {'refined_from': int(onset), 'refined_to': int(frames[j]),
                            'refine_gain': float(s[best] - s[at_o])}


def cascade(features, n, top, config, saliency):
    """top: [(onset, peak, [rank, score, gap, pct_embed, pct_top20, pct_base]), ...] best first. Returns (index, detail)."""
    signals = config['signals']
    sig, sal = clip_signals(features, signals, saliency)
    if len(sal) != n:
        raise ValueError('refiner feature length mismatch')
    frames_all, xs, owner = [], [], []
    for rank, (onset, peak, cand) in enumerate(top[:int(config['k'])]):
        frames, x = frame_matrix_v2(sig, sal, n, int(onset), int(peak), signals + ['saliency'])
        xs.append(np.concatenate([x, np.tile(np.asarray(cand, np.float64), (len(frames), 1))], 1))
        frames_all.append(frames); owner.append(np.full(len(frames), rank))
    frames = np.concatenate(frames_all)
    x = np.concatenate(xs).astype(np.float32).astype(np.float64)
    s = margin(x, config['model'])
    at_o = int(np.where((frames == int(top[0][0])) & (np.concatenate(owner) == 0))[0][0])
    order = np.argsort(-s)[:int(config['topk'])]  # same call as training
    d = (x[order][:, None, :] - x[order][None, :, :]).reshape(-1, x.shape[1]).astype(np.float32).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-margin(d, config['pair_model']))).reshape(len(order), len(order))
    np.fill_diagonal(p, 0.0)
    best = int(order[int(np.argmax(p.sum(1)))])
    j = best if s[best] - s[at_o] > float(config['delta']) else at_o
    return int(frames[j]), {'refined_from': int(top[0][0]), 'refined_to': int(frames[j]),
                            'refine_window': int(np.concatenate(owner)[j]), 'refine_gain': float(s[best] - s[at_o])}
