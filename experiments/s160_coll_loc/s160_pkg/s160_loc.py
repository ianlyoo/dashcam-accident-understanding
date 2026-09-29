"""S160 collision temporal localizer (package side; NumPy + torch CPU).

Rows = every frame of the top-K (10) E10_base candidate windows (onset +- 45 frames): refiner v2's 100 motion
features relative to that candidate, S156's 6 candidate features (rank, rank-mean score, gap to the best, member
percentiles), and the candidate's S142 base vector (H1 34 + box 30), S144 contact (14) and within-clip context (28).
Scores: HistGradientBoosting row margins (binary |t - event| <= 0.3 s) fused with a listwise temporal network
(softmax over all rows, trained to maximize the probability mass within +-0.3 s):
    log p = log_softmax(margin / temp) + w_nn * log_softmax(log mean_seeds p_nn); p = softmax(log p)
Row probabilities are summed per frame (windows overlap) and the pick is the frame with the most mass within
+-half frames. Mirrors candidates/s160_coll_loc/{build_rows,gbdt_k10,train_nn,fuse}.py (parity-tested on all 750
Nexar clips).
"""
import numpy as np

HALF = 45
W = 2 * HALF + 1


_TREES = {}


def _trees(model):
    """Padded node arrays [trees, max_nodes] (cached per model dict)."""
    key = id(model)
    if key not in _TREES:
        trees = model['trees']
        size = max(len(t['f']) for t in trees)
        arr = {}
        for k, dtype, fill in (('f', np.int64, 0), ('t', np.float64, 0.0), ('l', np.int64, 0), ('r', np.int64, 0),
                               ('v', np.float64, 0.0), ('leaf', bool, True)):
            a = np.full((len(trees), size), fill, dtype)
            for q, t in enumerate(trees):
                a[q, :len(t[k])] = t[k]
            arr[k] = a
        _TREES[key] = (model, arr)
    return _TREES[key][1]


def hgb_margin(x, model):
    """Portable HistGradientBoosting raw margin (numeric splits, no missing values); all trees advance together."""
    a = _trees(model)
    nt = a['f'].shape[0]
    tree = np.arange(nt)[:, None]
    node = np.zeros((nt, len(x)), np.int64)
    rows = np.broadcast_to(np.arange(len(x))[None, :], node.shape)
    while True:
        inner = ~a['leaf'][tree, node]
        if not inner.any():
            break
        go_left = x[rows, a['f'][tree, node]] <= a['t'][tree, node]
        node = np.where(inner, np.where(go_left, a['l'][tree, node], a['r'][tree, node]), node)
    return float(model['baseline']) + a['v'][tree, node].sum(0)


def rows(features, n, top, cand_vectors, signals, saliency, refine):
    """top: [(onset, peak, [rank, score, gap, pct_embed, pct_top20, pct_base]), ...] best first (K entries);
    cand_vectors: [K, 106] candidate base+contact+context rows in the same order. Returns padded arrays."""
    sig, sal = refine.clip_signals(features, signals, saliency)
    if len(sal) != n:
        raise ValueError('localizer feature length mismatch')
    k = len(top)
    xs, frames = [], []
    for rank, (onset, peak, cand) in enumerate(top):
        fr, x = refine.frame_matrix_v2(sig, sal, n, int(onset), int(peak), list(signals) + ['saliency'])
        c = np.concatenate([np.asarray(cand, np.float64), np.asarray(cand_vectors[rank], np.float64)])
        xs.append(np.concatenate([x, np.tile(c, (len(fr), 1))], 1).astype(np.float32))
        frames.append(fr)
    nf = xs[0].shape[1]
    X = np.zeros((k, W, nf), np.float32); M = np.zeros((k, W), bool); FR = np.full((k, W), -1, np.int64)
    for r in range(k):
        m = len(frames[r]); X[r, :m] = xs[r]; M[r, :m] = True; FR[r, :m] = frames[r]
    return X, M, FR


def nn_logits(X, M, net):
    """Listwise temporal network (train_nn.Net, conv arch) for one clip; returns row logits [K, W]."""
    import torch
    import torch.nn.functional as F
    z = (X - net['med']) / net['scale']
    z = np.sign(z) * np.log1p(np.abs(z))
    with torch.inference_mode():
        x = torch.from_numpy(z.astype(np.float32))[None]
        m = torch.from_numpy(M)[None]
        p = {k: torch.from_numpy(v) for k, v in net['params'].items()}
        h = F.gelu(F.linear(x, p['row.0.weight'], p['row.0.bias']))
        h = F.gelu(F.linear(h, p['row.3.weight'], p['row.3.bias']))
        B, K, Wn, H = h.shape
        mf = m.unsqueeze(-1).float()
        if 'c1.weight' in p:
            g = (h * mf).reshape(B * K, Wn, H).transpose(1, 2)
            g = g + F.gelu(F.conv1d(g, p['c1.weight'], p['c1.bias'], padding=2))
            g = g + F.gelu(F.conv1d(g, p['c2.weight'], p['c2.bias'], padding=4, dilation=2))
            h = g.transpose(1, 2).reshape(B, K, Wn, H)
        row = F.linear(h, p['head.weight'], p['head.bias']).squeeze(-1)
        mean = (h * mf).sum(2) / mf.sum(2).clamp(min=1)
        mx = (h - 1e4 * (1 - mf)).max(2).values
        mx = torch.where(mf.sum(2) > 0, mx, torch.zeros_like(mx))
        win = F.linear(F.gelu(F.linear(torch.cat([mean, mx], -1), p['win.0.weight'], p['win.0.bias'])),
                       p['win.2.weight'], p['win.2.bias'])
        logit = (row + win).masked_fill(~m, -1e4)
    return logit[0].double().numpy()


def log_softmax(z, M):
    z = np.where(M, z, -1e9)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def decode(p, FR, M, half):
    fr = FR[M]; pr = p[M]
    uf, inv = np.unique(fr, return_inverse=True)
    pf = np.bincount(inv, weights=pr)
    if half <= 0:
        return int(uf[np.argmax(pf)])
    cs = np.concatenate([[0.0], np.cumsum(pf)])
    lo = np.searchsorted(uf, uf - half, 'left'); hi = np.searchsorted(uf, uf + half, 'right')
    return int(uf[np.argmax(cs[hi] - cs[lo])])


def localize(X, M, FR, config, nets):
    """Fused decode for one clip. nets: list of loaded network dicts (may be empty if w_nn == 0)."""
    Xd = X.astype(np.float64)
    mg = np.mean([hgb_margin(Xd[M], m) for m in config['hgb']], 0)
    z = np.full(M.shape, -1e9); z[M] = mg / float(config['temp'])
    lp = log_softmax(z, M)
    if nets and float(config['w_nn']) > 0:
        pn = np.mean([np.exp(log_softmax(nn_logits(X, M, net), M)) for net in nets], 0)
        lp = lp + float(config['w_nn']) * log_softmax(np.log(np.maximum(pn, 1e-12)), M)
    p = np.exp(log_softmax(lp, M))
    return decode(p, FR, M, int(config['half'])), p
