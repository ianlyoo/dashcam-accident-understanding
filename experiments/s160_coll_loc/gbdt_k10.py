"""S160 row GBDT over the top-K windows (binary |t - event| <= 0.3 s, far negatives 1/3 with weight 3), OOF margins.
usage: python -B gbdt_k10.py --tag hgb [--k 10] [--model hgb|gbc]
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--model', choices=['hgb', 'gbc'], default='hgb')
    ap.add_argument('--feats', choices=['all', 'nocand'], default='all')
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--leaves', type=int, default=15)
    ap.add_argument('--lr', type=float, default=.05)
    ap.add_argument('--leaf', type=int, default=40)
    ap.add_argument('--seed', type=int, default=126)
    ap.add_argument('--sub', type=int, default=3, help='keep 1/sub far negatives (weight sub)')
    a = ap.parse_args()
    t0 = time.perf_counter()
    ids, X, M, E, FR, meta = C.load_padded(10)
    names = C.names()
    cols = list(range(len(names))) if a.feats == 'all' else list(range(names.index('cand_pct_base') + 1))
    k = a.k
    folds = np.asarray([m['fold'] for m in meta])
    marg = np.full(M[:, :k].shape, -50.0, np.float32)
    idx = np.arange(C.W)[None, None, :]
    for fold in range(5):
        tr = np.where(folds != fold)[0]; te = np.where(folds == fold)[0]
        m_tr = M[tr, :k]; e_tr = E[tr, :k]
        keep = m_tr & ((e_tr <= 1.5) | (np.broadcast_to(idx, m_tr.shape) % a.sub == 0))
        x = X[tr, :k][keep][:, cols]; e = e_tr[keep]
        y = (e <= .3 + 1e-6).astype(np.int8)
        w = np.where(e <= 1.5, 1.0, float(a.sub)) / np.broadcast_to(m_tr.sum((1, 2))[:, None, None], m_tr.shape)[keep] * 100
        if a.model == 'hgb':
            m = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_depth=None, max_leaf_nodes=a.leaves,
                                               min_samples_leaf=a.leaf, l2_regularization=1.0, random_state=a.seed)
        else:
            m = C.T.gbdt()
        m.fit(x, y, sample_weight=w)
        mt = M[te, :k]
        marg[te][mt] = 0  # placeholder to keep shape logic simple
        out = marg[te]
        out[mt] = m.decision_function(X[te, :k][mt][:, cols])
        marg[te] = out
        print(json.dumps({'event': 'fold', 'fold': fold, 'rows': len(x), 'pos': int(y.sum()), 'iters': int(getattr(m, 'n_iter_', 0)), 's': round(time.perf_counter() - t0, 1)}), flush=True)
    np.savez_compressed(C.OUT / ('marg_%s.npz' % a.tag), marg=marg, ids=np.asarray(ids))
    res = {}
    for half in (0, 3, 6):
        for temp in (1.0,):
            p = np.exp(marg / temp); p = p / p.sum((1, 2), keepdims=True)
            picks = {m_['id']: C.decode(p[c], FR[c], M[c], half) for c, m_ in enumerate(meta)}
            _, s = C.score(picks, meta)
            res['half%d' % half] = s
            print(json.dumps({'event': 'result', 'tag': a.tag, 'half': half, 'hit_0p3': s['hit_0p3'], 'hit_1p0': s['hit_1p0'][:2]}), flush=True)
    (C.OUT / ('summary_gbdt_%s.json' % a.tag)).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
