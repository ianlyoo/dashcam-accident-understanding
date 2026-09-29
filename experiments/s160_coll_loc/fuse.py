"""S160 decode/fusion grid over saved OOF outputs (marg_<tag>.npz = GBDT margins, probs_<tag>.npz = NN probabilities).
usage: python -B fuse.py --marg hgb10 [--probs nn_base5 ...] [--k 10]
Every setting printed here is compared on the same OOF (selection is exploratory; report the grid size).
"""
import argparse
import itertools
import json

import numpy as np

import common as C


def logsm(z, M):
    z = np.where(M, z, -1e9)
    z = z - z.max((1, 2), keepdims=True)
    return z - np.log(np.exp(z).sum((1, 2), keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marg', nargs='*', default=[])
    ap.add_argument('--probs', nargs='*', default=[])
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--temps', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    ap.add_argument('--halves', type=int, nargs='+', default=[6, 9, 12])
    ap.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0])
    a = ap.parse_args()
    ids, X, M, E, FR, meta = C.load_padded(10)
    del X
    k = a.k
    M = M[:, :k]; FR = FR[:, :k]
    lg = None
    if a.marg:
        mg = np.mean([np.load(C.OUT / ('marg_%s.npz' % t))['marg'][:, :k] for t in a.marg], 0)
    if a.probs:
        pn = np.mean([np.load(C.OUT / ('probs_%s.npz' % t))['probs'][:, :k] for t in a.probs], 0)
        lpn = logsm(np.log(np.maximum(pn, 1e-12)), M)
    best = []
    for temp, half, w in itertools.product(a.temps, a.halves, a.weights if (a.marg and a.probs) else [None]):
        if a.marg and a.probs:
            z = logsm(mg / temp, M) + w * lpn
        elif a.marg:
            z = logsm(mg / temp, M)
        else:
            z = lpn / temp
        p = np.exp(logsm(z, M))
        picks = {m['id']: C.decode(p[c], FR[c], M[c], half) for c, m in enumerate(meta)}
        _, s = C.score(picks, meta)
        row = {'temp': temp, 'half': half, 'w_nn': w, 'hit_0p3': s['hit_0p3'], 'hit_1p0': s['hit_1p0'][:2]}
        print(json.dumps(row), flush=True)
        best.append(row)
    b = max(best, key=lambda r: r['hit_0p3'][0])
    print(json.dumps({'best': b, 'grid': len(best)}))


if __name__ == '__main__':
    main()
