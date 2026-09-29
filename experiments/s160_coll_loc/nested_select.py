"""Selection-honest check: for each fold, choose (temp, half, w_nn) on the other four folds' OOF, score the held-out fold.
usage: python -B nested_select.py --marg hgb10_big hgb10_big_s1 --probs nn_soft1 ... """
import argparse
import itertools
import json

import numpy as np

import common as C
from fuse import logsm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marg', nargs='+', required=True)
    ap.add_argument('--probs', nargs='+', required=True)
    a = ap.parse_args()
    ids, X, M, E, FR, meta = C.load_padded(10)
    del X
    mg = np.mean([np.load(C.OUT / ('marg_%s.npz' % t))['marg'] for t in a.marg], 0)
    pn = np.mean([np.load(C.OUT / ('probs_%s.npz' % t))['probs'] for t in a.probs], 0)
    lpn = logsm(np.log(np.maximum(pn, 1e-12)), M)
    grid = list(itertools.product([0.7, 1.0, 1.5], [6, 9, 12], [0.5, 1.0, 2.0]))
    folds = np.asarray([m['fold'] for m in meta])
    hit = np.zeros((len(grid), len(ids)), bool)
    for g, (temp, half, w) in enumerate(grid):
        p = np.exp(logsm(logsm(mg / temp, M) + w * lpn, M))
        for c, m in enumerate(meta):
            f = C.decode(p[c], FR[c], M[c], half)
            hit[g, c] = abs(f / m['fps'] - m['toe']) <= .3 + 1e-9
    total = 0; chosen = []
    for fold in range(5):
        g = int(np.argmax(hit[:, folds != fold].sum(1)))
        chosen.append(grid[g]); total += int(hit[g, folds == fold].sum())
    base = C.s156_oof()
    print(json.dumps({'nested_hits_0p3': total, 's156': int(sum(base[i]['error_s'] <= .3 + 1e-9 for i in ids)), 'chosen': chosen,
                      'grid_min': int(hit.sum(1).min()), 'grid_mean': float(hit.sum(1).mean()), 'grid_max': int(hit.sum(1).max()),
                      'centre_temp1_half9_w1': int(hit[grid.index((1.0, 9, 1.0))].sum())}))


if __name__ == '__main__':
    main()
