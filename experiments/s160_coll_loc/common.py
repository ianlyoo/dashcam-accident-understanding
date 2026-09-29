"""S160 shared loading, decoding and scoring (Nexar 750, S135/S142 folds, compared with S156 OOF)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's143_s144_overnight'))
import s144_train as T  # noqa: E402

OUT = T.DATA / 's160_coll_loc'
W = 91  # frames per candidate window (onset +- 45)


def names():
    return json.loads((OUT / 'names.json').read_text())


def load_padded(k=10):
    """Padded arrays [clips, k, W, ...]; mask marks real rows (edge windows are shorter)."""
    ids = sorted(p.stem for p in (OUT / 'rows').glob('*.npz') if not p.stem.endswith('.tmp'))
    z0 = np.load(OUT / 'rows' / (ids[0] + '.npz'))
    nf = z0['x'].shape[1]
    X = np.zeros((len(ids), k, W, nf), np.float32)
    M = np.zeros((len(ids), k, W), bool)
    E = np.full((len(ids), k, W), 99.0, np.float32)
    FR = np.full((len(ids), k, W), -1, np.int32)
    meta = []
    for c, i in enumerate(ids):
        z = np.load(OUT / 'rows' / (i + '.npz'))
        for r in range(k):
            sel = z['rank'] == r
            m = int(sel.sum())
            X[c, r, :m] = z['x'][sel]; M[c, r, :m] = True; E[c, r, :m] = z['err'][sel]; FR[c, r, :m] = z['frames'][sel]
        meta.append({'id': i, 'fold': int(z['fold']), 'fps': float(z['fps']), 'toe': float(z['toe']), 'n': int(z['n'])})
    return ids, X, M, E, FR, meta


def s156_oof():
    return {o['id']: o for o in json.loads((T.OUT / 'refiner' / 'cascade_k2_oof.json').read_text())['0.2']}


def decode(p, FR, M, half=9, k=None):
    """p: row probabilities [k, W] for one clip. Sum per frame, then pick the frame with max mass in +-half frames."""
    k = k or p.shape[0]
    fr = FR[:k][M[:k]]; pr = p[:k][M[:k]]
    uf, inv = np.unique(fr, return_inverse=True)
    pf = np.bincount(inv, weights=pr)
    if half <= 0:
        return int(uf[np.argmax(pf)])
    cs = np.concatenate([[0.0], np.cumsum(pf)])
    lo = np.searchsorted(uf, uf - half, 'left'); hi = np.searchsorted(uf, uf + half, 'right')
    mass = cs[hi] - cs[lo]
    return int(uf[np.argmax(mass)])


def score(picks, meta, base=None):
    """picks: {id: frame}. Returns hits and the paired summary vs S156 (T.summary)."""
    base = base or s156_oof()
    oof = []
    for m in meta:
        f = picks[m['id']]
        oof.append({'id': m['id'], 'fold': m['fold'], 'index': f, 'error_s': abs(f / m['fps'] - m['toe']),
                    's142_error_s': base[m['id']]['error_s'], 'candidate_min_error_s': base[m['id']]['candidate_min_error_s']})
    s = T.summary(oof)
    return oof, {k: [s[k]['new_hits'], s[k]['s142_hits'], round(s[k]['delta'] * 100, 2), [round(c * 100, 2) for c in s[k]['ci95']],
                     round(s[k]['fold_se'] * 100, 2), s[k]['clear_win'], s[k]['repaired'], s[k]['broken']] for k in ('hit_0p3', 'hit_1p0')}
