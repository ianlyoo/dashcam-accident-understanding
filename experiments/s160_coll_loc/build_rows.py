"""S160: per-(candidate, frame) rows over the top-K E10_base candidate windows (Nexar 750, OOF candidate order).

Row features = refiner v2's 100 motion frame features (relative to that candidate's onset/peak)
             + S156's 6 candidate features (rank, rank-mean score, gap to best, member percentiles)
             + candidate-level S142 base vector (H1 34 + box 30) and S144 contact (14), plus their within-clip context.
Writes $DATA_DIR/s160_coll_loc/rows/<id>.npz (frames, rank, x, err, fold) and names.json.
usage: python -B build_rows.py --k 10 --shard 0 --shards 2
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's143_s144_overnight'))
import s147_refiner_v2 as V  # noqa: E402
import s144_features as F  # noqa: E402
from s156_cascade import candidate_order, CAND_NAMES  # noqa: E402

R, T = V.R, V.R.T
OUT = T.DATA / 's160_coll_loc'


def cand_matrix(r):
    x = np.concatenate([F.base_matrix(r), F.contact_matrix(r)], 1)
    names = list(T.BASE_NAMES) + ['contact_' + k for k in F.CONTACT_NAMES]
    x = np.concatenate([x, F.context_matrix(x, names)], 1)
    return x, names + F.context_names(names)


def build(r, S, k):
    i = r['id']
    z = np.load(R.REF / 'motion' / (i + '.npz'))
    sig, sal = R.clip_signals(z)
    order, s, (e, t, b) = candidate_order(S, i)
    cx, cnames = cand_matrix(r)
    frames_all, xs, ranks = [], [], []
    for rank, c in enumerate(order[:k]):
        cand = r['candidates'][c]
        frames, x, names = V.frame_matrix_v2(sig, sal, int(z['n']), cand['onset'], cand['peak'])
        cf = np.tile([rank, s[c], s[order[0]] - s[c], e[c], t[c], b[c]], (len(frames), 1))
        xs.append(np.concatenate([x, cf, np.tile(cx[c], (len(frames), 1))], 1))
        frames_all.append(frames); ranks.append(np.full(len(frames), rank))
    frames = np.concatenate(frames_all)
    return (frames.astype(np.int32), np.concatenate(ranks).astype(np.int8), np.concatenate(xs).astype(np.float32),
            np.abs(frames / r['fps'] - r['time_of_event']).astype(np.float32)), names + CAND_NAMES + cnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    S = json.loads((T.OUT / 'ensemble_scores.json').read_text())
    paths = sorted((T.S142 / 'cache').glob('nexar_*.json'))[a.shard::a.shards]
    t0 = time.perf_counter()
    for n, p in enumerate(paths):
        r = json.loads(p.read_text())
        dst = OUT / 'rows' / (r['id'] + '.npz')
        if dst.exists():
            continue
        (frames, rank, x, err), names = build(r, S, a.k)
        tmp = dst.with_suffix('.tmp.npz')
        np.savez(tmp, frames=frames, rank=rank, x=x, err=err, fold=T.REFERENCE[r['id']]['fold'], fps=r['fps'],
                 toe=r['time_of_event'], n=r['n'])
        os.replace(tmp, dst)
        if a.shard == 0 and not (OUT / 'names.json').exists():
            (OUT / 'names.json').write_text(json.dumps(names))
        if n % 50 == 0:
            print(json.dumps({'shard': a.shard, 'done': n, 'of': len(paths), 's': round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps({'shard': a.shard, 'finished': len(paths), 's': round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
