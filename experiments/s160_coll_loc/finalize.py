"""S160 final fit on all 750 Nexar clips: HGB row model(s) + listwise temporal network seeds, export to s160_pkg/,
then parity: package rows (from motion + cache + E10_base percentiles) == research rows, package picks == research picks.
usage: python -B finalize.py --hgb-seeds 126 --nn-seeds 0 1 2 3 4 --temp 1.0 --half 9 --w-nn 0.25 [--device cuda]
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
import json
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
import gpu_lease
import train_nn as N
import build_rows as B

PKG = C.HERE / 's160_pkg'


def export_hgb(m):
    trees = []
    for it in m._predictors:
        nd = it[0].nodes
        assert not nd['is_categorical'].any()
        trees.append({'f': nd['feature_idx'].astype(int).tolist(), 't': nd['num_threshold'].astype(float).tolist(),
                      'l': nd['left'].astype(int).tolist(), 'r': nd['right'].astype(int).tolist(),
                      'v': nd['value'].astype(float).tolist(), 'leaf': nd['is_leaf'].astype(int).tolist()})
    return {'baseline': float(np.ravel(m._baseline_prediction)[0]), 'trees': trees}


def fit_hgb(X, M, E, a, seed):
    idx = np.arange(C.W)[None, None, :]
    keep = M & ((E <= 1.5) | (np.broadcast_to(idx, M.shape) % 3 == 0))
    x = X[keep]; e = E[keep]
    y = (e <= .3 + 1e-6).astype(np.int8)
    w = np.where(e <= 1.5, 1.0, 3.0) / np.broadcast_to(M.sum((1, 2))[:, None, None], M.shape)[keep] * 100
    m = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr_hgb, max_depth=None, max_leaf_nodes=a.leaves,
                                       min_samples_leaf=a.leaf, l2_regularization=1.0, random_state=seed)
    return m.fit(x, y, sample_weight=w)


def fit_nn(X, M, E, a, arch, seed, dev):
    tr = np.where((E <= .3 + 1e-6).any((1, 2)))[0]
    cols = list(range(X.shape[-1]))
    med, scale = N.prep_fit(X[tr], M[tr], cols)
    Xtr = torch.from_numpy(N.prep_apply(X[tr], med, scale).astype(np.float32))
    Mtr = torch.from_numpy(M[tr]); Etr = torch.from_numpy(E[tr])
    pos = (Etr <= .3 + 1e-6) & Mtr
    sig = torch.exp(-0.5 * (Etr / a.sigma) ** 2) * Mtr
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = N.Net(len(cols), a.hidden, arch, 0.2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    steps = a.epochs * int(np.ceil(len(tr) / 16))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    net.train()
    for ep in range(a.epochs):
        perm = rng.permutation(len(tr))
        for b in range(0, len(tr), 16):
            idx = torch.from_numpy(perm[b:b + 16])
            loss = N.loss_fn(net(Xtr[idx].to(dev), Mtr[idx].to(dev)), pos[idx].to(dev), a.soft, sig[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval().cpu()
    return net, med, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hgb-seeds', type=int, nargs='+', default=[126])
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--leaves', type=int, default=31)
    ap.add_argument('--leaf', type=int, default=80)
    ap.add_argument('--lr-hgb', type=float, default=.05)
    ap.add_argument('--nets', nargs='*', default=['conv:%d' % i for i in range(5)] + ['mlp:%d' % i for i in range(5)],
                    help='arch:seed per network')
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--soft', type=float, default=1.0)
    ap.add_argument('--sigma', type=float, default=0.1)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--half', type=int, default=9)
    ap.add_argument('--w-nn', type=float, default=1.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--skip-fit', action='store_true', help='reuse exported models; parity only')
    ap.add_argument('--limit', type=int, default=0, help='parity on the first N clips only')
    a = ap.parse_args()
    t0 = time.perf_counter()
    ids, X, M, E, FR, meta = C.load_padded(10)
    names = C.names()
    PKG.mkdir(exist_ok=True)
    if not a.skip_fit:
        if a.device == 'cuda':
            gpu_lease.acquire('s160-final', 1.0); torch.cuda.set_per_process_memory_fraction(0.7 / 16)
        hgbs = [fit_hgb(X, M, E, a, s) for s in a.hgb_seeds]
        print(json.dumps({'event': 'hgb', 'iters': [int(m.n_iter_) for m in hgbs], 's': round(time.perf_counter() - t0, 1)}), flush=True)
        params = {}
        nets = {}
        for q, spec in enumerate(a.nets):
            arch, seed = spec.split(':')
            net, med, scale = fit_nn(X, M, E, a, arch, int(seed), torch.device(a.device))
            nets['n%d' % q] = net.state_dict()
            for kk, v in net.state_dict().items():
                params['n%d/%s' % (q, kk)] = v.numpy()
            params['n%d/med' % q] = med; params['n%d/scale' % q] = scale
            print(json.dumps({'event': 'net', 'spec': spec, 's': round(time.perf_counter() - t0, 1)}), flush=True)
        np.savez(PKG / 'loc_nn.npz', **params)
        config = {'format': 's160_loc_v1', 'signals': list(B.R.h1.MOTION.FEATURE_NAMES), 'k': 10, 'half_window': 45,
                  'n_ranked': 10, 'member_pct_order': [0, 1, 2], 'feature_names': names, 'temp': a.temp, 'half': a.half,
                  'w_nn': a.w_nn, 'nn_keys': ['n%d' % q for q in range(len(a.nets))], 'nn_arch': [x.split(':')[0] for x in a.nets],
                  'nn_file': 'loc_nn.npz',
                  'hgb': [export_hgb(m) for m in hgbs], 'train': vars(a)}
        (PKG / 'localizer.json').write_text(json.dumps(config, separators=(',', ':')))
        import joblib
        joblib.dump({'hgb': hgbs}, C.OUT / 'final_hgb.joblib')
        torch.save(nets, C.OUT / 'final_nn.pt')
        print(json.dumps({'event': 'exported', 's': round(time.perf_counter() - t0, 1)}), flush=True)
    parity(a, ids, X, M, FR, meta, t0)


def load_pkg():
    spec = importlib.util.spec_from_file_location('_s160_loc', PKG / 's160_loc.py')
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    spec = importlib.util.spec_from_file_location('_s147_refine', C.HERE.parent / 's143_s144_overnight/s156_pkg/s147_refine.py')
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    config = json.loads((PKG / 'localizer.json').read_text())
    z = np.load(PKG / config['nn_file'])
    nets = []
    for key in config['nn_keys']:
        pre = key + '/'
        nets.append({'med': z[pre + 'med'], 'scale': z[pre + 'scale'],
                     'params': {k[len(pre):]: z[k] for k in z.files if k.startswith(pre) and k[len(pre):] not in ('med', 'scale')}})
    return P, R, config, nets


def parity(a, ids, X, M, FR, meta, t0):
    import joblib
    from s156_cascade import candidate_order
    P, R, config, nets = load_pkg()
    hgbs = joblib.load(C.OUT / 'final_hgb.joblib')['hgb']
    tnets = []
    sds = torch.load(C.OUT / 'final_nn.pt')
    for q, key in enumerate(config['nn_keys']):
        net = N.Net(X.shape[-1], config['train']['hidden'], config['nn_arch'][q], 0.2); net.load_state_dict(sds[key]); net.eval()
        tnets.append((net, nets[q]))
    S = json.loads((C.T.OUT / 'ensemble_scores.json').read_text())
    xerr = 0.0; agree = 0; hits = 0; perr = 0.0; times = []; bad = []
    n_eval = a.limit or len(ids)
    for c, i in enumerate(ids[:n_eval]):
        r = json.loads((C.T.S142 / 'cache' / ('nexar_%s.json' % i)).read_text())
        z = np.load(B.R.REF / 'motion' / (i + '.npz'))
        tt = time.perf_counter()
        order, s, (e, t, b) = candidate_order(S, i)
        cx, _ = B.cand_matrix(r)
        top = [(r['candidates'][cc]['onset'], r['candidates'][cc]['peak'], [rank, s[cc], s[order[0]] - s[cc], e[cc], t[cc], b[cc]])
               for rank, cc in enumerate(order[:10])]
        Xp, Mp, FRp = P.rows({k: z[k] for k in config['signals']}, r['n'], top, cx[order[:10]], config['signals'],
                             B.R.h1.MOTION.collision_saliency, R)
        pick, _ = P.localize(Xp, Mp, FRp, config, nets)
        times.append(time.perf_counter() - tt)
        assert (Mp == M[c]).all() and (FRp[Mp] == FR[c][M[c]]).all(), i
        xerr = max(xerr, float(np.abs(Xp[Mp] - X[c][M[c]]).max()))
        # research path: sklearn + torch modules on the cached research rows
        mg = np.mean([m.decision_function(X[c][M[c]].astype(np.float64)) for m in hgbs], 0)
        zz = np.full(M[c].shape, -1e9); zz[M[c]] = mg / config['temp']
        lp = P.log_softmax(zz, M[c])
        if tnets and config['w_nn'] > 0:
            ps = []
            for net, pk in tnets:
                xin = N.prep_apply(X[c], pk['med'], pk['scale']).astype(np.float32)
                with torch.no_grad():
                    lg = net(torch.from_numpy(xin)[None], torch.from_numpy(M[c])[None])[0].double().numpy()
                ps.append(np.exp(P.log_softmax(lg, M[c])))
                perr = max(perr, float(np.abs(lg[M[c]] - P.nn_logits(Xp, Mp, pk)[Mp]).max()))
            lp = lp + config['w_nn'] * P.log_softmax(np.log(np.maximum(np.mean(ps, 0), 1e-12)), M[c])
        ref = P.decode(np.exp(P.log_softmax(lp, M[c])), FR[c], M[c], config['half'])
        agree += int(ref == pick)
        if ref != pick:
            bad.append([i, ref, pick])
        hits += int(abs(pick / meta[c]['fps'] - meta[c]['toe']) <= .3 + 1e-9)
    rep = {'clips': n_eval, 'row_max_abs_diff': xerr, 'nn_logit_max_abs_diff': perr, 'package_vs_research_agreement': agree,
           'disagree': bad[:20], 'in_sample_hit_0p3': hits, 'package_s_per_clip_mean': float(np.mean(times)),
           'package_s_per_clip_p95': float(np.quantile(times, .95)), 'seconds': round(time.perf_counter() - t0, 1)}
    (C.OUT / 'final_parity.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
