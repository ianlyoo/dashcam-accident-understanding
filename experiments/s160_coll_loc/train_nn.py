"""S160 temporal localizer: listwise softmax over all rows of the top-K candidate windows, trained to maximize the
probability mass within +-0.3 s of the label; decode = frame with max mass in +-half frames.
Same five grouped folds as S144/S156; OOF compared with S156 (393/750 at 0.3 s).
usage: python -B train_nn.py --tag base --k 10 --seeds 0 1 2 [--epochs 40] [--soft 0.0] [--arch conv]
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import gpu_lease

torch.set_num_threads(2)


def prep_fit(X, M, cols):
    x = X[..., cols][M]
    med = np.median(x, 0); iqr = np.quantile(x, .75, 0) - np.quantile(x, .25, 0)
    sd = x.std(0)
    scale = np.where(iqr > 1e-6, iqr, np.where(sd > 1e-6, sd, 1.0))
    return med.astype(np.float32), scale.astype(np.float32)


def prep_apply(x, med, scale):
    z = (x - med) / scale
    return np.sign(z) * np.log1p(np.abs(z))


class Net(nn.Module):
    def __init__(self, nf, h=96, arch='conv', drop=0.2):
        super().__init__()
        self.arch = arch
        self.row = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h), nn.GELU())
        if arch == 'conv':
            self.c1 = nn.Conv1d(h, h, 5, padding=2)
            self.c2 = nn.Conv1d(h, h, 5, padding=4, dilation=2)
        self.head = nn.Linear(h, 1)
        self.win = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x, m):  # x [B,K,W,F], m [B,K,W]
        B, K, Wn, F = x.shape
        h = self.row(x)  # [B,K,W,h]
        mf = m.unsqueeze(-1).float()
        if self.arch == 'conv':
            g = (h * mf).reshape(B * K, Wn, -1).transpose(1, 2)
            g = g + torch.nn.functional.gelu(self.c1(g))
            g = g + torch.nn.functional.gelu(self.c2(g))
            h = g.transpose(1, 2).reshape(B, K, Wn, -1)
        h = self.drop(h)
        row = self.head(h).squeeze(-1)
        mean = (h * mf).sum(2) / mf.sum(2).clamp(min=1)
        mx = (h - 1e4 * (1 - mf)).max(2).values
        mx = torch.where(mf.sum(2) > 0, mx, torch.zeros_like(mx))
        win = self.win(torch.cat([mean, mx], -1))  # [B,K,1]
        logit = row + win
        return logit.masked_fill(~m, -1e4)


def loss_fn(logit, pos, soft, sigma_w):
    lp = torch.log_softmax(logit.flatten(1), 1)
    posf = pos.flatten(1)
    mass = torch.logsumexp(lp.masked_fill(~posf, -1e4), 1)
    loss = -mass
    if soft > 0:
        t = sigma_w.flatten(1); t = t / t.sum(1, keepdim=True).clamp(min=1e-9)
        loss = loss + soft * -(t * lp).sum(1)
    return loss.mean()


def run(a, X, M, E, FR, meta, cols):
    folds = np.asarray([m['fold'] for m in meta])
    k = a.k
    probs = np.zeros(M[:, :k].shape, np.float64)
    for fold in range(5):
        tr = np.where(folds != fold)[0]; te = np.where(folds == fold)[0]
        tr = tr[(E[tr, :k] <= .3 + 1e-6).any((1, 2))]  # clips with a positive row in the top-k windows
        med, scale = prep_fit(X[tr][:, :k], M[tr][:, :k], cols)
        Xtr = torch.from_numpy(prep_apply(X[tr][:, :k][..., cols], med, scale).astype(np.float32))
        Xte = torch.from_numpy(prep_apply(X[te][:, :k][..., cols], med, scale).astype(np.float32))
        Mtr = torch.from_numpy(M[tr][:, :k]); Mte = torch.from_numpy(M[te][:, :k])
        Etr = torch.from_numpy(E[tr][:, :k])
        pos = (Etr <= .3 + 1e-6) & Mtr
        sig = torch.exp(-0.5 * (Etr / a.sigma) ** 2) * Mtr
        dev = torch.device(a.device)
        for seed in a.seeds:
            if a.device == 'cuda':
                gpu_lease.wait_request()
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            net = Net(len(cols), a.hidden, a.arch, a.drop).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            steps = a.epochs * int(np.ceil(len(tr) / a.batch))
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
            net.train()
            for ep in range(a.epochs):
                perm = rng.permutation(len(tr))
                for b in range(0, len(tr), a.batch):
                    idx = torch.from_numpy(perm[b:b + a.batch])
                    xb = Xtr[idx].to(dev, non_blocking=True)
                    if a.noise > 0:
                        xb = xb + a.noise * torch.randn_like(xb)
                    loss = loss_fn(net(xb, Mtr[idx].to(dev)), pos[idx].to(dev), a.soft, sig[idx].to(dev))
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad():
                p = np.concatenate([torch.softmax(net(Xte[b:b + 32].to(dev), Mte[b:b + 32].to(dev)).flatten(1), 1)
                                    .reshape(Mte[b:b + 32].shape).double().cpu().numpy() for b in range(0, len(te), 32)])
            probs[te] += p / len(a.seeds)
        print(json.dumps({'event': 'fold', 'fold': fold, 'train': len(tr), 's': round(time.perf_counter() - T0, 1)}), flush=True)
    return probs


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--soft', type=float, default=0.0)
    ap.add_argument('--sigma', type=float, default=0.15)
    ap.add_argument('--arch', choices=['conv', 'mlp'], default='conv')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--feats', choices=['all', 'nocand'], default='all')
    a = ap.parse_args()
    T0 = time.perf_counter()
    if a.device == 'cuda':
        gpu_lease.acquire('s160-' + a.tag, 1.0)
        torch.cuda.set_per_process_memory_fraction(0.7 / 16)
    ids, X, M, E, FR, meta = C.load_padded(10)
    names = C.names()
    cols = list(range(len(names))) if a.feats == 'all' else list(range(names.index('cand_pct_base') + 1))
    print(json.dumps({'event': 'loaded', 'clips': len(ids), 'features': len(cols), 's': round(time.perf_counter() - T0, 1)}), flush=True)
    probs = run(a, X, M, E, FR, meta, cols)
    np.savez_compressed(C.OUT / ('probs_%s.npz' % a.tag), probs=probs.astype(np.float32), ids=np.asarray(ids))
    res = {'args': vars(a)}
    for half in (0, 3, 6, 9):
        picks = {m['id']: C.decode(probs[c], FR[c], M[c], half) for c, m in enumerate(meta)}
        oof, s = C.score(picks, meta)
        res['half%d' % half] = s
        print(json.dumps({'event': 'result', 'tag': a.tag, 'half': half, 'hit_0p3': s['hit_0p3'], 'hit_1p0': s['hit_1p0'][:2]}), flush=True)
    (C.OUT / ('summary_%s.json' % a.tag)).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
