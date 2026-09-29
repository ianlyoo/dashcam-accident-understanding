"""Record the already-selected export and a fixed, explicitly incomplete OOF proxy."""
import hashlib
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path('$DATA_DIR') if os.name == 'nt' else Path('$DATA_DIR')
WORK = DATA / 's160_coll_loc'


def logsm(z, mask):
    z = np.where(mask, z, -1e9)
    z = z - z.max((1, 2), keepdims=True)
    return z - np.log(np.exp(z).sum((1, 2), keepdims=True))


def main():
    cfg = json.loads((HERE / 's160_pkg/localizer.json').read_text())
    summaries = {}
    for path in sorted(WORK.glob('summary_*.json')):
        value = json.loads(path.read_text())
        summaries[path.stem] = {k: v['hit_0p3'][0] for k,v in value.items() if k.startswith('half')}
    ids = np.load(WORK / 'probs_nn_soft1.npz')['ids'].tolist()
    mask = np.zeros((len(ids), 10, 91), bool)
    frames = np.full(mask.shape, -1, np.int32)
    meta = []
    for c, ident in enumerate(ids):
        with np.load(WORK / 'rows' / (ident + '.npz')) as z:
            ranks, fr = z['rank'], z['frames']
            for rank in range(10):
                selected = fr[ranks == rank]
                mask[c, rank, :len(selected)] = True
                frames[c, rank, :len(selected)] = selected
            meta.append((float(z['fps']), float(z['toe'])))
    def load(tag, kind):
        with np.load(WORK / (kind + '_' + tag + '.npz')) as z:
            if 'ids' in z:
                assert z['ids'].tolist() == ids
            return z['probs' if kind == 'probs' else 'marg'].astype(np.float64)
    conv5 = (3 * load('nn_soft1', 'probs') + 2 * load('nn_soft1b', 'probs')) / 5
    pn = (conv5 + load('nn_soft1_mlp', 'probs')) / 2
    mg = (load('hgb10_big', 'marg') + load('hgb10_big_s1', 'marg')) / 2
    p = np.exp(logsm(logsm(mg, mask) + logsm(np.log(np.maximum(pn, 1e-12)), mask), mask))
    hits = 0
    picks = []
    for c, ident in enumerate(ids):
        fr, inverse = np.unique(frames[c][mask[c]], return_inverse=True)
        pf = np.bincount(inverse, weights=p[c][mask[c]])
        cs = np.concatenate([[0.], np.cumsum(pf)])
        mass = cs[np.searchsorted(fr, fr+9, 'right')] - cs[np.searchsorted(fr, fr-9, 'left')]
        pick = int(fr[np.argmax(mass)])
        error = abs(pick / meta[c][0] - meta[c][1])
        hits += error <= .3 + 1e-9
        picks.append({'id': ident, 'frame': pick, 'error_s': error})
    rec = dict(selected='exported HGB(126,1) + Conv(0..4)/MLP(0..4), 25 epochs, temp=1, w_nn=1, half=9',
               reason='finalize.log, localizer.json train configuration and exported state agree; retain completed selected export',
               handoff_correction='nn_soft1_ep40 half0/3/6/9 are separate decoding settings, not an ensemble and not the exported final fit',
               config={k:v for k,v in cfg.items() if k not in ('hgb','feature_names')},
               final_parity=json.loads((WORK / 'final_parity.json').read_text()), summaries=summaries,
               exact_final_ensemble_oof_available=False,
               fixed_oof_proxy=dict(hits_0p3=int(hits), clips=len(ids), s156_hits_0p3=393,
                   description='Two matching HGB OOFs; five Conv seeds; only three MLP seeds available, weighted 50/50 by architecture. Final fit also uses MLP seeds 3,4. This is a proxy, not exact final-ensemble OOF; no new sweep.',
                   weight_decay='Both finalize.py and OOF train_nn.py explicitly use .001'),
               official_gain_measured=False,
               exported_weights={name: hashlib.sha256((HERE / 's160_pkg' / name).read_bytes()).hexdigest()
                                 for name in ('loc_nn.npz','localizer.json')})
    (HERE / 'selection.json').write_text(json.dumps(rec, indent=2))
    print(json.dumps({'selection':rec['selected'], 'fixed_oof_proxy':rec['fixed_oof_proxy'], 'summaries':summaries}), flush=True)


if __name__ == '__main__':
    main()
