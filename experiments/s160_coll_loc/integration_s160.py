"""S160 copy of integration_s144e.py (+ --loc: independent S160 localizer reference). Original docstring:

S144s long-folder integration for the member/ensemble package (WSL + CUDA), after s142_rr2/integration.py.



Staged S144s predict_stage2 on the three public native-rate Nexar JPEG folders (S127_nexar3) vs:

  * S109 base: entry/side/evasion preserved, entry = min(S109 entry, new collision);

  * independent research path: h1_core top-20 candidates from the captured motion arrays, a fresh detector,

    research box vectors, research DINOv2 embedding code on the same JPEGs, s144_train.features() and the

    sklearn joblib refits, independent rank-mean combination -> same collision frame.

Also reports JPEG-vs-MP4 CLS cosine against the training embedding cache (diagnostic only).

Writes $DATA_DIR/s144_rr3/integration_<name>.json.

usage: python -B candidates/s143_s144_overnight/integration_s144e.py --name E10b --stage S144s_rr3

"""

import os

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):

    os.environ[k] = '1'

import argparse

import importlib.util

import json

import sys

import tempfile

import time

from pathlib import Path



import cv2

import joblib

import numpy as np

import torch



ROOT = Path(__file__).resolve().parents[2]

HERE = Path(__file__).resolve().parents[1] / 's143_s144_overnight'

LOC = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / 'candidates/s126_longclip'))

sys.path.insert(0, str(ROOT / 'candidates/s142_rr2'))

sys.path.insert(0, str(HERE))

sys.path.insert(0, str(LOC))

import h1_core as h1  # noqa: E402

import tracker_source as track  # noqa: E402

from box_features import indices, vector  # noqa: E402

from extract import images  # noqa: E402

import s144_train as T  # noqa: E402

import s144_embed_extract as E  # noqa: E402

from s144_top20_extract import candidates20  # noqa: E402

from s144_ensemble import pct  # noqa: E402



DATA = Path('$DATA_DIR')





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--name', required=True)

    ap.add_argument('--stage', default='S144s_rr3')

    ap.add_argument('--work', default=str(Path('$DATA_DIR/stage2_s118/S127_nexar3')), help='folder with images/<id>/frame_%%06d.jpg')

    ap.add_argument('--entry-first', action='store_true', help='S146: expect entry = first frame number')

    ap.add_argument('--model-name', help='joblib name (default --name)')

    ap.add_argument('--refiner', action='store_true', help='S147: independent refiner reference (s147_refiner + sklearn)')

    ap.add_argument('--refiner2', action='store_true', help='S154: refiner v2 reference (s147_refiner_v2 + sklearn)')

    ap.add_argument('--cascade', help='S156: cascade joblib name, e.g. cascade_k2_final (s156_cascade features + sklearn)')

    ap.add_argument('--loc', action='store_true', help='S160: independent localizer reference (build_rows features + sklearn HGB + torch nets)')

    a = ap.parse_args()

    torch.set_num_threads(1)

    cv2.setNumThreads(1)

    torch.cuda.set_per_process_memory_fraction(1.75e9 / torch.cuda.get_device_properties(0).total_memory)

    stage = DATA / 'stage2_s118' / a.stage / 'candidate'

    config = json.loads((stage / 'model/stage2/s144/ranker.json').read_text())

    work = Path(a.work)

    spec = importlib.util.spec_from_file_location('_s144_integration', stage / 'inference.py')

    mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(mod)

    ns = mod.__dict__

    emb = ns['_S2_COLLISION_NAMESPACE']

    hooks = (emb['locate_collision'], emb['predict_folder'])

    t = time.perf_counter()

    base = ns['_S118_BASE_PREDICT_STAGE2'](work, stage / 'model/stage2')

    tb = time.perf_counter() - t

    feats = {}

    own_extract = emb['extract_folder_features']



    def capture(folder, *args, **kwargs):

        f = own_extract(folder, *args, **kwargs)

        feats[Path(folder).name] = f

        return f

    emb['extract_folder_features'] = capture

    try:

        torch.cuda.reset_peak_memory_stats()

        t = time.perf_counter()

        out = ns['predict_stage2'](work, stage / 'model/stage2')

        tc = time.perf_counter() - t

        cuda_candidate = torch.cuda.max_memory_reserved()

    finally:

        emb['extract_folder_features'] = own_extract

    diag = ns['_S118_LAST_DIAGNOSTICS']

    assert hooks == (emb['locate_collision'], emb['predict_folder'])

    assert diag.get('single_base_call') and not any(diag.get(k) for k in (

        'rule_error', 'rerank_error', 's142_fallbacks', 's142_load_error', 's142_capture_errors')), diag

    assert base[['ID', 'entry_side', 'evasion_space']].equals(out[['ID', 'entry_side', 'evasion_space']])

    if a.entry_first:

        assert out.entry_frame.tolist() == [int(h1.MOTION.frame_numbers(work / 'images' / str(i))[0]) for i in out.ID], out

    else:

        assert out.entry_frame.tolist() == [min(int(e), int(c)) for e, c in zip(base.entry_frame, out.collision_frame)]

    members = [(m['variant'], m['k'], m.get('vote_k', m['k']),

                joblib.load(DATA / 's144_rr3' / ('final_%s_%s_k%d.joblib' % (a.model_name or a.name, m['variant'], m['k']))))

               for m in config['members']]

    kmax = max(k for _, k, _, _ in members)

    embed_k = max([k for v, k, _, _ in members if 'embed' in v], default=0)

    det = track.Detector('frcnn', DATA / 's132_s2track/weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',

                         max_side=480, score=.3)

    device = torch.device('cuda')

    dino = E.backbone(device) if embed_k else None

    tmp = Path(tempfile.mkdtemp(prefix='integration_', dir=str(DATA / 's160_coll_loc')))
    (tmp / 'embed').mkdir()

    rows = []

    for b, c in zip(base.itertuples(), out.itertuples()):

        ident = str(c.ID)

        folder = work / 'images' / ident

        paths = h1.MOTION.frame_paths(folder)

        n = len(paths)

        assert n > 310

        cands = candidates20(feats[ident], n, 30.0)[:kmax] if kmax == 20 else h1.candidate_vectors(feats[ident], n, 30.)

        sch = [indices(x['peak'], n, 30.) for x in cands]

        wanted = sorted({i for s in sch for i in s})

        ims = images(folder, wanted, det)

        detections = {}

        for i in wanted:

            boxes = det([ims[i]])[0]

            hist = np.stack([track._hist(ims[i], r) for r in boxes]) if len(boxes) else np.zeros((0, 64), np.float32)

            detections[i] = (boxes, hist)

        for x, s in zip(cands, sch):

            x['box_vector'] = vector([detections[i] for i in s], track)

        dets = {str(i): {'boxes': np.asarray(bx, np.float64).tolist()} for i, (bx, _) in detections.items()}

        cos = None

        if embed_k:

            frames = sorted({i for s in sch[:embed_k] for i in s})

            cls = E.embed(dino, device, [cv2.imread(str(paths[i])) for i in frames]).astype(np.float16)

            np.savez(tmp / 'embed' / (ident + '.npz'), frames=np.asarray(frames, np.int32), cls=cls)

            cached = DATA / 's144_rr3/embed' / (ident + '.npz')

            if cached.exists():

                z = np.load(cached)

                at = {int(f): j for j, f in enumerate(z['frames'])}

                common = [j for j, f in enumerate(frames) if f in at]

                if common:

                    u = cls[common].astype(np.float64)

                    v = np.stack([z['cls'][at[frames[j]]] for j in common]).astype(np.float64)

                    cos = float(np.mean(np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))))

        saved_out = T.OUT

        T.OUT = tmp

        try:

            votes, count = np.zeros(len(cands)), np.zeros(len(cands)); member_pcts = []

            for variant, k, vote, model in members:

                row = {'id': ident, 'n': n, 'candidates': cands[:k], 'detections': dets}

                s = model.decision_function(T.features(row, variant)[0])[:vote]

                member_pcts.append(pct(s)); votes[:vote] += member_pcts[-1]

                count[:vote] += 1

        finally:

            T.OUT = saved_out

        j = int(np.argmax(np.where(count > 0, votes / np.maximum(count, 1), -1.)))

        index = cands[j]['onset']

        refined = None

        if a.refiner:

            import s147_refiner as RF

            cfg = json.loads((stage / 'model/stage2/s144/refiner.json').read_text())

            rmodel = joblib.load(DATA / 's144_rr3/refiner/refiner_final.joblib')

            z = {k: np.asarray(v, np.float32) for k, v in feats[ident].items() if k in RF.h1.MOTION.FEATURE_NAMES}

            sig, sal = RF.clip_signals(z)

            frames, xr, _ = RF.frame_matrix(sig, sal, n, index, cands[j]['peak'])

            sr = rmodel.decision_function(xr.astype(np.float32))

            at_o = int(np.where(frames == index)[0][0]); best = int(np.argmax(sr))

            refined = int(frames[best] if sr[best] - sr[at_o] > cfg['delta'] else index)

            assert not diag['s142_clips'][ident].get('refine_error'), diag['s142_clips'][ident]

            index = refined

        if a.cascade:

            import s147_refiner as RF

            import s147_refiner_v2 as RV

            cj = joblib.load(DATA / 's144_rr3/refiner' / (a.cascade + '.joblib'))

            z = {k: np.asarray(v, np.float32) for k, v in feats[ident].items() if k in RF.h1.MOTION.FEATURE_NAMES}

            sig, sal = RF.clip_signals(z)

            e_, t_, b_ = member_pcts[0][:10], member_pcts[1][:10], member_pcts[2][:10]

            sc = e_ + t_ + b_

            order = np.argsort(-sc, kind='stable')

            assert int(order[0]) == j, (ident, order[:3], j)

            xs, fr, own = [], [], []

            for rank, cc in enumerate(order[:cj['k']]):

                f_, x_, _ = RV.frame_matrix_v2(sig, sal, n, cands[cc]['onset'], cands[cc]['peak'])

                xs.append(np.concatenate([x_, np.tile([rank, sc[cc], sc[order[0]] - sc[cc], e_[cc], t_[cc], b_[cc]], (len(f_), 1))], 1))

                fr.append(f_); own.append(np.full(len(f_), rank))

            fr = np.concatenate(fr); own = np.concatenate(own); xr = np.concatenate(xs).astype(np.float32)

            sr = cj['bin'].decision_function(xr)

            sel = np.argsort(-sr)[:RV.TOPK]

            dd = (xr[sel][:, None, :] - xr[sel][None, :, :]).reshape(-1, xr.shape[1])

            pp = cj['pair'].predict_proba(dd)[:, 1].reshape(len(sel), len(sel)); np.fill_diagonal(pp, 0.0)

            best = int(sel[int(np.argmax(pp.sum(1)))])

            at_o = int(np.where((fr == index) & (own == 0))[0][0])

            refined = int(fr[best] if sr[best] - sr[at_o] > cj['delta'] else index)

            assert not diag['s142_clips'][ident].get('refine_error'), diag['s142_clips'][ident]

            index = refined

        if a.loc:

            import s147_refiner as RF

            import s147_refiner_v2 as RV

            import build_rows as BR

            import train_nn as NN

            import s144_features as SF

            lcfg = json.loads((stage / 'model/stage2/s144/localizer.json').read_text())

            hg = joblib.load(DATA / 's160_coll_loc/final_hgb.joblib')['hgb']

            sds = torch.load(DATA / 's160_coll_loc/final_nn.pt')

            znn = np.load(stage / 'model/stage2/s144/loc_nn.npz')

            z = {k: np.asarray(v, np.float32) for k, v in feats[ident].items() if k in RF.h1.MOTION.FEATURE_NAMES}

            sig, sal = RF.clip_signals(z)

            e_, t_, b_ = member_pcts[0][:10], member_pcts[1][:10], member_pcts[2][:10]

            sc = e_ + t_ + b_

            order = np.argsort(-sc, kind='stable')

            assert int(order[0]) == j, (ident, order[:3], j)

            cx, _ = BR.cand_matrix({'n': n, 'candidates': cands[:10], 'detections': dets})

            Xl = np.zeros((10, 91, len(lcfg['feature_names'])), np.float32); Ml = np.zeros((10, 91), bool); Fl = np.full((10, 91), -1)

            for rank, cc in enumerate(order[:10]):

                f_, x_, _ = RV.frame_matrix_v2(sig, sal, n, cands[cc]['onset'], cands[cc]['peak'])

                row_ = np.concatenate([x_, np.tile([rank, sc[cc], sc[order[0]] - sc[cc], e_[cc], t_[cc], b_[cc]], (len(f_), 1)),

                                       np.tile(cx[cc], (len(f_), 1))], 1).astype(np.float32)

                Xl[rank, :len(f_)] = row_; Ml[rank, :len(f_)] = True; Fl[rank, :len(f_)] = f_

            mg = np.mean([m.decision_function(Xl[Ml].astype(np.float64)) for m in hg], 0)

            zz = np.full(Ml.shape, -1e9); zz[Ml] = mg / lcfg['temp']



            def lsm(v):

                v = np.where(Ml, v, -1e9); v = v - v.max(); return v - np.log(np.exp(v).sum())

            lp = lsm(zz)

            if lcfg['w_nn'] > 0:

                ps = []

                for q_, key_ in enumerate(lcfg['nn_keys']):

                    net = NN.Net(Xl.shape[-1], lcfg['train']['hidden'], lcfg['nn_arch'][q_], 0.2); net.load_state_dict(sds[key_]); net.eval()

                    xin = NN.prep_apply(Xl, znn[key_ + '/med'], znn[key_ + '/scale']).astype(np.float32)

                    with torch.no_grad():

                        lg = net(torch.from_numpy(xin)[None], torch.from_numpy(Ml)[None])[0].double().numpy()

                    ps.append(np.exp(lsm(lg)))

                lp = lp + lcfg['w_nn'] * lsm(np.log(np.maximum(np.mean(ps, 0), 1e-12)))

            pl = np.exp(lsm(lp))

            fr_, pr_ = Fl[Ml], pl[Ml]

            uf, inv = np.unique(fr_, return_inverse=True); pf = np.bincount(inv, weights=pr_)

            cs = np.concatenate([[0.0], np.cumsum(pf)]); h = int(lcfg['half'])

            mass = cs[np.searchsorted(uf, uf + h, 'right')] - cs[np.searchsorted(uf, uf - h, 'left')]

            clip_diag = diag['s142_clips'][ident]

            assert not clip_diag.get('loc_error') and not clip_diag.get('loc_load_error'), clip_diag

            assert clip_diag['loc_from'] == index, (ident, clip_diag, index)

            refined = int(uf[np.argmax(mass)])

            index = refined

        if a.refiner2:

            import s147_refiner as RF

            import s147_refiner_v2 as RV

            cfg = json.loads((stage / 'model/stage2/s144/refiner.json').read_text())

            mods = joblib.load(DATA / 's144_rr3/refiner/refiner_v2_final.joblib')

            z = {k: np.asarray(v, np.float32) for k, v in feats[ident].items() if k in RF.h1.MOTION.FEATURE_NAMES}

            sig, sal = RF.clip_signals(z)

            frames, xr, _ = RV.frame_matrix_v2(sig, sal, n, index, cands[j]['peak'])

            xr = xr.astype(np.float32)

            sr = mods['bin'].decision_function(xr)

            top = np.argsort(-sr)[:RV.TOPK]

            dd = (xr[top][:, None, :] - xr[top][None, :, :]).reshape(-1, xr.shape[1])

            pp = mods['pair'].predict_proba(dd)[:, 1].reshape(len(top), len(top)); np.fill_diagonal(pp, 0.0)

            best = int(top[int(np.argmax(pp.sum(1)))])

            at_o = int(np.where(frames == index)[0][0])

            refined = int(frames[best] if sr[best] - sr[at_o] > cfg['delta'] else index)

            assert cfg['format'] == 's147_refiner_v2' and not diag['s142_clips'][ident].get('refine_error'), diag['s142_clips'][ident]

            index = refined

        expected = int(h1.MOTION.frame_numbers(folder)[index])

        assert expected == int(c.collision_frame), (ident, expected, int(c.collision_frame))

        rows.append({'ID': ident, 'n': n, 's109_collision': int(b.collision_frame), 's144_collision': int(c.collision_frame),

                     'independent': expected, 'unrefined_index': int(cands[j]['onset']), 'refined_index': refined, 's109_entry': int(b.entry_frame), 'entry': int(c.entry_frame),

                     'jpeg_vs_mp4_cls_cosine': cos})

    peak = int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmHWM:')))

    result = {'passed': True, 'name': a.name, 'stage': a.stage, 'rows': rows, 'seconds_base': tb, 'seconds_candidate': tc,

              'added_per_clip': (tc - tb) / len(rows), 's142_added_per_clip_reference': 1.598712980666581,

              'rss_peak_kb': peak, 'cuda_peak_reserved_bytes_candidate': cuda_candidate,

              'diagnostics': diag, 'scope': 'Public Nexar train-positive native-rate JPEG folders (%s); no hidden evaluation files' % a.work}

    (DATA / 's160_coll_loc' / ('integration_%s%s.json' % (a.name, '' if Path(a.work).name == 'S127_nexar3' else '_' + Path(a.work).name))).write_text(json.dumps(result, indent=1, default=str))

    print(json.dumps(result, default=str))





if __name__ == '__main__':

    main()

