"""S161 crossing-only evaluation on public labels, with S156 OOF collision anchors."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path('$DATA_DIR') if os.name == 'nt' else Path('$DATA_DIR')
WORK = DATA / 's161_entry_cross'
sys.path.insert(0, str(HERE))


def rows(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def dataset():
    ccd = json.load(open(DATA / 's132_s2track/ccd_labels.json'))
    individual = {}
    for path in sorted((DATA / 'annot_s122').glob('labels_*.jsonl')):
        for r in rows(path):
            if r.get('entry_frame') is not None:
                level = 'confident' if r.get('entry_conf') in ('medium','high') else 'low'
                individual.setdefault(r['id'], []).append(dict(
                    source='ccd_'+path.stem+'_'+level, interval=[r['entry_frame']/10]*2))
    for r in ccd:
        labs = list(individual.get(r['id'], []))
        if r['entry'] is not None:
            labs.append(dict(source='ccd_consensus', interval=[r['entry']/10]*2))
        if labs:
            yield dict(id=r['id'], kind='ccd', fps=10., collision=r['s109'][0],
                       base_entry=min(r['s109'][1], r['s109'][0]),
                       labels=labs,
                       folder=str(DATA / 'ccd/frames' / r['id']))
    oof = {r['id']: r for r in json.load(open(DATA / 's144_rr3/refiner/cascade_k2_oof.json'))['0.2']}
    base = {r['id']: r['s109']['entry_frame'] for r in rows(DATA / 's126_longclip/entry_base_rows.jsonl')}
    for p in sorted((DATA / 's132_s2track/e2e').glob('*.json')):
        r = json.load(open(p))
        for k,v in r['diag']['clips'].items():
            base.setdefault(k, v['s109'][1])
    # Replay uses the identical S109 motion entry implementation. Prefer actual
    # exported S109 rows wherever available; report fallback provenance per row.
    replay = {}
    for p in sorted((DATA / 's143_reanchor').glob('nexar_s*.jsonl')):
        replay.update({r['id']:r for r in rows(p) if 'result' in r})
    labs = {}
    for r in rows(REPO / 'candidates/s126_longclip/blind_labels_frozen.jsonl'):
        if r.get('entry_time_s') is not None:
            source = 'nexar_blind_' + ('confident' if r['entry_confidence'] in ('high','medium') else 'low')
            labs.setdefault(r['id'], []).append(dict(source=source, interval=[r['entry_time_s']]*2))
    for r in json.load(open(REPO / 'tracking/s117_entry_annotations.json'))['clips']:
        iv = r['entry_interval_or_unknown']
        if iv['status'] == 'interval':
            labs.setdefault(r['clip'], []).append(dict(source='s117_interval', interval=[iv['earliest'],iv['latest']]))
    for vid, labels in sorted(labs.items()):
        cache = json.load(open(DATA / 's142_rr2/cache' / ('nexar_'+vid+'.json')))
        coll = oof[vid]['index']
        source = 'exported_s109' if vid in base else 's109_motion_replay'
        entry = base.get(vid, replay.get(vid, {}).get('result', {}).get('s109', {}).get('entry'))
        if entry is None:
            raise RuntimeError('missing S109 entry: '+vid)
        yield dict(id=vid, kind='nexar', fps=cache['fps'], collision=coll,
                   base_entry=min(entry, coll), base_entry_source=source, fold=oof[vid]['fold'],
                   labels=labels, folder=str(DATA / 's132_s2track/nexar/images' / vid), n=cache['n'])


def prepare_folder(row):
    import cv2
    folder = Path(row['folder'])
    if folder.exists():
        return folder
    folder = WORK / 'images' / row['id']
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(DATA / 'nexar_collision/train/positive' / (row['id']+'.mp4')))
    count = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        path = folder / ('frame_%06d.jpg' % count)
        if not path.exists():
            assert cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY,95])
        count += 1
    cap.release()
    assert count == row['n'], (row['id'], count, row['n'])
    return folder


def score(records):
    # Refresh annotation views when extending coverage. Predictions are reused
    # unchanged; repeated annotations of one clip are never pooled as new clips.
    labels = {(r['kind'],r['id']):r['labels'] for r in dataset()}
    out = {}
    for r in records:
        for lab in labels[(r['kind'],r['id'])]:
            s = out.setdefault(lab['source'], dict(n=0, base_hits=0, candidate_hits=0, repairs=[], breaks=[], changed=0))
            lo,hi = lab['interval']
            b = lo-.3-1e-9 <= r['base_entry']/r['fps'] <= hi+.3+1e-9
            a = lo-.3-1e-9 <= r['entry']/r['fps'] <= hi+.3+1e-9
            s['n'] += 1
            s['base_hits'] += b
            s['candidate_hits'] += a
            s['changed'] += r['entry'] != r['base_entry']
            if a and not b: s['repairs'].append(r['id'])
            if b and not a: s['breaks'].append(r['id'])
    for s in out.values():
        s['delta_137_equivalent'] = 137*(s['candidate_hits']-s['base_hits'])/s['n']
        s['delta_total_score_proxy'] = .14*(s['candidate_hits']-s['base_hits'])/s['n']
    long = [r['seconds'] for r in records if r['kind']=='nexar']
    if long:
        out['runtime'] = dict(n=len(long), mean_seconds=sum(long)/len(long),
                              max_seconds=max(long), extrapolated_137_seconds=137*sum(long)/len(long))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ids', default='')
    ap.add_argument('--use-cache',action='store_true')
    a=ap.parse_args()
    import torch
    import cv2
    import predict
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if a.device=='cuda':
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=json.load(open(HERE/'params.json'))
    p=predict.Params(**cfg['params'])
    class LazyDetector:
        actual=None
        def ready(self):
            if self.actual is None:
                self.actual=predict.Detector('frcnn', DATA/'stage2_s118/S156_casc/candidate/model/stage2/s144/detector.pth',device=a.device)
            return self.actual
        def load(self,path):return self.ready().load(path)
        def __call__(self,images):return self.ready()(images)
    det=LazyDetector()
    caches={}
    if a.use_cache:
        import pickle
        for kind in ['ccd','nexar']:
            caches[kind]=pickle.load(open(DATA/'s132_s2track'/('cache_'+kind+'_frcnn_640_0.3.pkl'),'rb'))
    target=WORK/'evaluation.jsonl'
    done={r['kind']+'/'+r['id'] for r in rows(target)} if target.exists() else set()
    wanted=set(filter(None,a.ids.split(',')))
    with open(target,'a',encoding='utf-8') as out:
        for row in dataset():
            if row['kind']+'/'+row['id'] in done or (wanted and row['id'] not in wanted): continue
            if a.device=='cuda' and os.name=='nt' and Path('$USER_HOME/Documents/code/shared_project/scratchpad/gpu.request').exists():
                print('GPU request: checkpointed before next clip',flush=True)
                break
            folder=prepare_folder(row)
            paths=sorted(folder.glob('*.jpg'))
            numbers=[int(p.stem.split('_')[-1]) for p in paths]
            ci=min(range(len(numbers)),key=lambda i:abs(numbers[i]-row['collision']))
            t=time.perf_counter()
            try:
                result=predict.analyze(paths,numbers,ci,det,p,cache=caches.get(row['kind']))
            except predict.FrameBudgetExceeded:
                result=dict(reason='frame_budget',entry_index=None)
            pos,_=predict.decide(result,'s109')
            row.update(entry=min(row['base_entry'] if pos is None else numbers[pos],row['collision']),
                       seconds=time.perf_counter()-t,result=result,folder=str(folder),
                       detector_cache_reused=a.use_cache,device=a.device)
            out.write(json.dumps(row,default=float)+'\n'); out.flush()
            print(row['kind'],row['id'],result['reason'],row['base_entry'],row['entry'],round(row['seconds'],3),flush=True)
    report=score(rows(target))
    previous=json.load(open(WORK/'metrics.json')) if (WORK/'metrics.json').exists() else {}
    report['cuda_peak_bytes']=max(previous.get('cuda_peak_bytes',0),torch.cuda.max_memory_allocated() if a.device=='cuda' else 0)
    import psutil
    memory=psutil.Process().memory_info()
    report['memory_bytes']={k:getattr(memory,k) for k in ('rss','peak_wset','peak_pagefile') if hasattr(memory,k)}
    report['coverage']={'predicted_unique_clips':len(rows(target)), 'available_unique_clips':len(list(dataset()))}
    (WORK/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
