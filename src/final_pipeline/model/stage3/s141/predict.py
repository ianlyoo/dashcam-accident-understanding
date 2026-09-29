"""S141 steering regression, sharing S109's cached motion and V-JEPA pass."""
from pathlib import Path
import numpy as np

OFFSETS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20])


def design(motion, cells, weights):
    motion = np.asarray(motion, dtype=np.float32)
    cells = np.asarray(cells, dtype=np.float32)
    n = len(motion)
    if motion.shape != (n, 40) or cells.shape != ((n+3)//4, 16, 768) or n == 0:
        raise ValueError('S141 feature shape mismatch')
    mi = np.clip(np.arange(n)[:, None] + OFFSETS, 0, n-1)
    projected = cells.mean(1) @ weights['projection']
    vi = np.clip((np.arange(n)[:, None] + np.array([-10, 0, 10])) // 4, 0, len(projected)-1)
    x = np.concatenate([motion[mi].reshape(n,360), projected[vi].reshape(n,384)],1)
    if not np.isfinite(x).all():
        raise ValueError('S141 nonfinite input')
    return x


def continuous(x, weights):
    values=[]
    for start in range(0,len(x),2048):
        h=(np.asarray(x[start:start+2048],dtype=np.float32)-weights['mu'])/weights['sd']
        h=np.maximum(h @ weights['w1'].T+weights['b1'],0)
        h=np.maximum(h @ weights['w2'].T+weights['b2'],0)
        values.append((h @ weights['w3'].T+weights['b3'])*weights['ys']+weights['ym'])
    out=np.concatenate(values)
    if not np.isfinite(out).all(): raise ValueError('S141 nonfinite output')
    return out


def steering(values, weights):
    sign,left,right=weights['steer_thresholds']
    yaw=values[:,2]*sign
    out=np.full(len(yaw),1,dtype=np.int64)
    out[yaw>left]=0;out[yaw<-right]=2
    return out


def predict(ns, data_dir, model_dir, package_dir):
    diag={'rule_error':None,'videos':{},'regression':True}
    ns['_S118_STAGE3_DIAGNOSTICS']=diag
    base_predict=ns['_S118_BASE_PREDICT_STAGE3']
    try:
        with np.load(Path(package_dir)/'weights.npz',allow_pickle=False) as data:
            weights={k:data[k] for k in data.files}
        assert weights['w1'].shape==(128,744) and weights['projection'].shape==(768,128)
        assert np.all(weights['sd']>0)
        runtime=ns['_s108_runtime']()
        folder=Path(model_dir) if model_dir is not None else Path(package_dir).parent
        models=runtime.load_models(folder)
    except Exception as exc:
        diag['rule_error']='load %s: %s'%(type(exc).__name__,exc)
        return base_predict(data_dir,model_dir)
    extract_fn=ns['extract_video_features'];matrix_fn=ns['_s019_feature_matrix']
    active=[None];records={}

    def extract(path):
        active[0]=str(path.stem);records[active[0]]={}
        return extract_fn(path)

    def matrix(*args,**kwargs):
        value=matrix_fn(*args,**kwargs)
        if active[0] is not None: records[active[0]]['motion']=value[0]
        return value

    class Models:
        def __init__(self,base): self.base=base;self.head=self.run_heads
        def encode(self,crops): return self.base.encode(crops)
        def run_heads(self,tensor):
            result=self.base.head(tensor)
            record=records.get(active[0])
            if record is not None:
                try:
                    motion=record.pop('motion')
                    x=design(motion,tensor.detach().cpu().numpy(),weights)
                    record['steer']=steering(continuous(x,weights),weights)
                except Exception as exc:
                    record['error']='regression %s: %s'%(type(exc).__name__,exc)
            return result

    ns['extract_video_features']=extract;ns['_s019_feature_matrix']=matrix
    try:
        with ns['_S108_LOCK']:
            s109=ns['_s109_adapter']()
            original_combine=s109.combine_probabilities
            s109.combine_probabilities=combine_probabilities
            try:
                base=s109.predict(ns,data_dir,folder,runtime=runtime,models=Models(models))
            finally:
                s109.combine_probabilities=original_combine
    finally:
        ns['extract_video_features']=extract_fn;ns['_s019_feature_matrix']=matrix_fn;active[0]=None
    out=base.copy()
    for key,indices in out.groupby('ID',sort=False).indices.items():
        record=records.get(str(key),{})
        try:
            if record.get('error'): raise ValueError(record['error'])
            y=record['steer']
            if len(y)!=len(indices): raise ValueError('regression row count differs')
            out.iloc[indices,out.columns.get_loc('steer_label')]=[ns['STEER'][int(v)] for v in y]
            diag['videos'][str(key)]={'rows':len(y),'regressed':True}
        except Exception as exc:
            diag['videos'][str(key)]={'regressed':False,'error':'%s: %s'%(type(exc).__name__,exc)}
    return out


# S162 acceleration-only scoped extension; sealed S109 loader stays intact.
VIDEO_WEIGHT=0.25
EXPECTED_BIAS=(-0.25,-0.5,0.0,-0.75)

def combine_probabilities(base, video, smooth, window, bias):
    base = np.asarray(base, dtype=np.float64)
    video = np.asarray(video, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 4 or base.shape != video.shape or len(base) == 0:
        raise ValueError('aligned nonempty Nx4 probability arrays required')
    for probabilities in (base, video):
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5, rtol=0)):
            raise ValueError('invalid class probabilities')
    if window != 10 or tuple(bias) != EXPECTED_BIAS:
        raise ValueError('exact incumbent acceleration decoder required')
    mixture = (1.0 - VIDEO_WEIGHT) * base + VIDEO_WEIGHT * video
    smoothed = smooth(mixture, window)
    if smoothed.shape != mixture.shape or not np.isfinite(smoothed).all():
        raise ValueError('invalid smoothed probabilities')
    logits = np.log(np.maximum(smoothed, 1e-9)) + np.asarray(bias)
    baseline = logits.argmax(1).astype(np.int64)
    try:
        return decode(logits, **{'target': 0.4, 'cap': 0.75, 'width': 11})
    except Exception:
        # Per-clip S156 labels survive any failure in the added decoder.
        return baseline


"""Per-file bounded CONSTANT anchoring. No learned/adaptive model parameters."""
import numpy as np


def decode(logits, target=.50, cap=.75, width=1):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('finite nonempty Nx4 logits required')
    base = logits.argmax(1)
    moving = base != 3
    scores = logits.copy()
    if width > 1:
        scores = np.stack([np.convolve(np.pad(scores[:,i], width//2, mode='edge'), np.ones(width)/width, 'valid') for i in range(4)],1)
    # This arm never introduces or removes STOPPED. Only ACC/DEC/CONSTANT compete.
    if moving.any():
        margins = scores[moving,:2].max(1)-scores[moving,2]
        required = float(np.quantile(margins, target))
        bias = min(cap, max(0., required + 1e-12))
        scores[:,2] += bias
        result = scores[:,:3].argmax(1)
        result[~moving] = 3
    else:
        result = base.copy()
    return result.astype(np.int64)
