"""S109: per-file 75% scored motion + 25% frozen video probabilities.

Reuses the unchanged S108 shared decoder/encoder/weights and the exact S105
motion/steering graph. This is new deployment postprocessing, not new training.
"""
import numpy as np

VIDEO_WEIGHT = 0.25
EXPECTED_BIAS = (-0.25, -0.5, 0.0, -0.75)


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
    return (np.log(np.maximum(smoothed, 1e-9)) + np.asarray(bias)).argmax(1).astype(np.int64)


class AnchoredModels:
    def __init__(self, models, runtime, namespace):
        self.models, self.runtime, self.namespace = models, runtime, namespace
        self.base = None

    def clear(self):
        self.base = None

    def encode(self, crops):
        return self.models.encode(crops)

    def classify(self, cells_fp16, n):
        import torch
        base, self.base = self.base, None
        if base is None or np.shape(base) != (n, 4):
            raise ValueError('current-file incumbent probability capture missing')
        tensor = torch.from_numpy(self.runtime.cache.dequantize_fp16(cells_fp16))
        with torch.no_grad():
            logits = self.models.head(tensor)
            if tuple(logits.shape) != ((n + 3) // 4, 4, 4) or not torch.isfinite(logits).all():
                raise ValueError('video head output contract differs')
            video = torch.softmax(logits.float(), dim=-1).reshape(-1, 4).cpu().numpy()[:n]
        ns = self.namespace
        result = combine_probabilities(base, video, ns['_s015_smooth_probabilities'],
                                       int(round(ns['_S019_ACCEL_WINDOW_SECONDS'] / 0.1)),
                                       ns['_S019_ACCEL_LOG_BIAS'])
        ns['_S109_LAST_DIAGNOSTICS'].append({'n': n, 'video_weight': VIDEO_WEIGHT,
                                             'mixed': True})
        return result


def predict(namespace, data_dir, model_dir, runtime=None, models=None):
    """Scope all aliases and probability state to this call and current file.

Optional runtime/models are dependency injection for tests only. The shipped
dispatch supplies neither, so the real sealed S108 assets always load.
"""
    ns = namespace
    if (ns['_S051_ROBUST_ALPHA'] != 1.0 or ns['HIDDEN_DT_SECONDS'] != 0.1
            or tuple(ns['_S019_ACCEL_LOG_BIAS']) != EXPECTED_BIAS
            or ns['_S019_ACCEL_WINDOW_SECONDS'] != 1.0):
        raise ValueError('unexpected S105 base/decoder configuration')
    ns['_S109_LAST_DIAGNOSTICS'] = []
    runtime = runtime if runtime is not None else ns['_s108_runtime']()
    models = models if models is not None else runtime.load_models(model_dir)
    proxy = AnchoredModels(models, runtime, ns)
    original_classify = ns['_s024_classify_video']
    original_probabilities = ns['_s019_accel_probabilities']
    legacy = ns['_S108_LEGACY_PREDICT_STAGE3']

    def capture(features, dt, original_weights, robust_weights, steer_weights):
        proxy.clear()

        def probabilities(values, weights):
            result = original_probabilities(values, weights)
            if robust_weights is not None and weights is robust_weights:
                proxy.base = result
            return result

        ns['_s019_accel_probabilities'] = probabilities
        try:
            return original_classify(features, dt, original_weights, robust_weights, steer_weights)
        finally:
            ns['_s019_accel_probabilities'] = original_probabilities

    def guarded_legacy(data, folder):
        # At this point S108 has installed its observer/classification wrapper.
        # Clear our slot even when the observer fails before classify is called.
        wrapped_classify = ns['_s024_classify_video']

        def file_classify(*args, **kwargs):
            try:
                return wrapped_classify(*args, **kwargs)
            finally:
                proxy.clear()

        ns['_s024_classify_video'] = file_classify
        try:
            return legacy(data, folder)
        finally:
            ns['_s024_classify_video'] = wrapped_classify

    ns['_s024_classify_video'] = capture
    try:
        return runtime.predict(ns, guarded_legacy, data_dir, model_dir, models=proxy)
    finally:
        proxy.clear()
        ns['_s024_classify_video'] = original_classify
        ns['_s019_accel_probabilities'] = original_probabilities
