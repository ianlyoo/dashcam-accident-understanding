"""Acceleration-only adapter around the exact S105 Stage3 call chain.

The observer is per file. Only this inference module's cv2 alias is replaced;
the process-wide cv2 module and Stage1/2 callables are never modified.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from . import cache, windows5hz as windows
from .contracts import PackageError, validate_config, read_json, verify, strict_tensor_load


def prepare_crop(frame):
    """Same NumPy/OpenCV branch as S107 and the S108 training extractor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    short = 438
    oh, ow = ((int(short * h / w), short) if w < h
              else (short, int(short * w / h)))
    if (h, w) != (oh, ow):
        rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LINEAR)
    y, x = int(round((oh - 384) / 2.0)), int(round((ow - 384) / 2.0))
    # Copy releases the surrounding resized buffer, with identical pixels.
    return rgb[y:y + 384, x:x + 384].copy()


def stack_normalize(crops):
    import torch
    values = np.stack([im.transpose(2, 0, 1) for im in crops])
    tensor = (torch.from_numpy(values).float() / 255.0).permute(1, 0, 2, 3).contiguous()
    tensor = tensor.clone()
    mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)
    std = torch.as_tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)
    tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return tensor.unsqueeze(0)


def pool_encode_output(tokens):
    spatial = cache.tokens_to_spatial(tokens, 32)[8:24]
    cells = cache.spatial_to_cells(cache.pool_24_to_4(spatial))
    return cache.quantize_fp16(cells)


class Models:
    """Read-only model weights; no decoded inputs or feature cache retained."""

    def __init__(self, encoder, head, device):
        self.encoder, self.head, self.device = encoder, head, device

    def encode(self, crops):
        import torch
        tensor = stack_normalize(crops).to(self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = self.encoder(tensor)
        cells = pool_encode_output(tokens.float().cpu().numpy())
        if cells.shape != (16, 16, 768) or not np.isfinite(cells).all():
            raise ValueError('invalid pooled encoder output')
        return cells

    def classify(self, cells_fp16, n):
        import torch
        # Head sees the entire ordered sequence, including neighboring rows
        # across 64-frame block boundaries. No rowwise or blockwise Conv1d.
        tensor = torch.from_numpy(cache.dequantize_fp16(cells_fp16))
        with torch.no_grad():
            logits = self.head(tensor).cpu().numpy()
        if logits.shape != (windows.temporal_rows(n), 4, 4) or not np.isfinite(logits).all():
            raise ValueError('invalid ordered head output')
        return logits.reshape(-1, 4).argmax(axis=1).astype(np.int64)[:n]


def load_models(model_dir):
    """Fatal configuration/load boundary; no per-file try/except here."""
    import torch
    from .contracts import ENCODER_KWARGS
    from .vendor.vision_transformer import vit_base
    from .head import S108Head
    folder = Path(model_dir) / 's108'
    try:
        config = validate_config(read_json(folder / 'runtime_config.json'))
        for name, record in config['assets'].items():
            verify(folder / name, record)
        for name, record in config['legacy_assets'].items():
            verify(Path(model_dir) / name, record)
        selection = read_json(folder / 'selection.json')
        if (selection.get('schema') != 's108-parent-selection-v1'
                or selection.get('decision') != 'export-final-trained-head'
                or selection.get('head') != config['assets']['head.pt']
                or selection.get('operator') != config['operator']):
            raise PackageError('runtime selection/head/operator binding differs')
        encoder = vit_base(**ENCODER_KWARGS).float().eval()
        strict_tensor_load(encoder, torch.load(folder / 'encoder.pt', map_location='cpu', weights_only=True))
        head = S108Head(frames_per_row=4).float().eval()
        strict_tensor_load(head, torch.load(folder / 'head.pt', map_location='cpu', weights_only=True))
        for model in (encoder, head):
            model.requires_grad_(False)
        if not torch.cuda.is_available():
            raise PackageError('S108 requires CUDA; no implicit CPU inference substitution')
        encoder.to('cuda')
        # This small temporal head remains FP32 on CPU, like the training
        # cache contract. GPU is used only by the frozen video encoder.
        return Models(encoder, head, 'cuda')
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError('cannot initialize the selected S108 package') from exc


class Observer:
    def __init__(self, models, prepare=prepare_crop):
        self.models, self.prepare = models, prepare
        self.count = self.next_block = self.max_crops = 0
        self.crops, self.blocks = {}, []
        self.failure = None
        self.encoded_windows = 0

    def failed(self, error):
        self.failure = type(error).__name__
        self.crops.clear()
        self.blocks.clear()

    def add(self, frame):
        i = self.count
        self.count += 1
        if self.failure is not None:
            return
        try:
            self.crops[i] = self.prepare(frame)
            self.max_crops = max(self.max_crops, len(self.crops))
            while self.next_block + 94 < self.count:
                self.emit(windows.input_indices(self.next_block, self.count))
        except PackageError:
            raise
        except Exception as exc:
            # Disable only this file's new acceleration branch; let S105
            # finish consuming the same original frames for its steering.
            self.failed(exc)

    def emit(self, indices):
        cells = self.models.encode([self.crops[i] for i in indices])
        if cells.shape != (16, 16, 768) or cells.dtype != np.float16 or not np.isfinite(cells).all():
            raise ValueError('encoder must return exact fp16 cache rows')
        self.blocks.append(cells)
        self.encoded_windows += 1
        self.next_block += 64
        keep = max(self.next_block - 32, 0)
        self.crops = {i: v for i, v in self.crops.items() if i >= keep}

    def finish(self):
        if self.failure is not None:
            raise ValueError('observer disabled for this file: ' + self.failure)
        if self.count < 1:
            raise ValueError('empty decoded video')
        while self.next_block < self.count:
            self.emit(windows.input_indices(self.next_block, self.count))
        result = np.concatenate(self.blocks)[:windows.temporal_rows(self.count)]
        self.crops.clear()
        self.blocks.clear()
        return result


class Capture:
    def __init__(self, original, observer):
        self.original, self.observer = original, observer
        self.released = False

    def __getattr__(self, name):
        return getattr(self.original, name)

    def read(self):
        ok, frame = self.original.read()
        if ok and frame is not None:
            self.observer.add(frame)
        return ok, frame

    def release(self):
        if not self.released:
            self.released = True
            self.original.release()


class CV2Proxy:
    def __init__(self, original, observer):
        self.original, self.observer = original, observer
        self.captures = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def VideoCapture(self, *args, **kwargs):
        capture = Capture(self.original.VideoCapture(*args, **kwargs), self.observer)
        self.captures.append(capture)
        return capture


def joint_extract(namespace, extractor, path, observer):
    original = namespace['cv2']
    proxy = CV2Proxy(original, observer)
    try:
        namespace['cv2'] = proxy
        features = extractor(path)
    finally:
        namespace['cv2'] = original
        for capture in proxy.captures:
            capture.release()
    if len(proxy.captures) != 1 or any(len(v) != observer.count for v in features.values()):
        raise ValueError('shared decode/legacy feature alignment differs')
    return features


def predict(namespace, legacy_predict, data_dir, model_dir, models=None):
    """Run S105 unchanged except for successful acceleration replacements.

    Model/config errors fail before the legacy per-file catch. Per-file new
    branch errors keep the exact old S105 result. Original extraction errors
    keep S105's existing grab-count CONSTANT/STRAIGHT fallback, including its
    exceptional second counting pass. No feature data is persisted.
    """
    namespace['_S108_LAST_DIAGNOSTICS'] = {}
    # No global model cache: repeated calls cannot inherit per-file tensors.
    if models is None:
        models = load_models(model_dir)
    old_extract = namespace['extract_video_features']
    old_classify = namespace['_s024_classify_video']
    old_fallback = namespace['_fallback_frame_count']
    state = {}

    def extract(path):
        state.clear()
        state.update(path=path, started=time.perf_counter(), observer=Observer(models))
        try:
            return joint_extract(namespace, old_extract, path, state['observer'])
        except Exception as exc:
            state['legacy_error'] = type(exc).__name__
            raise

    def record(fallback, reason=None):
        obs = state.get('observer')
        diagnostic = {'fallback': fallback, 'reason': reason,
                      'decoded_frames': obs.count if obs else 0,
                      'encoded_windows': obs.encoded_windows if obs else 0,
                      'max_crops': obs.max_crops if obs else 0,
                      'seconds': time.perf_counter() - state['started']}
        namespace['_S108_LAST_DIAGNOSTICS'][str(state['path'].stem)] = diagnostic

    def classify(*args, **kwargs):
        # This calls exactly the S024 classifier under S051's alpha=1.00,
        # preserving sparse features, 2.0s smoothing, bias and steering.
        old_accel, steer, count = old_classify(*args, **kwargs)
        try:
            obs = state['observer']
            if obs.count != count:
                raise ValueError('legacy/output frame count differs')
            cells = obs.finish()
            accel = models.classify(cells, count)
            if accel.shape != (count,) or np.any((accel < 0) | (accel > 3)):
                raise ValueError('new acceleration labels violate contract')
            record(False)
            return accel, steer, count
        except Exception as exc:
            record(True, 'new_accel:' + type(exc).__name__)
            return old_accel, steer, count
        finally:
            state['observer'].crops.clear()
            state['observer'].blocks.clear()

    def fallback(path):
        record(True, 'legacy:' + state.get('legacy_error', 'classification'))
        return old_fallback(path)

    namespace['extract_video_features'] = extract
    namespace['_s024_classify_video'] = classify
    namespace['_fallback_frame_count'] = fallback
    try:
        return legacy_predict(data_dir, model_dir)
    finally:
        namespace['extract_video_features'] = old_extract
        namespace['_s024_classify_video'] = old_classify
        namespace['_fallback_frame_count'] = old_fallback
        state.clear()
