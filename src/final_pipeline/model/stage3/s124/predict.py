"""Frozen S124 steering head on the S108 cell tensor, with no encoder pass."""
import importlib
import numpy as np


def load(path, runtime):
    import torch
    from pathlib import Path
    head_module = importlib.import_module(runtime.__package__ + '.head')
    model = head_module.S108Head(n_classes=3, frames_per_row=4).float().eval()
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    runtime.strict_tensor_load(model, state) if hasattr(runtime, 'strict_tensor_load') else model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model


def probabilities_tensor(head, tensor, n):
    import torch
    with torch.no_grad():
        logits = head(tensor)
        if tuple(logits.shape) != ((n + 3) // 4, 4, 3) or not torch.isfinite(logits).all():
            raise ValueError('S124 steering head output contract differs')
        return torch.softmax(logits.float(), dim=-1).reshape(-1, 3).cpu().numpy()[:n]


def combine_steer(base, video, smooth, window, bias, w):
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3 or len(base) == 0:
        raise ValueError('nonempty Nx3 base steering probabilities required')
    if window != 20 or tuple(bias) != (-0.5, 0.0, -0.5):
        raise ValueError('exact incumbent steering decoder required')
    if w == 0.0:
        mixture = base
    else:
        video = np.asarray(video, dtype=np.float64)
        if video.shape != base.shape or not np.isfinite(video).all():
            raise ValueError('aligned finite Nx3 video steering probabilities required')
        mixture = (1.0 - w) * base + w * video
    smoothed = smooth(mixture, window)
    logits = np.log(np.maximum(smoothed, 1e-9))
    logits += np.asarray(bias, dtype=np.float64)
    return logits.argmax(axis=1).astype(np.int64)
