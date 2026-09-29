"""Native multiscale DINOv2 descriptors. No decoder, fitting, or network I/O.

Coordinates are (top, left) in the symmetrically padded image. Half ties round
up. Crop order is top-left, top-right, bottom-left, bottom-right. Receipts are
JSON-compatible; callers must enforce their native-resolution acceptance gate.
"""
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn

MODEL_PATH = Path("$DATA_DIR/pretrained/dinov2-base-f9e44c8")
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
BLOCK_INDICES = (2, 5, 8, 11)
HIDDEN_STATE_INDICES = (3, 6, 9, 12)
FEATURE_DIM = 6144
BRANCH_NAMES = ("native224", "native448", "global")
BRANCH_CROPS = {"native224": 4, "native448": 4, "global": 1}
NATIVE_BRANCH_INDICES = (0, 1)
FULL_FRAME_BRANCH_INDEX = 2
QUADRANT_CENTERS = ((.25, .25), (.25, .75), (.75, .25), (.75, .75))
IMAGENET_MEAN = np.asarray((.485, .456, .406), dtype=np.float32)
IMAGENET_STD = np.asarray((.229, .224, .225), dtype=np.float32)


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB ndarrays")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("frame must have nonempty shape (H,W,3)")


def _round(value: float) -> int:
    return int(np.floor(value + .5))


def native_crops(frame: np.ndarray, size: int, *, jitter: float = 0.,
                 rng: np.random.Generator | None = None) -> tuple[list[np.ndarray], dict]:
    validate_frame(frame)
    if size not in (224, 448):
        raise ValueError("native size must be 224 or 448")
    if not np.isfinite(jitter) or not 0 <= jitter <= .1:
        raise ValueError("jitter must be in [0,.1]")
    if jitter and not isinstance(rng, np.random.Generator):
        raise ValueError("jitter requires an explicit numpy Generator")
    h, w = frame.shape[:2]
    dh, dw = max(0, size-h), max(0, size-w)
    pad = (dh//2, dh-dh//2, dw//2, dw-dw//2)
    # symmetric includes the boundary pixel and handles one-pixel dimensions.
    base = np.pad(frame, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)),
                  mode="symmetric") if dh or dw else frame
    available = np.asarray((base.shape[0]-size, base.shape[1]-size))
    coords, crops = [], []
    for center in QUADRANT_CENTERS:
        fraction = np.asarray(center)
        if jitter:
            fraction = fraction + rng.uniform(-jitter, jitter, size=2)
        y, x = [_round(float(v)) for v in np.clip(fraction, 0, 1)*available]
        coords.append([y, x])
        crops.append(base[y:y+size, x:x+size].copy())
    return crops, {"input_hw": [h, w], "size": size, "coords_yx": coords,
                   "padding_tblr": list(pad), "padded": bool(dh or dw),
                   "padding_mode": "symmetric", "resized": False}


def global_view(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Shrink long edge to 448; smaller inputs retain native size, centered.

    Output is float32 RGB in [0,1], allowing exact ImageNet-mean padding.
    """
    validate_frame(frame)
    h, w = frame.shape[:2]
    scale = min(1., 448/max(h, w))
    rh, rw = max(1, _round(h*scale)), max(1, _round(w*scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    top, left = (448-rh)//2, (448-rw)//2
    output = np.empty((448, 448, 3), dtype=np.float32)
    output[...] = IMAGENET_MEAN
    output[top:top+rh, left:left+rw] = resized.astype(np.float32)/255.
    return output, {"input_hw": [h, w], "resized_hw": [rh, rw],
                    "scale": scale, "effective_scale_yx": [rh/h, rw/w],
                    "padding_tblr": [top, 448-rh-top, left, 448-rw-left],
                    "interpolation": "INTER_AREA" if scale < 1 else "none",
                    "upscaled": False, "padding_rgb": IMAGENET_MEAN.tolist()}


def normalize_views(views: Sequence[np.ndarray]) -> torch.Tensor:
    """uint8 RGB or float32 unit RGB -> finite float32 NCHW tensor on CPU."""
    if not len(views):
        raise ValueError("empty views")
    prepared = []
    for view in views:
        if not isinstance(view, np.ndarray) or view.ndim != 3 or view.shape[-1] != 3:
            raise ValueError("views must have shape (H,W,3)")
        if min(view.shape[:2]) < 1 or view.dtype not in (np.uint8, np.float32):
            raise ValueError("views must be nonempty uint8 or float32")
        value = view.astype(np.float32, copy=True)
        if view.dtype == np.uint8:
            value /= 255.
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("nonfinite or out-of-range RGB")
        prepared.append((value-IMAGENET_MEAN)/IMAGENET_STD)
    if len({v.shape for v in prepared}) != 1:
        raise ValueError("a model chunk must contain one shape")
    return torch.from_numpy(np.stack(prepared).transpose(0, 3, 1, 2).copy())


def augment_frame(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-independent HFlip/brightness/saturation/grayscale; source untouched."""
    validate_frame(frame)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("augmentation requires an explicit numpy Generator")
    value = frame.astype(np.float32)
    if rng.random() < .5:
        value = value[:, ::-1].copy()
    value *= rng.uniform(.8, 1.2)
    gray = np.sum(value*np.asarray((.299, .587, .114), np.float32), axis=-1, keepdims=True)
    value = gray + rng.uniform(.8, 1.2)*(value-gray)
    if rng.random() < .2:
        value = np.broadcast_to(gray, value.shape)
    return np.clip(np.floor(value+.5), 0, 255).astype(np.uint8)


def load_backbone(path: str | Path = MODEL_PATH, *, device: str | torch.device = "cpu") -> nn.Module:
    """Fail closed before invoking HF; never falls back to Hub/cache identifiers."""
    root = Path(path).resolve()
    for name in ("config.json", "model.safetensors"):
        if not (root/name).is_file():
            raise FileNotFoundError(root/name)
    import json
    validate_backbone_config(json.loads((root / "config.json").read_text(encoding="utf-8")))
    from transformers import Dinov2Model
    backbone, loading = Dinov2Model.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True)
    if any(loading.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("incomplete or incompatible DINOv2-base checkpoint")
    validate_backbone_config(backbone.config.to_dict())
    return backbone.requires_grad_(False).eval().to(device=device, dtype=torch.float32)


def validate_backbone_config(config: dict) -> None:
    """Require the ordinary one-CLS, no-register Base graph before loading."""
    expected = dict(model_type="dinov2", hidden_size=768, num_hidden_layers=12,
                    patch_size=14, num_attention_heads=12, num_channels=3,
                    use_swiglu_ffn=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("expected ordinary DINOv2-base 768/12/14/12 configuration")
    if config.get("architectures") != ["Dinov2Model"] or config.get("num_register_tokens", 0) != 0:
        raise ValueError("expected Dinov2Model with one CLS and no register tokens")


def summarize_hidden_states(hidden_states: Sequence[torch.Tensor], layernorm: nn.Module,
                            *, expected_token_count: int,
                            expected_batch_size: int) -> torch.Tensor:
    if len(hidden_states) != 13:
        raise ValueError("expected embedding plus 12 DINO block hidden states")
    if (type(expected_token_count) is not int or expected_token_count < 2
            or type(expected_batch_size) is not int or expected_batch_size < 1):
        raise ValueError("expected positive batch and CLS plus patch tokens")
    shape = (expected_batch_size, expected_token_count, 768)
    if any(not isinstance(state, torch.Tensor) or tuple(state.shape) != shape
           for state in hidden_states):
        raise ValueError("hidden states must match exact batch/token/768 shape")
    summaries = []
    for index in HIDDEN_STATE_INDICES:
        patches = layernorm(hidden_states[index])[:, 1:, :]
        summaries.extend((patches.mean(dim=1), patches.std(dim=1, unbiased=False)))
    result = torch.cat(summaries, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite DINO descriptors")
    return result


def extract_features(frames: Sequence[np.ndarray], backbone: nn.Module, *, max_batch: int = 8,
                     device: str | torch.device | None = None, jitter: float = 0.,
                     rng: np.random.Generator | None = None,
                     output_device: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Differentiable shared path. Returns (branches, per-frame geometry receipts).

    Each tensor is (F,C,6144). Bounded same-shape chunks are sent to the model;
    retaining autograd retains activations, so callers must bound video length
    for adapter training. Frozen extraction should use extract_frozen_features.
    """
    if isinstance(frames, np.ndarray) or not len(frames):
        raise ValueError("provide a nonempty sequence of RGB frame ndarrays")
    if not isinstance(max_batch, int) or isinstance(max_batch, bool) or not 1 <= max_batch <= 8:
        raise ValueError("max_batch must be an integer in [1,8]")
    for frame in frames:
        validate_frame(frame)
    if not np.isfinite(jitter) or not 0 <= jitter <= .1 or (jitter and rng is None):
        raise ValueError("jitter requires range [0,.1] and explicit Generator")
    parameter = next(backbone.parameters(), None)
    target = torch.device(device) if device is not None else (parameter.device if parameter is not None else torch.device("cpu"))
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if parameter is not None and (parameter.device != target or parameter.dtype != torch.float32):
        raise ValueError("backbone must already be float32 on the requested device")
    receipts = [{} for _ in frames]
    result = {}
    for branch in BRANCH_NAMES:
        pending, outputs = [], []

        def flush() -> None:
            pixels = normalize_views(pending).to(target)
            with torch.autocast(device_type=target.type, enabled=False):
                hidden = backbone(pixel_values=pixels, output_hidden_states=True, return_dict=True).hidden_states
                outputs.append(summarize_hidden_states(
                    hidden, backbone.layernorm,
                    expected_token_count=(pixels.shape[-2] // 14) * (pixels.shape[-1] // 14) + 1,
                    expected_batch_size=pixels.shape[0]).to(output_device))
            pending.clear()

        for index, frame in enumerate(frames):
            if branch == "global":
                view, receipt = global_view(frame)
                views = [view]
            else:
                views, receipt = native_crops(frame, int(branch[6:]), jitter=jitter, rng=rng)
            receipts[index][branch] = receipt
            for view in views:
                pending.append(view)
                if len(pending) == max_batch:
                    flush()
        if pending:
            flush()
        result[branch] = torch.cat(outputs).reshape(len(frames), BRANCH_CROPS[branch], FEATURE_DIM)
    return result, receipts


def extract_frozen_features(frames: Sequence[np.ndarray], backbone: nn.Module, **kwargs):
    """No-grad outputs remain ordinary tensors usable as head-training inputs."""
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("frozen extraction requires all backbone parameters frozen")
    if backbone.training:
        raise ValueError("frozen extraction requires backbone.eval()")
    with torch.no_grad():
        return extract_features(frames, backbone, **kwargs)
