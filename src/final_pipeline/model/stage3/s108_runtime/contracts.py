"""Portable artifact and tensor contracts, independent of the workspace."""
import hashlib
import json
from pathlib import Path

BASE_BYTES = 727575668
BASE_SHA256 = '55349240fb2d0e8dafb40a3096e270130f3a3da96d54a6282141b54eecc3b441'
BASE_MEMBERS = 47
REVISION = '204698b45b3712590f06245fbfba32d3be539812'
CHECKPOINT_BYTES = 1664223428
CHECKPOINT_SHA256 = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'
ENCODER_STATE_SHA256 = '185513167d48c88cc160d21a7ca7abddf1da3ee8f0b2fc6d370207deb4bfabea'
ENCODER_FILE = {'bytes': 347379587, 'sha256': '813375a4c3622ddd5d019a38ce938fede962e07bf280df54780a49029f2194f9'}
ENCODER_KWARGS = dict(patch_size=16, img_size=(384, 384), num_frames=64,
                      tubelet_size=2, use_sdpa=True, use_SiLU=False,
                      wide_SiLU=True, uniform_power=False, use_rope=True,
                      img_temporal_dim_size=1, interpolate_rope=True)
OPERATOR = {
    'version': 's108-5hz-in-10hz-out-v1', 'block_frames': 64,
    'input_indices': 'clip(s-32+2*j,0,N-1);j=0..63',
    'keep_temporal': [8, 24], 'frames_per_row': 4,
    'pool': 'numpy-float64-6x6-mean-to-float32-row-major-4x4',
    'codec': 'float32-to-float16-to-float32',
    'head': 'S108Head', 'head_kwargs': {'frames_per_row': 4},
    'head_sequence': 'whole-file-ordered', 'decoder': 'argmax',
    'encoder_parameters': 'float32', 'encoder_autocast': 'cuda-bfloat16',
    'crop': 384, 'resize_short': 438, 'interpolation': 'cv2.INTER_LINEAR',
    'rgb_mean': [0.485, 0.456, 0.406], 'rgb_std': [0.229, 0.224, 0.225],
    'fallback': 's105-accel-and-steering-on-observer-or-head-failure',
}


class PackageError(RuntimeError):
    """Fatal asset/configuration error; never a per-file fallback."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    return {'bytes': p.stat().st_size, 'sha256': sha256(p)}


def verify(path, record):
    try:
        actual = identity(path)
    except OSError as exc:
        raise PackageError('missing or unreadable package asset: ' + str(path)) from exc
    if actual != {k: record[k] for k in ('bytes', 'sha256')}:
        raise PackageError('package identity mismatch: ' + str(path))
    return actual


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def strict_tensor_load(module, state):
    import torch
    expected = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise PackageError('state keys differ from the exact model contract')
    for key, ref in expected.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != ref.shape
                or value.dtype != ref.dtype or not torch.isfinite(value).all()):
            raise PackageError('state shape/dtype/finite mismatch: ' + key)
    module.load_state_dict(state, strict=True)


def clean_ema(checkpoint):
    import torch
    if not isinstance(checkpoint, dict) or 'ema_encoder' not in checkpoint:
        raise PackageError('checkpoint must contain ema_encoder')
    cleaned = {}
    for key, value in checkpoint['ema_encoder'].items():
        name = key.replace('module.', '').replace('backbone.', '')
        if name in cleaned or not isinstance(value, torch.Tensor):
            raise PackageError('duplicate/non-tensor EMA entry: ' + name)
        cleaned[name] = value.detach().cpu().contiguous().clone()
    if len(cleaned) != 158:
        raise PackageError('expected 158 EMA tensor keys')
    return cleaned


def validate_config(config):
    if (config.get('schema') != 's108-runtime-config-v1'
            or config.get('operator') != OPERATOR
            or config.get('base_sha256') != BASE_SHA256
            or config.get('release_ready') is not True):
        raise PackageError('final selected S108 runtime configuration required')
    if set(config.get('assets', {})) != {'encoder.pt', 'head.pt', 'selection.json'}:
        raise PackageError('encoder/head/selection asset binding required')
    if config['assets']['encoder.pt'] != ENCODER_FILE:
        raise PackageError('encoder asset differs from the reviewed EMA export')
    if set(config.get('legacy_assets', {})) != {
            'stage3_accel_mlp.npz', 'stage3_accel_robust.npz', 'stage3_temporal_ensemble.npz'}:
        raise PackageError('S105 acceleration fallback and steering asset bindings required')
    return config
