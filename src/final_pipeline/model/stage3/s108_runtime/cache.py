"""S108 pooled cache: 24x24 -> 4x4 mean pooling plus exact fp16 codec."""
from __future__ import annotations

import numpy as np

GRID = 24
POOL = 4
FACTOR = GRID // POOL
DIM = 768
CELLS = POOL * POOL
FEATURES_PER_TOKEN = CELLS * DIM
BYTES_PER_TOKEN_FP16 = FEATURES_PER_TOKEN * 2


def check_spatial(spatial, t=None):
    arr = np.asarray(spatial)
    if arr.ndim != 4 or arr.shape[1] != GRID or arr.shape[2] != GRID:
        raise ValueError("spatial must be (T,24,24,D), got %r" % (arr.shape,))
    if t is not None and arr.shape[0] != t:
        raise ValueError("T %d != %d" % (arr.shape[0], t))
    if arr.shape[3] != DIM:
        raise ValueError("dim %d != 768" % arr.shape[3])
    return arr


def pool_24_to_4(spatial):
    """Canonical mean pool (T,24,24,768) -> (T,4,4,768); 6x6 block means."""
    arr = check_spatial(spatial).astype(np.float64)
    t, _, _, d = arr.shape
    blocked = arr.reshape(t, POOL, FACTOR, POOL, FACTOR, d)
    return blocked.mean(axis=(2, 4)).astype(np.float32)


def spatial_to_cells(pooled):
    """(T,4,4,768) -> (T,16,768) row-major cell order; exact inverse below."""
    arr = np.asarray(pooled)
    if arr.ndim != 4 or arr.shape[1] != POOL or arr.shape[2] != POOL:
        raise ValueError("pooled must be (T,4,4,D), got %r" % (arr.shape,))
    t = arr.shape[0]
    return arr.reshape(t, CELLS, arr.shape[3])


def cells_to_spatial(cells):
    arr = np.asarray(cells)
    if arr.ndim != 3 or arr.shape[1] != CELLS or arr.shape[2] != DIM:
        raise ValueError("cells must be (T,16,768), got %r" % (arr.shape,))
    return arr.reshape(arr.shape[0], POOL, POOL, DIM)


def quantize_fp16(cells_f32):
    """Exact cache encoder: float32 -> float16 (round-half-to-even)."""
    return np.asarray(cells_f32, dtype=np.float32).astype(np.float16)


def dequantize_fp16(cells_f16):
    """Exact cache decoder: float16 -> float32. Must reuse this at deploy."""
    return np.asarray(cells_f16, dtype=np.float16).astype(np.float32)


def tokens_to_spatial(tokens, t):
    """(1,T*576,768) encoder rows -> (T,24,24,768)."""
    arr = np.asarray(tokens)
    if arr.shape != (1, t * GRID * GRID, DIM):
        raise ValueError("tokens %r != (1,%d,768)" % (arr.shape, t * GRID * GRID))
    return arr.reshape(t, GRID, GRID, DIM)


def cache_bytes(n_frames):
    """Projected fp16 cache bytes for n native frames: ceil(n/2) rows."""
    rows = (int(n_frames) + 1) // 2
    return rows * BYTES_PER_TOKEN_FP16
