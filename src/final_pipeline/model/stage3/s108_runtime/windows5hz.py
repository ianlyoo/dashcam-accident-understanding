"""S108 5 Hz-input window operator: unchanged 10 Hz output, halved windows.

EXPLICIT PRE-TRAINING OPERATOR REVISION (not a silent change): the encoder
sees a 5 Hz subsample while outputs stay at native 10 Hz frames.

OUTPUT blocks start at s = 0, 64, 128, ... < N (64 native frames each).
The 64 model inputs use original 10 Hz indices clip(s - 32 + 2*j, 0, N-1)
for j = 0..63 (nominal span s-32 .. s+94 step 2, closer to the published
4 fps pretraining cadence than 10 Hz). Temporal indices 8..23 are kept
(16 tokens). Kept token k (window t = 8+k) nominally covers model inputs
j = 2t, 2t+1, i.e. the native SOURCE pair (s+4k, s+4k+2), with the single
nominal source-pair anchor (s+4k+1)/10. It PREDICTS the output quartet
s+4k+[0,1,2,3], masked past N. Output-label quartet times are each
frame/10; the quartet midpoint differs from the source-pair anchor.
The last block start is NOT realigned. Cache holds ceil(N/4) rows.
"""
from __future__ import annotations

FPS = 10.0
BLOCK = 64
STEP = 2
N_INPUTS = 64
SPAN_LO = 32
KEEP_LO = 8
KEEP_HI = 24
N_KEEP = KEEP_HI - KEEP_LO
ROW_FRAMES = 4
GRID = 24
POOL = 4
DIM = 768
CELLS = POOL * POOL


def block_starts(n):
    """Output block starts s = 0, 64, ... < n."""
    if n < 1:
        raise ValueError("n must be >= 1, got %r" % (n,))
    return list(range(0, n, BLOCK))


def input_indices(s, n):
    """64 subsampled model inputs clip(s - 32 + 2*j, 0, n-1)."""
    if n < 1:
        raise ValueError("n must be >= 1, got %r" % (n,))
    if s < 0 or s >= n or s % BLOCK:
        raise ValueError("bad block start s=%r for n=%r" % (s, n))
    return [min(max(s - SPAN_LO + STEP * j, 0), n - 1) for j in range(N_INPUTS)]


def input_span(s, n):
    """(min, max) native index referenced by block s inputs."""
    grid = input_indices(s, n)
    return (grid[0], grid[-1])


def output_frames_for_block(s, n):
    """Valid native output frames emitted by block s (truncated to n)."""
    if n < 1:
        raise ValueError("n must be >= 1, got %r" % (n,))
    return list(range(s, min(s + BLOCK, n)))


def temporal_rows(n):
    """Cached rows ceil(n / 4); each row predicts 4 native frames."""
    if n < 1:
        raise ValueError("n must be >= 1, got %r" % (n,))
    return (n + ROW_FRAMES - 1) // ROW_FRAMES


def head_output_mask(n):
    """(T, 4) validity of the four per-row head outputs."""
    rows = temporal_rows(n)
    mask = [[True] * ROW_FRAMES for _ in range(rows)]
    tail = n - ROW_FRAMES * (rows - 1)
    for c in range(tail, ROW_FRAMES):
        mask[rows - 1][c] = False
    return mask


def kept_token_records(s, n):
    """Per-kept-token lineage for block s: 16 records with source pair,
    single nominal anchor (s+4k+1)/10, actual clamped source spans, and
    the predicted output quartet with valid-only sensor times."""
    grid = input_indices(s, n)
    replicated = (len(set(grid)) != N_INPUTS)
    records = []
    for k in range(N_KEEP):
        nominal_pair = [s + ROW_FRAMES * k, s + ROW_FRAMES * k + 2]
        actual = sorted({min(max(f, 0), n - 1) for f in nominal_pair})
        quartet = [s + ROW_FRAMES * k + c for c in range(ROW_FRAMES)]
        kept = [f for f in quartet if f < n]
        records.append({
            "block_s": s,
            "keep_k": k,
            "window_t": KEEP_LO + k,
            "nominal_source_pair": nominal_pair,
            "nominal_source_anchor_s": (s + ROW_FRAMES * k + 1) / FPS,
            "actual_source_indices": actual,
            "actual_source_times_s": [f / FPS for f in actual],
            "out_quartet": quartet,
            "valid_frames": kept,
            "valid_times_s": [f / FPS for f in kept],
            "valid": bool(kept),
            "padded": len(kept) != ROW_FRAMES,
            "grid_replicated": replicated,
        })
    return records


def coverage_frames(n):
    """Concatenated valid output frames across blocks, in emission order."""
    out = []
    for s in block_starts(n):
        out.extend(output_frames_for_block(s, n))
    return out


def check_coverage(n):
    """Raise unless every frame 0..n-1 is covered exactly once, in order."""
    cov = coverage_frames(n)
    if cov != list(range(n)):
        raise ValueError("coverage mismatch for n=%r" % (n,))
    if temporal_rows(n) != (n + ROW_FRAMES - 1) // ROW_FRAMES:
        raise ValueError("temporal row count mismatch for n=%r" % (n,))
    return True
