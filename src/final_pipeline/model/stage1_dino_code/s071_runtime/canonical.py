import cv2
import numpy as np
from . import features as f

def canonicalize_frames(frames):
    """Return independent RGB arrays, short edge 720, positive half ties up."""
    if isinstance(frames, np.ndarray) or not len(frames):
        raise ValueError('Expected a nonempty sequence of RGB frames')
    output = []
    for frame in frames:
        f.validate_frame(frame)
        h, w = frame.shape[:2]
        short = min(h, w)
        # Integer arithmetic gives exact half-up rounding without float ties.
        rh, rw = ((2 * size * 720 + short) // (2 * short) for size in (h, w))
        output.append(frame.copy() if short == 720 else cv2.resize(
            frame, (rw, rh), interpolation=cv2.INTER_AREA if short > 720 else cv2.INTER_LINEAR))
    return output
