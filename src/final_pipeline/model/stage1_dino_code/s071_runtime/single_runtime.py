"""S078 compatibility repair: ordinal sampling; optional timestamp diagnostics.
Model, feature geometry, weights and all other runtime operations are unchanged.
"""
from contextlib import contextmanager
import hashlib
import io
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import json

if __package__.startswith("candidates."):
    from candidates.s065_stage1_dino_transfer import features as f, model as m
else:
    from . import features as f, model as m
from . import canonical

BACKBONE_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
FRACTIONS = np.linspace(.1, .9, 4)
BACKEND_POLICY = {
    "matmul_tf32": False,
    "cudnn_tf32": True,
    "tensor_dtype": "float32",
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinals(count):
    if count < 1:
        raise ValueError("at least one decoded native frame required")
    indices = np.rint(FRACTIONS*(count-1)).astype(int).tolist()
    return indices


def _optional_position_ms(capture):
    """Do not promote an optional VideoCapture property to a frame contract."""
    try:
        value = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    except (cv2.error, TypeError, ValueError, OverflowError):
        return None, "query_errors"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, None

def _decode_pass(path, *, target_count=None, use_header=False, debug=False):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 4])
    frames, samples = [], []
    count, previous = 0, None
    first_pts, last_pts = None, None
    anomalies = dict(query_errors=0, nonfinite=0, nonincreasing=0)
    try:
        if not capture.isOpened():
            raise ValueError("cannot open video with the four-thread FFmpeg decoder")
        raw_header = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        header = int(raw_header) if math.isfinite(raw_header) and raw_header >= 0 and raw_header.is_integer() else None
        if use_header:
            target_count = header if header is not None and header >= 1 else None
        indices = _ordinals(target_count) if target_count is not None else []
        selected = set(indices)
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or
                bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1):
                raise ValueError("decoded frames must be native uint8 BGR with positive dimensions")
            pts, issue = _optional_position_ms(capture)
            if issue is not None:
                anomalies[issue] += 1
            if pts is not None and previous is not None and pts <= previous:
                anomalies["nonincreasing"] += 1
            if count == 0:
                first_pts = pts
            previous, last_pts = pts, pts
            # Selection is based solely on decoded frame ordinal, never on time.
            if count in selected:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                sample = dict(ordinal=count, pts_ms=pts, shape=list(rgb.shape))
                if debug:
                    sample["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
                samples.append(sample)
            count += 1
    finally:
        capture.release()
    if count < 1:
        raise ValueError(f"insufficient decoded frames: {count}; no prediction produced")
    by_ordinal = {sample["ordinal"]: (frame, sample) for frame, sample in zip(frames, samples)}
    slots = [by_ordinal[index] for index in indices if index in by_ordinal]
    frames = [frame for frame, sample in slots]
    samples = [dict(sample) for frame, sample in slots]
    return frames, dict(header_frame_count=header, target_count=target_count, decoded_count=count,
                        first_pts_ms=first_pts, last_pts_ms=last_pts, samples=samples,
                        timestamp_anomalies=anomalies)


def decode_video(path, *, debug=False):
    """Keep sequential EOF/count correction and four slots; never invent timestamps.

    Legacy pts_ms field names contain observed CAP_PROP_POS_MSEC values, not
    independently verified container PTS. Missing/nonfinite observations are null.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    frames, first = _decode_pass(path, use_header=True, debug=debug)
    passes = [first["decoded_count"]]
    timestamp_passes = [first["timestamp_anomalies"]]
    corrected = first["target_count"] != first["decoded_count"]
    final = first
    if corrected:
        frames.clear()
        frames, final = _decode_pass(path, target_count=first["decoded_count"], debug=debug)
        passes.append(final["decoded_count"])
        timestamp_passes.append(final["timestamp_anomalies"])
        if final["decoded_count"] != first["decoded_count"]:
            raise ValueError("decoded EOF count changed on the recount pass; aborted")
    if len(frames) != 4 or [s["ordinal"] for s in final["samples"]] != _ordinals(final["decoded_count"]):
        raise ValueError("incomplete four-frame native selection")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("video file changed during sequential decoding")
    return frames, dict(header_frame_count=first["header_frame_count"], decoded_count=final["decoded_count"],
                        decode_passes=len(passes), pass_decoded_counts=passes, count_corrected=corrected,
                        first_pts_ms=final["first_pts_ms"], last_pts_ms=final["last_pts_ms"],
                        samples=final["samples"], selected_frames=4, views=36, decoder_threads=4,
                        sampling_basis="decoded_frame_ordinal",
                        timestamp_source="OpenCV_CAP_PROP_POS_MSEC_not_verified_container_PTS",
                        timestamp_anomalies=final["timestamp_anomalies"],
                        pass_timestamp_anomalies=timestamp_passes)


@contextmanager
def _fp32_context(device):
    """Apply Stage A: float32 tensors, matmul TF32 off, cuDNN TF32 on.

    cuDNN may use TF32 for the DINO patch Conv2d; this is not pure IEEE FP32
    convolution. Restore both caller flags even on failure.

    Backend TF32 flags are process-wide: the embedding application must serialize
    concurrent CUDA calls that require different policies. Thread counts and
    environment settings are never modified here.
    """
    flags = None
    if device.type == "cuda":
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark)
    try:
        if flags is not None:
            torch.backends.cuda.matmul.allow_tf32 = BACKEND_POLICY["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = BACKEND_POLICY["cudnn_tf32"]
            torch.backends.cudnn.benchmark = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        if flags is not None:
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark) = flags


def _config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != "s071-runtime-v1" or config.get("head_type") not in ("mlp", "linear"):
        raise ValueError("unsupported canonical runtime config")
    expected = config.get("head_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("exact parent-manifest head SHA-256 required")
    if config.get("backbone_sha256") != BACKBONE_SHA256 or config.get("backbone_config_sha256") != CONFIG_SHA256:
        raise ValueError("frozen backbone identity mismatch")
    return config


def branch_vector(branches):
    """Float64 F,C means of float32 descriptors; native224/448/global order."""
    if set(branches) != set(f.BRANCH_NAMES):
        raise ValueError("expected exactly three descriptor branches")
    pooled = []
    for name in f.BRANCH_NAMES:
        value = branches[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != (4, f.BRANCH_CROPS[name], 3072) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("expected finite float32 four-slot branch descriptors")
        pooled.append(value.mean(axis=(0, 1), dtype=np.float64))
    return np.concatenate(pooled)


class LinearHead:
    """Training NPZ: mean/scale (9216,), coef (1,9216), intercept (1,).

    mean/scale are weighted TRAIN-only statistics; constant scales equal one.
    """
    def __init__(self, payload):
        with np.load(io.BytesIO(payload), allow_pickle=False) as state:
            if set(state.files) != {"mean", "scale", "coef", "intercept"}:
                raise ValueError("unexpected linear asset keys")
            for name in ("mean", "scale", "coef", "intercept"):
                value = state[name]
                if value.dtype != np.float64 or not np.isfinite(value).all():
                    raise ValueError("linear parameters must be finite float64")
                if name == "intercept":
                    if value.shape != (1,):
                        raise ValueError("linear intercept must be scalar")
                    self.intercept = float(value.reshape(-1)[0])
                else:
                    if value.shape != ((1, 9216) if name == "coef" else (9216,)):
                        raise ValueError("linear vector must have 9216 elements")
                    setattr(self, name, value.reshape(-1).copy())
        if np.any(self.scale <= 0):
            raise ValueError("linear std must be positive")

    def logit(self, branches):
        vector = branch_vector(branches)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = ((vector-self.mean)/self.scale)[None, :]
            logit = float(np.sum(standardized*self.coef, axis=1, dtype=np.float64)[0] + self.intercept)
        if not math.isfinite(logit):
            raise ValueError("nonfinite linear logit")
        return logit

    def probability(self, branches):
        logit = self.logit(branches)
        return float(np.exp(-np.logaddexp(0., -logit)))


class Predictor:
    def __init__(self, backbone_path, head_path, config_path, device="cuda:0"):
        backbone_path, head_path = Path(backbone_path).resolve(), Path(head_path).resolve()
        config = _config(config_path)
        for name, expected in (("model.safetensors", BACKBONE_SHA256), ("config.json", CONFIG_SHA256)):
            if _sha256_file(backbone_path/name) != expected:
                raise ValueError(f"local {name} SHA-256 mismatch")
        payload = head_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != config["head_sha256"]:
            raise ValueError("head SHA-256 mismatch")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("predictor supports explicit CPU or CUDA")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA unavailable; no CPU fallback")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        self.head_type = config["head_type"]
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
            self.head = LinearHead(payload) if self.head_type == "linear" else m.VideoHead()
            if self.head_type == "mlp":
                self.head.load_state_dict(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), strict=True)
                self.head.to(device=self.device, dtype=torch.float32).requires_grad_(False).eval()
            self.backbone = f.load_backbone(backbone_path, device=self.device)
        self.backbone.requires_grad_(False).eval()
        self.identities = dict(head_sha256=config["head_sha256"], backbone_sha256=BACKBONE_SHA256,
                               backbone_config_sha256=CONFIG_SHA256, head_type=self.head_type)

    def predict_video(self, path, debug=False):
        if not isinstance(debug, bool):
            raise ValueError("debug must be bool")
        started = time.perf_counter()
        frames, decode = decode_video(path, debug=debug)
        decoded = time.perf_counter()
        with _fp32_context(self.device), torch.no_grad():
            canonical_frames = canonical.canonicalize_frames(frames)
            branches, geometry = f.extract_frozen_features(
                canonical_frames, self.backbone, max_batch=8, output_device=self.device)
            if self.head_type == "linear":
                probability = self.head.probability(branches)
            else:
                probability = float(self.head({k: v.unsqueeze(0) for k, v in branches.items()}).sigmoid().item())
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("nonfinite/out-of-range prediction; no default produced")
            diagnostics = None
            if debug:
                diagnostics = dict(
                    selected_rgb_sha256=[s["rgb_sha256"] for s in decode["samples"]],
                    canonical_rgb_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in canonical_frames],
                    descriptor_sha256={k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                       for k, v in branches.items()}, geometry=geometry)
        finished = time.perf_counter()
        result = dict(probability=probability, prediction=int(probability >= .5), threshold=.5,
                      decode=decode, timing=dict(decode_seconds=decoded-started,
                      inference_seconds=finished-decoded, total_seconds=finished-started))
        if debug:
            result["debug"] = diagnostics
        return result
