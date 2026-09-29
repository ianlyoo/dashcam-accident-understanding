# BASELINE_INFERENCE_PART
"""3-Stage 영상 분석 베이스라인 추론 코드.

각 Stage의 평가 데이터를 예측하여 정해진 형식의 DataFrame을 반환한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import mvit_v2_s

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
cv2.setNumThreads(1)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("이 제출물은 CUDA GPU 평가환경을 필요로 합니다.")
    return torch.device("cuda")


def _video_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 1: MViTv2-S 기반 재녹화 분류기
# ---------------------------------------------------------------------------
def _clip_ids(path: Path, n: int, slot: int, slots: int):
    cap = cv2.VideoCapture(str(path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    center = (slot + 0.5) * total / slots
    start = max(0, min(total - n, round(center - n / 2)))
    return np.linspace(start, min(total - 1, start + n - 1), n).round().astype(int)


def _decode_stage1_clip(path: Path, size: int, frame_ids):
    cap = cv2.VideoCapture(str(path))
    out = []
    wanted = [int(x) for x in frame_ids]
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    pos = wanted[0]
    for idx in wanted:
        ok = False
        bgr = None
        while pos <= idx:
            ok, bgr = cap.read()
            pos += 1
            if not ok:
                break
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - size) // 2, (nw - size) // 2
        out.append(rgb[y : y + size, x : x + size])
    cap.release()
    if not out:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(out) < len(wanted):
        out.append(out[-1])
    x = torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2).float() / 255.0
    return (x - S1_MEAN) / S1_STD


class _Stage1Clips(Dataset):
    def __init__(self, videos, slots, size, frames):
        self.videos, self.slots, self.size, self.frames = videos, slots, size, frames

    def __len__(self):
        return len(self.videos) * self.slots

    def __getitem__(self, index):
        video_index, slot = index // self.slots, index % self.slots
        path = self.videos[video_index]
        try:
            x = _decode_stage1_clip(path, self.size, _clip_ids(path, self.frames, slot, self.slots))
            valid = 1
        except Exception:
            x = torch.zeros(3, self.frames, self.size, self.size)
            valid = 0
        return x, video_index, valid


def predict_stage1(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    size, frames = int(checkpoint["size"]), int(checkpoint["frames"])
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    root = Path(data_dir) / "videos"
    videos = _video_paths(root)
    slots = 3
    dataset = _Stage1Clips(videos, slots, size, frames)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
    scores = [[] for _ in videos]
    with torch.inference_mode():
        for clips, video_indices, valid in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prob = torch.softmax(model(clips.to(device, non_blocking=True)), 1)[:, 1]
            for idx, value, ok in zip(video_indices.tolist(), prob.float().cpu().tolist(), valid.tolist()):
                if ok:
                    scores[idx].append(float(value))

    rows = []
    for path, values in zip(videos, scores):
        probability = float(np.mean(values)) if values else 1.0
        rows.append({"ID": path.stem, "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL"})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 2: ResNet18 + BiGRU 기반 네 과업 모델
# ---------------------------------------------------------------------------
class _Stage2Frames(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _Stage2Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def forward(self, x):
        h, _ = self.r(x)
        collision_logits = self.tc(h).squeeze(-1)
        entry_logits = self.te(h).squeeze(-1)
        collision_index = collision_logits.argmax(1)
        entry_index = entry_logits.argmax(1)
        batch = torch.arange(len(h), device=h.device)
        scene_input = torch.cat([h[batch, collision_index], h[batch, entry_index]], 1)
        return collision_index, entry_index, self.scene(scene_input)


def _frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True))
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    temporal = _Stage2Temporal()
    temporal.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    temporal.to(device).eval()

    image_root = Path(data_dir) / "images"
    folders = sorted(p for p in image_root.iterdir() if p.is_dir())
    rows = []
    with torch.inference_mode():
        for folder in folders:
            # 평가 정의상 모든 원본 프레임을 사용한다(stride=1).
            paths = sorted(
                (p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                key=_frame_number,
            )
            if not paths:
                continue
            loader = DataLoader(_Stage2Frames(paths, transform), batch_size=256, num_workers=6, pin_memory=True)
            features = []
            for images in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
            sequence = torch.cat(features)[None].to(device)
            collision_idx, entry_idx, scene = temporal(sequence)
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[int(collision_idx)],
                    "entry_frame": frame_numbers[int(entry_idx)],
                    "evasion_space": int(scene[:, :2].argmax(1)),
                    "entry_side": "RIGHT" if int(scene[:, 2:].argmax(1)) else "LEFT",
                }
            )
    del backbone, temporal
    torch.cuda.empty_cache()
    return pd.DataFrame(
        rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
    )

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 3: MViTv2-S 기반 가감속/조향 다중헤드 모델
# ---------------------------------------------------------------------------
class _Stage3MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dimension = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dimension, 4)
        self.steer = nn.Linear(dimension, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.accel(features), self.steer(features)


def _stage3_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)))
        width, height = image.size
        x, y = (width - 224) // 2, (height - 224) // 2
        image = image.crop((x, y, x + 224, y + 224))
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = _Stage3MViT()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            centers = np.arange(count)
            accel_predictions, steer_predictions = [], []
            for start in range(0, count, 8):
                center = centers[start : start + 8]
                indices = np.clip(center[:, None] - 8 + np.arange(16)[None, :], 0, count - 1)
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - S3_MEAN[None, :, None, :, :]) / S3_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": ACCEL[accel],
                        "steer_label": STEER[steer],
                    }
                )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])

# CONTROLLED CANDIDATE OVERRIDE
"""S002 Stage 1 challenger: deterministic spectral re-capture detector.

Drop-in replacement for Baseline.inference.predict_stage1. It keeps the exact
contract (read data_dir/videos, return a DataFrame of ID,answer for every input
video) but decides from pixel forensics instead of the MViTv2 checkpoint.

Evidence (see lopo_report.json, produced by evaluate_lopo.py):
  leave-one-source-pair-out over the 5 Stage 1 pairs, both members of a pair
  always held out together, so scene content cannot leak across the split.

    all_original (S002 baseline behaviour)   macro-F1 0.3333   accuracy 0.50
    spec_peak_db threshold (this candidate)  macro-F1 1.0000   accuracy 1.00

  Fitted per-fold cuts stayed inside 25.28..25.97 and the class values are
  17.75..22.56 (ORIGINAL) versus 28.19..33.45 (RERECORDED), a 5.6-unit gap.
  DECISION_THRESHOLD is frozen at the midpoint of that gap.

Signal: a re-recorded clip photographs a display, so the display pixel grid
beats against the capture sensor grid and leaves narrow periodic peaks in the
2-D power spectrum. spec_peak_db measures the strongest such peak after the
natural 1/f falloff is whitened away radially.

Deliberately excluded: container/codec metadata. Locally every ORIGINAL is FMP4
and every RERECORDED is h264, which would score perfectly and teach nothing --
it is how the sample was packaged, not evidence about the recorded signal.

Runtime: no network, no model files, no RNG, no CUDA requirement. Uses only
numpy + opencv-python from Baseline/requirements.txt. ~0.35 s per 50-frame 720p
video single-threaded on the dev box.
"""


import math
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}

DECISION_THRESHOLD = 25.37
#: Values inside this band are treated as undecided by the primary feature and
#: resolved by corroborating high-frequency evidence. On the labelled sample no
#: video lands in the band, so this only guards unseen inputs.
AMBIGUOUS_BAND = (24.0, 27.0)
SHARPNESS_CUT = 2.11
HIGH_RATIO_CUT = 0.025

MAX_FRAMES = 32
CROP = 512
EPS = 1e-9
#: Fallback when a video cannot be decoded at all. ORIGINAL is the majority
#: answer of the previous S002 submission, so an undecodable file degrades to
#: known baseline behaviour instead of an unsupported guess.
FALLBACK_ANSWER = "ORIGINAL"


def _video_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def _read_frames(path: Path, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            frames.append(bgr)
    finally:
        capture.release()
    return frames


def _center_crop(gray: np.ndarray, size: int = CROP) -> np.ndarray:
    height, width = gray.shape[:2]
    side = min(size, height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return gray[top : top + side, left : left + side]


def _radial_grid(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    return np.hypot(fy, fx) / 0.5


def _frame_evidence(gray: np.ndarray) -> tuple[float, float, float]:
    """Return (spec_peak_db, spec_high_ratio, sharp_lap_log) for one frame."""
    patch = _center_crop(gray).astype(np.float64)
    patch = patch - patch.mean()
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(patch * window))) ** 2
    radius = _radial_grid(patch.shape)

    total = power[(radius > 0.02) & (radius <= 1.0)].sum() + EPS
    high = power[(radius > 0.45) & (radius <= 0.95)].sum()

    log_power = np.log10(power + EPS)
    whitened = np.zeros_like(log_power)
    edges = np.linspace(0.05, 0.85, 25)
    for lo, hi in pairwise(edges):
        mask = (radius >= lo) & (radius < hi)
        if mask.any():
            band = log_power[mask]
            whitened[mask] = band - float(np.median(band))
    residual = whitened[(radius >= 0.2) & (radius <= 0.9)]
    peak_db = float(np.percentile(residual, 99.99)) * 10.0

    laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return peak_db, float(high / total), float(math.log10(laplacian.var() + EPS))


def score_video(path: Path) -> dict[str, float]:
    """Deterministic forensic score for one video (public for tests/tools)."""
    frames = _read_frames(path)
    if not frames:
        return {"decoded_frames": 0.0}
    grays = [cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) for bgr in frames]
    picks = sorted({0, len(grays) // 2, len(grays) - 1})
    peaks, highs, sharps = [], [], []
    for index in picks:
        peak_db, high_ratio, sharpness = _frame_evidence(grays[index])
        peaks.append(peak_db)
        highs.append(high_ratio)
        sharps.append(sharpness)
    return {
        "decoded_frames": float(len(frames)),
        "spec_peak_db": float(np.mean(peaks)),
        "spec_high_ratio": float(np.mean(highs)),
        "sharp_lap_log": float(np.mean(sharps)),
    }


def classify(score: dict[str, float]) -> str:
    if not score.get("decoded_frames"):
        return FALLBACK_ANSWER
    peak_db = score["spec_peak_db"]
    if AMBIGUOUS_BAND[0] <= peak_db <= AMBIGUOUS_BAND[1]:
        corroboration = int(score["sharp_lap_log"] >= SHARPNESS_CUT) + int(
            score["spec_high_ratio"] >= HIGH_RATIO_CUT
        )
        return "RERECORDED" if corroboration >= 1 else "ORIGINAL"
    return "RERECORDED" if peak_db >= DECISION_THRESHOLD else "ORIGINAL"


def predict_stage1(data_dir, model_dir=None) -> pd.DataFrame:
    """Stage 1 prediction with the same signature/return contract as Baseline.

    model_dir is accepted and ignored: this candidate loads no checkpoint.
    """
    del model_dir
    cv2.setNumThreads(1)
    videos = _video_paths(Path(data_dir) / "videos")
    rows = [{"ID": path.stem, "answer": classify(score_video(path))} for path in videos]
    return pd.DataFrame(rows, columns=["ID", "answer"])


# CONTROLLED CANDIDATE OVERRIDE
"""Deterministic classical ego-motion challenger for Stage 3.

This module is a self-contained drop-in replacement for
`Baseline/inference.py::predict_stage3`. It has no repository imports, loads no
checkpoint, performs no network access, and streams one frame at a time so peak
memory is independent of video length.

Contract preserved from the baseline
------------------------------------
`predict_stage3(data_dir, model_dir)` returns a `pandas.DataFrame` with columns
`["ID", "sample_index", "accel_label", "steer_label"]` and exactly one row per
decoded frame of every video under `<data_dir>/videos`, with `sample_index`
running `0..n_decoded-1`. `model_dir` is accepted and ignored.

Cadence assumption
------------------
Official clarification states hidden Stage 3 inputs are true 10 Hz and the
decoded frame count equals the required `sample_index` count, so the inference
path assumes `dt = 0.1 s` between decoded frames. The public fixtures decode at
20 fps (~1200 frames over 60 s of content) with corrupt `CAP_PROP_FPS` metadata
(~479), so local fitting/evaluation passes `dt = 0.05`. All features are
expressed per second and normalized by frame size, which is what makes one set
of thresholds valid across both cadences and across resolutions.

Dependencies (pinned by Baseline/requirements.txt): numpy==1.26.4,
pandas==2.2.2, opencv-python==4.10.0.84.
"""


from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]

HIDDEN_DT_SECONDS = 0.1
PUBLIC_DT_SECONDS = 0.05

FEATURE_NAMES = ("speed", "yaw", "expansion", "tracked")


class MotionParams:
    """Feature-extraction geometry. Not fitted; chosen for cost and robustness."""

    def __init__(
        self,
        work_width=320,
        top_margin=0.10,
        bottom_margin=0.16,
        max_corners=150,
        quality_level=0.01,
        min_distance=8,
        block_size=7,
        lk_window=15,
        lk_levels=3,
        min_tracked=12,
        max_backward_error=1.0,
        trim_sigma=2.5,
    ):
        self.work_width = work_width
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.lk_window = lk_window
        self.lk_levels = lk_levels
        self.min_tracked = min_tracked
        self.max_backward_error = max_backward_error
        self.trim_sigma = trim_sigma


class DecisionParams:
    """Thresholds and smoothing windows fitted on public sparse labels.

    `speed` is median tracked flow magnitude in frame-heights per second.
    `rel_accel` is d(speed)/dt divided by local speed, i.e. units of 1/s.
    `yaw` is modeled horizontal image displacement at the principal point in
    frame-widths per second.
    """

    # Values below are the public-label fit (see artifacts/fit_report.json).
    # LOVO agreed on these to within one grid step on every fold.
    def __init__(
        self,
        speed_smooth_seconds=1.5,
        accel_window_seconds=2.0,
        yaw_smooth_seconds=1.5,
        stopped_speed=0.005,
        accel_threshold=0.20,
        decel_threshold=0.12,
        speed_floor=0.01,
        yaw_threshold=0.014,
        yaw_sign=1.0,
        accel_median_seconds=0.9,
        steer_median_seconds=0.9,
    ):
        self.speed_smooth_seconds = speed_smooth_seconds
        self.accel_window_seconds = accel_window_seconds
        self.yaw_smooth_seconds = yaw_smooth_seconds
        self.stopped_speed = stopped_speed
        self.accel_threshold = accel_threshold
        self.decel_threshold = decel_threshold
        self.speed_floor = speed_floor
        self.yaw_threshold = yaw_threshold
        self.yaw_sign = yaw_sign
        self.accel_median_seconds = accel_median_seconds
        self.steer_median_seconds = steer_median_seconds


DEFAULT_MOTION = MotionParams()
DEFAULT_DECISION = DecisionParams()


def video_paths(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXT
    )


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _window_frames(seconds: float, dt: float, minimum: int = 1) -> int:
    frames = round(seconds / dt) if dt > 0 else minimum
    return max(minimum, frames)


def _similarity_from_pairs(
    source: np.ndarray, target: np.ndarray, trim_sigma: float
) -> tuple[float, float, np.ndarray] | None:
    """Closed-form 2D similarity (scale, rotation, translation), trimmed once.

    Deterministic by construction: no RANSAC, no RNG. Returns
    `(scale, theta, translation)` or None when the fit is degenerate.
    """

    def fit(p: np.ndarray, q: np.ndarray):
        if len(p) < 3:
            return None
        p_mean = p.mean(axis=0)
        q_mean = q.mean(axis=0)
        dp = p - p_mean
        dq = q - q_mean
        denominator = float(np.sum(dp * dp))
        if denominator <= 1e-9:
            return None
        a = float(np.sum(dp * dq))
        b = float(np.sum(dp[:, 0] * dq[:, 1] - dp[:, 1] * dq[:, 0]))
        scale = float(np.hypot(a, b) / denominator)
        theta = float(np.arctan2(b, a))
        if not np.isfinite(scale) or scale <= 1e-6:
            return None
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float64,
        )
        matrix = scale * rotation
        translation = q_mean - matrix @ p_mean
        return scale, theta, translation, matrix

    first = fit(source, target)
    if first is None:
        return None
    _, _, translation, matrix = first
    residual = np.linalg.norm(target - (source @ matrix.T + translation), axis=1)
    median = float(np.median(residual))
    scale_mad = float(np.median(np.abs(residual - median))) * 1.4826
    if scale_mad > 1e-6:
        keep = residual <= median + trim_sigma * scale_mad
        if int(keep.sum()) >= max(3, len(source) // 4):
            second = fit(source[keep], target[keep])
            if second is not None:
                first = second
    scale, theta, translation, _ = first
    return scale, theta, translation


def _prepare_gray(frame: np.ndarray, work_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > work_width:
        scale = work_width / float(width)
        new_size = (work_width, max(2, round(height * scale)))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _feature_mask(shape: tuple[int, int], params: MotionParams) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    top = round(height * params.top_margin)
    bottom = height - round(height * params.bottom_margin)
    if bottom <= top:
        top, bottom = 0, height
    mask[top:bottom, :] = 255
    return mask


def extract_video_features(
    path: Path, params: MotionParams = DEFAULT_MOTION
) -> dict[str, np.ndarray]:
    """Stream a video and return per-decoded-frame ego-motion features.

    Every returned array has length equal to the decoded frame count. Index 0
    duplicates index 1 because motion needs a frame pair; single-frame videos
    yield zero motion.
    """

    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {Path(path).name}")

    speed: list[float] = []
    yaw: list[float] = []
    expansion: list[float] = []
    tracked: list[float] = []

    previous: np.ndarray | None = None
    mask: np.ndarray | None = None
    center: np.ndarray | None = None
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            decoded += 1
            gray = _prepare_gray(frame, params.work_width)
            if previous is None:
                previous = gray
                mask = _feature_mask(gray.shape[:2], params)
                center = np.array(
                    [gray.shape[1] / 2.0, gray.shape[0] / 2.0], dtype=np.float64
                )
                continue

            height, width = gray.shape[:2]
            corners = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=params.max_corners,
                qualityLevel=params.quality_level,
                minDistance=params.min_distance,
                mask=mask,
                blockSize=params.block_size,
            )
            frame_speed = 0.0
            frame_yaw = 0.0
            frame_expansion = 0.0
            frame_tracked = 0.0
            if corners is not None and len(corners) >= 3:
                lk = {
                    "winSize": (params.lk_window, params.lk_window),
                    "maxLevel": params.lk_levels,
                    "criteria": (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                }
                forward, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous, gray, corners, None, **lk
                )
                if forward is not None:
                    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
                        gray, previous, forward, None, **lk
                    )
                    good = status.reshape(-1).astype(bool)
                    if backward is not None and status_back is not None:
                        good &= status_back.reshape(-1).astype(bool)
                        error = np.linalg.norm(
                            corners.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
                        )
                        good &= error <= params.max_backward_error
                    source = corners.reshape(-1, 2)[good].astype(np.float64)
                    target = forward.reshape(-1, 2)[good].astype(np.float64)
                    frame_tracked = float(len(source))
                    if len(source) >= params.min_tracked:
                        displacement = np.linalg.norm(target - source, axis=1)
                        frame_speed = float(np.median(displacement)) / float(height)
                        fit = _similarity_from_pairs(source, target, params.trim_sigma)
                        if fit is not None:
                            scale, theta, translation = fit
                            rotation = np.array(
                                [
                                    [np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)],
                                ],
                                dtype=np.float64,
                            )
                            moved = (scale * rotation) @ center + translation
                            frame_yaw = float(moved[0] - center[0]) / float(width)
                            frame_expansion = float(scale) - 1.0
            speed.append(frame_speed)
            yaw.append(frame_yaw)
            expansion.append(frame_expansion)
            tracked.append(frame_tracked)
            previous = gray
    finally:
        capture.release()

    if decoded == 0:
        raise ValueError(f"cannot decode video: {Path(path).name}")
    if not speed:
        zeros = np.zeros(decoded, dtype=np.float64)
        return {name: zeros.copy() for name in FEATURE_NAMES}

    def head_padded(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return np.concatenate(([array[0]], array))

    features = {
        "speed": head_padded(speed),
        "yaw": head_padded(yaw),
        "expansion": head_padded(expansion),
        "tracked": head_padded(tracked),
    }
    for name, array in features.items():
        if len(array) != decoded:
            features[name] = _resize_to(array, decoded)
    return features


def _resize_to(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) == length:
        return array
    if len(array) > length:
        return array[:length]
    pad = np.full(length - len(array), array[-1], dtype=np.float64)
    return np.concatenate((array, pad))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, _odd(int(window)))
    if window == 1 or len(values) == 0:
        return values.astype(np.float64, copy=True)
    half = window // 2
    padded = np.pad(values.astype(np.float64), half, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, _odd(int(window)))
    if window == 1 or len(values) == 0:
        return values.astype(np.float64, copy=True)
    half = window // 2
    padded = np.pad(values.astype(np.float64), half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strided, axis=1)


def _mode_filter(labels: np.ndarray, window: int, classes: int) -> np.ndarray:
    """Deterministic majority filter; ties resolve to the lowest class index."""

    window = max(1, _odd(int(window)))
    if window == 1 or len(labels) == 0:
        return labels.astype(np.int64, copy=True)
    half = window // 2
    padded = np.pad(labels.astype(np.int64), half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    counts = np.stack(
        [(strided == value).sum(axis=1) for value in range(classes)], axis=1
    )
    return counts.argmax(axis=1)


def classify_features(
    features: dict[str, np.ndarray],
    dt: float,
    params: DecisionParams = DEFAULT_DECISION,
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-frame features to accel/steer class indices.

    Returns index arrays into `ACCEL` and `STEER`.
    """

    raw_speed = np.asarray(features["speed"], dtype=np.float64) / max(dt, 1e-9)
    raw_yaw = np.asarray(features["yaw"], dtype=np.float64) / max(dt, 1e-9)
    count = len(raw_speed)
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    speed = _moving_average(
        _median_filter(raw_speed, _window_frames(params.speed_smooth_seconds, dt)),
        _window_frames(params.speed_smooth_seconds, dt),
    )
    yaw = _moving_average(
        _median_filter(raw_yaw, _window_frames(params.yaw_smooth_seconds, dt)),
        _window_frames(params.yaw_smooth_seconds, dt),
    )

    lag = _window_frames(params.accel_window_seconds / 2.0, dt)
    forward_index = np.clip(np.arange(count) + lag, 0, count - 1)
    backward_index = np.clip(np.arange(count) - lag, 0, count - 1)
    span = (forward_index - backward_index).astype(np.float64) * dt
    span[span <= 0] = dt
    slope = (speed[forward_index] - speed[backward_index]) / span
    reference = np.maximum(
        (speed[forward_index] + speed[backward_index]) / 2.0, params.speed_floor
    )
    relative = slope / reference

    accel = np.full(count, ACCEL.index("CONSTANT"), dtype=np.int64)
    accel[relative > params.accel_threshold] = ACCEL.index("ACCELERATING")
    accel[relative < -params.decel_threshold] = ACCEL.index("DECELERATING")
    accel[speed < params.stopped_speed] = ACCEL.index("STOPPED")
    accel = _mode_filter(
        accel, _window_frames(params.accel_median_seconds, dt), len(ACCEL)
    )

    signed_yaw = yaw * float(params.yaw_sign)
    steer = np.full(count, STEER.index("STRAIGHT"), dtype=np.int64)
    steer[signed_yaw > params.yaw_threshold] = STEER.index("LEFT")
    steer[signed_yaw < -params.yaw_threshold] = STEER.index("RIGHT")
    steer = _mode_filter(
        steer, _window_frames(params.steer_median_seconds, dt), len(STEER)
    )
    return accel, steer


def predict_video(
    path: Path,
    dt: float = HIDDEN_DT_SECONDS,
    motion: MotionParams = DEFAULT_MOTION,
    decision: DecisionParams = DEFAULT_DECISION,
) -> tuple[np.ndarray, np.ndarray]:
    features = extract_video_features(path, motion)
    return classify_features(features, dt, decision)


def _fallback_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while capture.grab():
            decoded += 1
    finally:
        capture.release()
    return decoded


def predict_stage3(data_dir, model_dir=None) -> pd.DataFrame:
    """Drop-in replacement for the baseline Stage 3 predictor.

    One row per decoded frame per video, ordered by video then sample_index.
    `model_dir` is accepted for signature compatibility and ignored.
    """

    del model_dir
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames: list[pd.DataFrame] = []
    for path in videos:
        try:
            accel, steer = predict_video(path, HIDDEN_DT_SECONDS)
        except Exception:  # noqa: BLE001 - one bad video must not abort a submission
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel = np.full(count, ACCEL.index("CONSTANT"), dtype=np.int64)
            steer = np.full(count, STEER.index("STRAIGHT"), dtype=np.int64)
        count = len(accel)
        if count == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(value)] for value in accel],
                    "steer_label": [STEER[int(value)] for value in steer],
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["ID", "sample_index", "accel_label", "steer_label"]
        )
    table = pd.concat(frames, ignore_index=True)
    return table[["ID", "sample_index", "accel_label", "steer_label"]]


def with_decision(**overrides) -> DecisionParams:
    names = (
        "speed_smooth_seconds",
        "accel_window_seconds",
        "yaw_smooth_seconds",
        "stopped_speed",
        "accel_threshold",
        "decel_threshold",
        "speed_floor",
        "yaw_threshold",
        "yaw_sign",
        "accel_median_seconds",
        "steer_median_seconds",
    )
    values = {name: getattr(DEFAULT_DECISION, name) for name in names}
    values.update(overrides)
    return DecisionParams(**values)

# ISOLATED STAGE 2 COLLISION-ONLY OVERRIDE
_S2_CHAMPION_PREDICT_STAGE2 = predict_stage2
_S2_COLLISION_NAMESPACE = {"__name__": "_s2_collision_candidate"}
exec(
    compile('"""Deterministic Stage 2 temporal-localization challenger (collision-first).\n\nSelf-contained drop-in replacement for Baseline/inference.py::predict_stage2.\nNo repository imports, no checkpoint, no network, no dataclass, no GPU.\n\nContract preserved from the baseline\n-----------------------------------\npredict_stage2(data_dir, model_dir) returns a pandas.DataFrame with columns\n["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"] and one\nrow per immediate subdirectory of <data_dir>/images, ordered by folder name.\nFrame values are the trailing integer of the image file stem, exactly like the\nbaseline _frame_number, so they are the same numbering the official frame-time\nmap uses. model_dir is accepted and ignored.\n\nWhy not the baseline head\n------------------------\nThe shipped Stage 2 model is an ImageNet ResNet-18 plus a BiGRU whose collision\nhead saw five public clips for one epoch and whose entry and scene heads were\nnever trained at all. Its full-sequence GRU also holds one 512-d feature per\nframe, so memory grows without bound on unknown hidden clip lengths.\n\nThis module localizes the collision from sparse Lucas-Kanade tracking plus a\nclosed-form global similarity fit: an impact is a sudden, short-lived break in\notherwise smooth ego motion. It streams two frames at a time, so peak memory is\nindependent of clip length.\n\nValidation honesty\n------------------\nOnly t_collision is labeled in the public data (Baseline/data/stage2/labels.csv\ncarries t_entry = evasion_space = entry_side = -1). The collision path is fitted\nand reported under leave-one-video-out. The entry, entry_side and evasion_space\npaths are deterministic, interpretable and UNVALIDATED; see README.md.\n\nDependencies, all pinned by Baseline/requirements.txt: numpy==1.26.4,\npandas==2.2.2, opencv-python==4.10.0.84.\n"""\n\n\nimport re\nfrom pathlib import Path\n\nimport cv2\nimport numpy as np\nimport pandas as pd\n\n# When this file is appended to the official inference.py, retain the shipped\n# Stage 2 predictor so the final override can preserve its untested entry/scene\n# outputs and replace collision_frame only. Standalone execution keeps the full\n# deterministic candidate for diagnostics.\n_S2_BASELINE_PREDICT_STAGE2 = globals().get("predict_stage2")\n\nIMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}\n\nOUTPUT_COLUMNS = (\n    "ID",\n    "collision_frame",\n    "entry_frame",\n    "evasion_space",\n    "entry_side",\n)\n\nFEATURE_NAMES = (\n    "ego_speed",\n    "ego_scale",\n    "ego_theta",\n    "ego_shift_y",\n    "resid_median",\n    "resid_outlier",\n    "resid_energy",\n    "resid_center_x",\n    "warp_diff",\n    "tracked",\n)\n\n\nclass MotionParams:\n    """Feature geometry. Cost/robustness choices, not fitted to labels.\n\n    Plain class rather than a dataclass on purpose: the official harness loads\n    inference.py through an unregistered dynamic module, where dataclass field\n    resolution of postponed annotations fails.\n    """\n\n    def __init__(\n        self,\n        work_width=320,\n        max_corners=300,\n        quality_level=0.01,\n        min_distance=6,\n        block_size=7,\n        lk_window=15,\n        lk_levels=3,\n        min_tracked=16,\n        max_backward_error=1.0,\n        trim_sigma=2.0,\n        residual_floor=0.75,\n        outlier_scale=3.0,\n    ):\n        self.work_width = work_width\n        self.max_corners = max_corners\n        self.quality_level = quality_level\n        self.min_distance = min_distance\n        self.block_size = block_size\n        self.lk_window = lk_window\n        self.lk_levels = lk_levels\n        self.min_tracked = min_tracked\n        self.max_backward_error = max_backward_error\n        self.trim_sigma = trim_sigma\n        self.residual_floor = residual_floor\n        self.outlier_scale = outlier_scale\n\n\nclass DecisionParams:\n    """Detector thresholds. Collision values are the leave-one-out fit.\n\n    baseline_window   frames of local history used as the smooth-motion baseline\n    jolt_weights      relative weight of each saliency channel\n    onset_ratio       fraction of the peak saliency that defines impact onset\n    max_backtrack     hard cap on how far the onset search may walk back\n    guard_frames      ignore the first/last frames, where motion is one-sided\n    entry_lead_*      entry search geometry, unvalidated (no public labels)\n    """\n\n    def __init__(\n        self,\n        baseline_window=9,\n        onset_ratio=0.5,\n        max_backtrack=4,\n        guard_frames=2,\n        weight_speed=1.0,\n        weight_theta=0.5,\n        weight_warp=0.5,\n        entry_min_lead=3,\n        entry_max_lead=20,\n        entry_onset_ratio=0.6,\n        entry_default_lead=8,\n        side_window=3,\n        evasion_threshold=0.10,\n    ):\n        self.baseline_window = baseline_window\n        self.onset_ratio = onset_ratio\n        self.max_backtrack = max_backtrack\n        self.guard_frames = guard_frames\n        self.weight_speed = weight_speed\n        self.weight_theta = weight_theta\n        self.weight_warp = weight_warp\n        self.entry_min_lead = entry_min_lead\n        self.entry_max_lead = entry_max_lead\n        self.entry_onset_ratio = entry_onset_ratio\n        self.entry_default_lead = entry_default_lead\n        self.side_window = side_window\n        self.evasion_threshold = evasion_threshold\n\n\nDEFAULT_MOTION = MotionParams()\nDEFAULT_DECISION = DecisionParams()\n\n\ndef with_decision(**overrides) -> DecisionParams:\n    """Copy DEFAULT_DECISION with overrides, for fitting and ablations."""\n\n    values = dict(vars(DEFAULT_DECISION))\n    unknown = sorted(set(overrides) - set(values))\n    if unknown:\n        raise ValueError(f"unknown decision parameters: {unknown}")\n    values.update(overrides)\n    return DecisionParams(**values)\n\n\n# ---------------------------------------------------------------------------\n# Frame discovery\n# ---------------------------------------------------------------------------\ndef frame_number(path) -> int:\n    """Trailing integer of the file stem, matching the baseline convention."""\n\n    match = re.search(r"(\\d+)$", Path(path).stem)\n    return int(match.group(1)) if match else 0\n\n\ndef frame_paths(folder) -> list[Path]:\n    folder = Path(folder)\n    if not folder.is_dir():\n        return []\n    return sorted(\n        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),\n        key=frame_number,\n    )\n\n\ndef frame_numbers(folder) -> list[int]:\n    return [frame_number(path) for path in frame_paths(folder)]\n\n\n# ---------------------------------------------------------------------------\n# Motion features\n# ---------------------------------------------------------------------------\ndef _prepare_gray(image: np.ndarray, work_width: int) -> np.ndarray:\n    height, width = image.shape[:2]\n    if width > work_width:\n        scale = work_width / float(width)\n        image = cv2.resize(\n            image,\n            (work_width, max(2, int(round(height * scale)))),\n            interpolation=cv2.INTER_AREA,\n        )\n    if image.ndim == 3:\n        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n    return image\n\n\ndef _fit_similarity(source: np.ndarray, target: np.ndarray):\n    """Closed-form 2D similarity fit. Deterministic: no RANSAC, no RNG."""\n\n    if len(source) < 3:\n        return None\n    source_mean = source.mean(axis=0)\n    target_mean = target.mean(axis=0)\n    ds = source - source_mean\n    dt = target - target_mean\n    denominator = float(np.sum(ds * ds))\n    if denominator <= 1e-9:\n        return None\n    a = float(np.sum(ds * dt))\n    b = float(np.sum(ds[:, 0] * dt[:, 1] - ds[:, 1] * dt[:, 0]))\n    scale = float(np.hypot(a, b) / denominator)\n    theta = float(np.arctan2(b, a))\n    if not np.isfinite(scale) or scale <= 1e-6:\n        return None\n    matrix = scale * np.array(\n        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],\n        dtype=np.float64,\n    )\n    translation = target_mean - matrix @ source_mean\n    return scale, theta, translation, matrix\n\n\ndef _robust_similarity(source: np.ndarray, target: np.ndarray, trim_sigma: float):\n    """Similarity fit trimmed once so independent movers cannot dominate it."""\n\n    first = _fit_similarity(source, target)\n    if first is None:\n        return None\n    _, _, translation, matrix = first\n    residual = np.linalg.norm(target - (source @ matrix.T + translation), axis=1)\n    median = float(np.median(residual))\n    spread = float(np.median(np.abs(residual - median))) * 1.4826\n    if spread > 1e-6:\n        keep = residual <= median + trim_sigma * spread\n        if int(keep.sum()) >= max(3, len(source) // 4):\n            second = _fit_similarity(source[keep], target[keep])\n            if second is not None:\n                first = second\n    return first\n\n\ndef _pair_features(previous, current, params: MotionParams) -> dict:\n    """Features for one consecutive frame pair, all geometry-normalized."""\n\n    height, width = current.shape[:2]\n    result = {name: 0.0 for name in FEATURE_NAMES}\n    result["warp_diff"] = (\n        float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))\n        / 255.0\n    )\n\n    corners = cv2.goodFeaturesToTrack(\n        previous,\n        maxCorners=params.max_corners,\n        qualityLevel=params.quality_level,\n        minDistance=params.min_distance,\n        blockSize=params.block_size,\n    )\n    if corners is None or len(corners) < 3:\n        return result\n\n    lk = {\n        "winSize": (params.lk_window, params.lk_window),\n        "maxLevel": params.lk_levels,\n        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),\n    }\n    forward, status, _ = cv2.calcOpticalFlowPyrLK(\n        previous, current, corners, None, **lk\n    )\n    if forward is None or status is None:\n        return result\n    good = status.reshape(-1).astype(bool)\n    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(\n        current, previous, forward, None, **lk\n    )\n    if backward is not None and status_back is not None:\n        good &= status_back.reshape(-1).astype(bool)\n        error = np.linalg.norm(corners.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)\n        good &= error <= params.max_backward_error\n    source = corners.reshape(-1, 2)[good].astype(np.float64)\n    target = forward.reshape(-1, 2)[good].astype(np.float64)\n\n    result["tracked"] = float(len(source))\n    if len(source) < params.min_tracked:\n        return result\n\n    displacement = np.linalg.norm(target - source, axis=1)\n    result["ego_speed"] = float(np.median(displacement)) / float(height)\n\n    fit = _robust_similarity(source, target, params.trim_sigma)\n    if fit is None:\n        return result\n    scale, theta, translation, matrix = fit\n    result["ego_scale"] = float(scale) - 1.0\n    result["ego_theta"] = float(theta)\n    result["ego_shift_y"] = float(translation[1]) / float(height)\n\n    residual = np.linalg.norm(target - (source @ matrix.T + translation), axis=1)\n    median = float(np.median(residual))\n    spread = float(np.median(np.abs(residual - median))) * 1.4826\n    cutoff = max(median + params.outlier_scale * spread, params.residual_floor)\n    outlier = residual > cutoff\n    result["resid_median"] = median / float(height)\n    result["resid_outlier"] = float(np.mean(outlier))\n    if bool(outlier.any()):\n        result["resid_energy"] = float(np.mean(residual[outlier])) / float(height)\n        centroid_x = float(np.mean(target[outlier, 0]))\n        result["resid_center_x"] = (centroid_x - width / 2.0) / float(width)\n\n    warped = cv2.warpAffine(\n        previous,\n        np.hstack([matrix, translation.reshape(2, 1)]).astype(np.float32),\n        (width, height),\n        flags=cv2.INTER_LINEAR,\n        borderMode=cv2.BORDER_REPLICATE,\n    )\n    border = max(2, int(round(0.05 * min(height, width))))\n    inner_current = current[border:-border, border:-border].astype(np.float32)\n    inner_warped = warped[border:-border, border:-border].astype(np.float32)\n    if inner_current.size:\n        result["warp_diff"] = (\n            float(np.mean(np.abs(inner_current - inner_warped))) / 255.0\n        )\n    return result\n\n\ndef extract_folder_features(folder, params: MotionParams = DEFAULT_MOTION) -> dict:\n    """Per-frame features for one sample folder.\n\n    Each array has one entry per image. Index 0 repeats index 1 because motion\n    needs a frame pair. Only two grayscale frames are resident at a time.\n    """\n\n    cv2.setNumThreads(1)\n    paths = frame_paths(folder)\n    count = len(paths)\n    if count == 0:\n        return {name: np.zeros(0, dtype=np.float64) for name in FEATURE_NAMES}\n\n    values = {name: [] for name in FEATURE_NAMES}\n    previous = None\n    for path in paths:\n        image = cv2.imread(str(path), cv2.IMREAD_COLOR)\n        current = previous if image is None else _prepare_gray(image, params.work_width)\n        if current is None:\n            for name in FEATURE_NAMES:\n                values[name].append(0.0)\n            continue\n        if previous is None:\n            previous = current\n            continue\n        if current.shape != previous.shape:\n            current = cv2.resize(\n                current,\n                (previous.shape[1], previous.shape[0]),\n                interpolation=cv2.INTER_AREA,\n            )\n        pair = _pair_features(previous, current, params)\n        for name in FEATURE_NAMES:\n            values[name].append(float(pair[name]))\n        previous = current\n\n    features = {}\n    for name in FEATURE_NAMES:\n        array = np.asarray(values[name], dtype=np.float64)\n        if array.size == 0:\n            array = np.zeros(1, dtype=np.float64)\n        array = np.concatenate(([array[0]], array))\n        features[name] = _resize_to(array, count)\n    return features\n\n\ndef _resize_to(array: np.ndarray, length: int) -> np.ndarray:\n    if len(array) == length:\n        return array\n    if len(array) > length:\n        return array[:length]\n    pad = np.full(length - len(array), array[-1], dtype=np.float64)\n    return np.concatenate((array, pad))\n\n\n# ---------------------------------------------------------------------------\n# Saliency and event localization\n# ---------------------------------------------------------------------------\ndef _odd(value: int) -> int:\n    value = max(1, int(value))\n    return value if value % 2 == 1 else value + 1\n\n\ndef _trailing_baseline(values: np.ndarray, window: int) -> np.ndarray:\n    """Median of the preceding window, i.e. what smooth motion predicts here."""\n\n    window = max(1, int(window))\n    count = len(values)\n    baseline = np.empty(count, dtype=np.float64)\n    for index in range(count):\n        start = max(0, index - window)\n        segment = values[start : index + 1]\n        baseline[index] = float(np.median(segment)) if segment.size else 0.0\n    return baseline\n\n\ndef _normalized_jolt(values: np.ndarray, window: int) -> np.ndarray:\n    """Positive break of a signal against its own recent history, scale-free."""\n\n    values = np.asarray(values, dtype=np.float64)\n    if values.size == 0:\n        return values\n    baseline = _trailing_baseline(values, window)\n    excess = np.maximum(values - baseline, 0.0)\n    scale = float(np.median(np.abs(values - baseline)))\n    if scale <= 1e-9:\n        scale = float(np.median(np.abs(values))) or 1.0\n    normalized = excess / (scale + 1e-9)\n    peak = float(np.max(normalized))\n    return normalized / peak if peak > 0 else normalized\n\n\ndef collision_saliency(\n    features: dict, decision: DecisionParams = DEFAULT_DECISION\n) -> np.ndarray:\n    """Combined impact saliency in [0, 1], one value per frame.\n\n    An impact shows up as a simultaneous break in tracked flow magnitude\n    (ego_speed), camera roll/jolt (ego_theta) and ego-compensated photometric\n    change (warp_diff). Each channel is normalized against its own history, so\n    no channel can dominate through raw units.\n    """\n\n    window = decision.baseline_window\n    speed = _normalized_jolt(np.asarray(features["ego_speed"], dtype=np.float64), window)\n    theta = _normalized_jolt(\n        np.abs(np.asarray(features["ego_theta"], dtype=np.float64)), window\n    )\n    warp = _normalized_jolt(np.asarray(features["warp_diff"], dtype=np.float64), window)\n    score = (\n        decision.weight_speed * speed\n        + decision.weight_theta * theta\n        + decision.weight_warp * warp\n    )\n    peak = float(np.max(score)) if score.size else 0.0\n    return score / peak if peak > 0 else score\n\n\ndef _guarded_argmax(score: np.ndarray, guard: int) -> int:\n    count = len(score)\n    if count == 0:\n        return 0\n    guard = int(max(0, min(guard, (count - 1) // 2)))\n    window = score[guard : count - guard] if count - 2 * guard > 0 else score\n    offset = guard if count - 2 * guard > 0 else 0\n    return offset + int(np.argmax(window))\n\n\ndef _onset_index(score: np.ndarray, peak: int, ratio: float, max_backtrack: int) -> int:\n    """Walk back from the peak to where the event actually starts.\n\n    The tracked-flow break is largest one to three frames after contact, because\n    the camera keeps reacting. Reporting the onset removes that lag bias.\n    """\n\n    index = int(peak)\n    limit = max(0, int(peak) - int(max_backtrack))\n    threshold = ratio * float(score[peak]) if len(score) else 0.0\n    while index > limit and float(score[index - 1]) >= threshold:\n        index -= 1\n    return index\n\n\ndef locate_collision(\n    features: dict, decision: DecisionParams = DEFAULT_DECISION\n) -> int:\n    """Index (not frame number) of the predicted collision frame."""\n\n    score = collision_saliency(features, decision)\n    if score.size == 0:\n        return 0\n    peak = _guarded_argmax(score, decision.guard_frames)\n    return _onset_index(score, peak, decision.onset_ratio, decision.max_backtrack)\n\n\ndef locate_entry(\n    features: dict, collision_index: int, decision: DecisionParams = DEFAULT_DECISION\n) -> int:\n    """Index of the first lane-entry frame. UNVALIDATED: no public entry labels.\n\n    Interpretable proxy: before impact, the intruding vehicle appears as tracks\n    the global ego-motion model cannot explain. The entry frame is taken as the\n    onset of that independent-mover residual energy inside a bounded lead window\n    before the collision. When no residual structure is present the estimate\n    falls back to a fixed lead, which keeps the output in range and finite.\n    """\n\n    count = len(features["resid_energy"])\n    if count == 0:\n        return 0\n    high = max(0, int(collision_index) - int(decision.entry_min_lead))\n    low = max(0, int(collision_index) - int(decision.entry_max_lead))\n    if high <= low:\n        return max(0, min(count - 1, high))\n\n    energy = np.asarray(features["resid_energy"], dtype=np.float64)[low : high + 1]\n    outlier = np.asarray(features["resid_outlier"], dtype=np.float64)[low : high + 1]\n    signal = energy * np.maximum(outlier, 1e-6)\n    peak_value = float(np.max(signal)) if signal.size else 0.0\n    if peak_value <= 0:\n        fallback = int(collision_index) - int(decision.entry_default_lead)\n        return max(0, min(count - 1, fallback))\n    local_peak = int(np.argmax(signal))\n    threshold = decision.entry_onset_ratio * peak_value\n    while local_peak > 0 and float(signal[local_peak - 1]) >= threshold:\n        local_peak -= 1\n    return max(0, min(count - 1, low + local_peak))\n\n\ndef decide_entry_side(\n    features: dict, entry_index: int, decision: DecisionParams = DEFAULT_DECISION\n) -> str:\n    """LEFT/RIGHT from the intruder centroid. UNVALIDATED: no public labels."""\n\n    center = np.asarray(features["resid_center_x"], dtype=np.float64)\n    count = len(center)\n    if count == 0:\n        return "LEFT"\n    low = max(0, int(entry_index) - int(decision.side_window))\n    high = min(count, int(entry_index) + int(decision.side_window) + 1)\n    weight = np.abs(\n        np.asarray(features["resid_energy"], dtype=np.float64)[low:high]\n    )\n    segment = center[low:high]\n    if segment.size == 0:\n        return "LEFT"\n    if float(np.sum(weight)) <= 1e-9:\n        value = float(np.mean(segment))\n    else:\n        value = float(np.sum(segment * weight) / np.sum(weight))\n    return "RIGHT" if value > 0 else "LEFT"\n\n\ndef decide_evasion_space(\n    features: dict, collision_index: int, decision: DecisionParams = DEFAULT_DECISION\n) -> int:\n    """Binary evasion space at collision time. UNVALIDATED: no public labels.\n\n    Proxy: a large share of unexplained tracks at impact means the approaching\n    object occupies much of the field, i.e. little room to evade. Returning both\n    classes deterministically avoids the degenerate single-class Macro-F1 term a\n    constant answer would produce.\n    """\n\n    outlier = np.asarray(features["resid_outlier"], dtype=np.float64)\n    if outlier.size == 0:\n        return 0\n    low = max(0, int(collision_index) - 2)\n    high = min(len(outlier), int(collision_index) + 3)\n    occupancy = float(np.mean(outlier[low:high]))\n    return 0 if occupancy >= decision.evasion_threshold else 1\n\n\ndef predict_folder(\n    folder,\n    motion: MotionParams = DEFAULT_MOTION,\n    decision: DecisionParams = DEFAULT_DECISION,\n) -> dict:\n    """Full Stage 2 answer for one sample folder, in frame-number space."""\n\n    numbers = frame_numbers(folder)\n    if not numbers:\n        return {\n            "ID": Path(folder).name,\n            "collision_frame": 0,\n            "entry_frame": 0,\n            "evasion_space": 0,\n            "entry_side": "LEFT",\n        }\n    features = extract_folder_features(folder, motion)\n    collision_index = locate_collision(features, decision)\n    entry_index = locate_entry(features, collision_index, decision)\n    return {\n        "ID": Path(folder).name,\n        "collision_frame": int(numbers[collision_index]),\n        "entry_frame": int(numbers[entry_index]),\n        "evasion_space": int(decide_evasion_space(features, collision_index, decision)),\n        "entry_side": decide_entry_side(features, entry_index, decision),\n    }\n\n\ndef predict_stage2(data_dir, model_dir=None) -> pd.DataFrame:\n    """Drop-in replacement for the baseline Stage 2 predictor.\n\n    One row per sample folder under <data_dir>/images. Each folder is processed\n    independently, as the rules require. model_dir is accepted and ignored.\n    """\n\n    del model_dir\n    cv2.setNumThreads(1)\n    image_root = Path(data_dir) / "images"\n    folders = (\n        sorted(p for p in image_root.iterdir() if p.is_dir())\n        if image_root.is_dir()\n        else []\n    )\n    rows = []\n    for folder in folders:\n        try:\n            rows.append(predict_folder(folder))\n        except Exception:  # noqa: BLE001 - one bad folder must not void a submission\n            numbers = frame_numbers(folder)\n            middle = numbers[len(numbers) // 2] if numbers else 0\n            first = numbers[0] if numbers else 0\n            rows.append(\n                {\n                    "ID": folder.name,\n                    "collision_frame": int(middle),\n                    "entry_frame": int(first),\n                    "evasion_space": 0,\n                    "entry_side": "LEFT",\n                }\n            )\n    if not rows:\n        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))\n    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))\n\n\n_S2_MOTION_PREDICT_STAGE2 = predict_stage2\n\n\ndef predict_stage2(data_dir, model_dir=None) -> pd.DataFrame:\n    """Replace collision timing only; preserve every other baseline output."""\n\n    motion = _S2_MOTION_PREDICT_STAGE2(data_dir, model_dir)\n    if not callable(_S2_BASELINE_PREDICT_STAGE2):\n        return motion\n\n    baseline = _S2_BASELINE_PREDICT_STAGE2(data_dir, model_dir).copy()\n    collision_by_id = dict(\n        zip(motion["ID"].astype(str), motion["collision_frame"].astype(int))\n    )\n    baseline["collision_frame"] = [\n        collision_by_id.get(str(sample_id), int(old_value))\n        for sample_id, old_value in zip(\n            baseline["ID"], baseline["collision_frame"]\n        )\n    ]\n    return baseline\n', "s2_collision_candidate.py", "exec"),
    _S2_COLLISION_NAMESPACE,
)
_S2_COLLISION_PREDICT_STAGE2 = _S2_COLLISION_NAMESPACE["predict_stage2"]


def predict_stage2(data_dir, model_dir):
    baseline = _S2_CHAMPION_PREDICT_STAGE2(data_dir, model_dir).copy()
    motion = _S2_COLLISION_PREDICT_STAGE2(data_dir, model_dir)
    collision_by_id = dict(
        zip(motion["ID"].astype(str), motion["collision_frame"].astype(int))
    )
    baseline["collision_frame"] = [
        collision_by_id.get(str(sample_id), int(old_value))
        for sample_id, old_value in zip(
            baseline["ID"], baseline["collision_frame"]
        )
    ]
    return baseline

# CONTROLLED CANDIDATE OVERRIDE
"""Replace only entry_frame with a fixed nine-frame lead from collision."""

from pathlib import Path
import re


_S007_BASE_PREDICT_STAGE2 = predict_stage2
_S007_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _s007_frame_number(path):
    match = re.search(r"(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else 0


def _s007_entry_by_id(data_dir, predictions, lead_frames=9):
    image_root = Path(data_dir) / "images"
    result = {}
    collision_by_id = dict(
        zip(predictions["ID"].astype(str), predictions["collision_frame"].astype(int))
    )
    if not image_root.is_dir():
        return result
    for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
        paths = sorted(
            (
                path
                for path in folder.iterdir()
                if path.suffix.lower() in _S007_IMAGE_SUFFIXES
            ),
            key=_s007_frame_number,
        )
        numbers = [_s007_frame_number(path) for path in paths]
        if not numbers or folder.name not in collision_by_id:
            continue
        collision = collision_by_id[folder.name]
        target = int(collision) - int(lead_frames)
        result[folder.name] = int(min(numbers, key=lambda number: abs(number - target)))
    return result


def predict_stage2(data_dir, model_dir):
    frame = _S007_BASE_PREDICT_STAGE2(data_dir, model_dir).copy()
    entry_by_id = _s007_entry_by_id(data_dir, frame)
    frame["entry_frame"] = [
        entry_by_id.get(str(sample_id), int(old_value))
        for sample_id, old_value in zip(frame["ID"], frame["entry_frame"])
    ]
    return frame

# CONTROLLED CANDIDATE OVERRIDE
"""S008: replace only Stage 2 entry_side using bounded vehicle tracking.

The module is appended to the frozen S007 inference.py. It calls S007 exactly
once, detects COCO road vehicles in a short entry-to-collision window, anchors
the likely collision vehicle near impact, tracks it backward, and reports the
screen side where that target first appears.
"""

from pathlib import Path as _S008Path
import re as _s008_re

import cv2 as _s008_cv2
import numpy as _s008_np
import torch as _s008_torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large as _s008_ssdlite,
)


_S008_BASE_PREDICT_STAGE2 = predict_stage2
_S008_WEIGHT_NAME = "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
_S008_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_S008_VEHICLE_LABELS = {2, 3, 4, 6, 8}
_S008_SCORE_THRESHOLD = 0.18
_S008_MAX_FRAMES = 28


def _s008_frame_number(path):
    match = _s008_re.search(r"(\d+)$", _S008Path(path).stem)
    return int(match.group(1)) if match else 0


def _s008_frame_paths(folder):
    return sorted(
        (
            path
            for path in _S008Path(folder).iterdir()
            if path.suffix.lower() in _S008_IMAGE_SUFFIXES
        ),
        key=_s008_frame_number,
    )


def _s008_window(paths, entry_frame, collision_frame):
    if not paths:
        return []
    numbers = [_s008_frame_number(path) for path in paths]
    collision_index = min(
        range(len(numbers)), key=lambda index: abs(numbers[index] - int(collision_frame))
    )
    entry_index = min(
        range(len(numbers)), key=lambda index: abs(numbers[index] - int(entry_frame))
    )
    end = min(len(paths) - 1, collision_index + 3)
    broad = _s008_np.linspace(0, end, min(16, end + 1))
    local_start = max(0, entry_index - 6)
    local = _s008_np.linspace(
        local_start, end, min(16, end - local_start + 1)
    )
    indices = sorted(
        {int(round(float(index))) for index in _s008_np.concatenate((broad, local))}
    )
    if len(indices) > _S008_MAX_FRAMES:
        keep = _s008_np.linspace(0, len(indices) - 1, _S008_MAX_FRAMES)
        indices = [indices[int(round(float(index)))] for index in keep]
    return [paths[index] for index in indices]


def _s008_load_model(model_dir, device):
    model = _s008_ssdlite(weights=None, weights_backbone=None, num_classes=91)
    state = _s008_torch.load(
        _S008Path(model_dir) / _S008_WEIGHT_NAME,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _s008_tensor(path):
    image = _s008_cv2.imread(str(path), _s008_cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    rgb = _s008_cv2.cvtColor(image, _s008_cv2.COLOR_BGR2RGB)
    tensor = _s008_torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return tensor, (height, width)


def _s008_appearance(path, box):
    image = _s008_cv2.imread(str(path), _s008_cv2.IMREAD_COLOR)
    if image is None:
        return _s008_np.zeros(80, dtype=_s008_np.float32)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return _s008_np.zeros(80, dtype=_s008_np.float32)
    hsv = _s008_cv2.cvtColor(crop, _s008_cv2.COLOR_BGR2HSV)
    color = _s008_cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
    gray = _s008_cv2.cvtColor(crop, _s008_cv2.COLOR_BGR2GRAY)
    tone = _s008_cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel()
    descriptor = _s008_np.concatenate((color, tone)).astype(_s008_np.float32)
    total = float(descriptor.sum())
    return descriptor / total if total > 1e-9 else descriptor


def _s008_detections(output, shape, path):
    height, width = shape
    rows = []
    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < _S008_SCORE_THRESHOLD:
            break
        if int(label) not in _S008_VEHICLE_LABELS:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
        box_width = normalized[2] - normalized[0]
        box_height = normalized[3] - normalized[1]
        area = box_width * box_height
        if area > 0.45 or box_width > 0.82 or box_height > 0.90:
            continue
        rows.append(
            {
                "box": normalized,
                "label": int(label),
                "score": float(score),
                "appearance": _s008_appearance(path, normalized),
            }
        )
    return rows


def _s008_geometry(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), (x2 - x1) * (y2 - y1), y2)


def _s008_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    area = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - area
    return area / union if union > 1e-9 else 0.0


def _s008_impact_score(detection, frame_distance):
    center_x, _center_y, area, bottom = _s008_geometry(detection["box"])
    centrality = max(0.0, 1.0 - 1.4 * abs(center_x - 0.5))
    proximity = 0.35 + 0.65 * bottom
    size = min(1.0, 3.0 * _s008_np.sqrt(max(area, 0.0)))
    temporal = 1.0 / (1.0 + 0.18 * abs(float(frame_distance)))
    return detection["score"] * (0.25 + centrality) * proximity * (0.25 + size) * temporal


def _s008_anchor(per_frame, numbers, collision_frame):
    best = None
    best_value = -1.0
    for index, detections in enumerate(per_frame):
        distance = numbers[index] - int(collision_frame)
        if abs(distance) > 5:
            continue
        for detection in detections:
            value = _s008_impact_score(detection, distance)
            if value > best_value:
                best = (index, detection)
                best_value = value
    return best


def _s008_appearance_similarity(left, right):
    left_vector = left["appearance"]
    right_vector = right["appearance"]
    denominator = float(
        _s008_np.linalg.norm(left_vector) * _s008_np.linalg.norm(right_vector)
    )
    if denominator <= 1e-9:
        return 0.0
    return float(_s008_np.dot(left_vector, right_vector) / denominator)


def _s008_association(candidate, current, anchor):
    cx, cy, area, _bottom = _s008_geometry(candidate["box"])
    nx, ny, next_area, _next_bottom = _s008_geometry(current["box"])
    distance = float(_s008_np.hypot(cx - nx, cy - ny))
    if distance > 0.42:
        return -1e9
    ratio = min(area, next_area) / max(area, next_area, 1e-9)
    label_bonus = 0.18 if candidate["label"] == current["label"] else 0.0
    appearance = 0.6 * _s008_appearance_similarity(candidate, current) + 0.4 * (
        _s008_appearance_similarity(candidate, anchor)
    )
    return (
        2.5 * _s008_iou(candidate["box"], current["box"])
        + 1.2 * (1.0 - distance)
        + 0.55 * ratio
        + 1.8 * appearance
        + label_bonus
        + 0.15 * candidate["score"]
    )


def _s008_track_origin(per_frame, numbers, collision_frame):
    anchor = _s008_anchor(per_frame, numbers, collision_frame)
    if anchor is None:
        return None
    index, current = anchor
    anchor_detection = current
    track = [(index, current)]
    misses = 0
    for previous in range(index - 1, -1, -1):
        if not per_frame[previous]:
            misses += 1
            if misses > 2:
                break
            continue
        ranked = sorted(
            (
                (_s008_association(candidate, current, anchor_detection), candidate)
                for candidate in per_frame[previous]
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        value, candidate = ranked[0]
        if value < 2.0:
            misses += 1
            if misses > 2:
                break
            continue
        current = candidate
        track.append((previous, current))
        misses = 0
    ordered = sorted(track, key=lambda item: item[0])
    if len(ordered) < 3:
        return None
    earliest_x = _s008_geometry(ordered[0][1]["box"])[0]
    anchor_x = _s008_geometry(anchor_detection["box"])[0]
    displacement = anchor_x - earliest_x
    if earliest_x <= 0.35:
        return "LEFT"
    if earliest_x >= 0.65:
        return "RIGHT"
    # When the target first becomes detectable near the image centre, infer the
    # side from its horizontal travel into the collision region. Moving right
    # means it entered from the left, and vice versa.
    if displacement >= 0.10:
        return "LEFT"
    if displacement <= -0.10:
        return "RIGHT"
    return None


def _s008_predict_sides(data_dir, model_dir, baseline):
    image_root = _S008Path(data_dir) / "images"
    if not image_root.is_dir():
        return {}
    device = _s008_torch.device("cuda" if _s008_torch.cuda.is_available() else "cpu")
    model = _s008_load_model(model_dir, device)
    baseline_by_id = {str(row.ID): row for row in baseline.itertuples(index=False)}
    result = {}
    for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
        row = baseline_by_id.get(folder.name)
        if row is None:
            continue
        paths = _s008_window(
            _s008_frame_paths(folder), row.entry_frame, row.collision_frame
        )
        tensors = []
        shapes = []
        numbers = []
        for path in paths:
            tensor, shape = _s008_tensor(path)
            if tensor is None:
                continue
            tensors.append(tensor)
            shapes.append(shape)
            numbers.append(_s008_frame_number(path))
        if not tensors:
            continue
        with _s008_torch.inference_mode():
            outputs = model([tensor.to(device) for tensor in tensors])
        detections = [
            _s008_detections(output, shape, path)
            for output, shape, path in zip(outputs, shapes, paths)
        ]
        side = _s008_track_origin(detections, numbers, int(row.collision_frame))
        if side is not None:
            result[folder.name] = side
        del outputs, tensors
    del model
    if device.type == "cuda":
        _s008_torch.cuda.empty_cache()
    return result


def predict_stage2(data_dir, model_dir):
    frame = _S008_BASE_PREDICT_STAGE2(data_dir, model_dir).copy()
    sides = _s008_predict_sides(data_dir, model_dir, frame)
    frame["entry_side"] = [
        sides.get(str(sample_id), old_value)
        for sample_id, old_value in zip(frame["ID"], frame["entry_side"])
    ]
    return frame

# CONTROLLED CANDIDATE OVERRIDE
"""S010: replace only Stage 2 entry_frame with the cached motion onset.

S008 already executes the frozen S006 temporal candidate once to obtain its
collision prediction.  That candidate computes a complete motion table,
including the independent-mover entry onset, but the outer isolation wrapper
discards every field except collision_frame.  This override temporarily wraps
the embedded full predictor, captures that already-computed table during the
normal S008 call, restores the namespace, and then replaces entry_frame only.

No second optical-flow pass is added.  If the expected frozen namespace is not
present, the function returns the S008 champion unchanged.
"""


_S010_BASE_PREDICT_STAGE2 = predict_stage2
_S010_MOTION_KEY = "_S2_MOTION_PREDICT_STAGE2"


def predict_stage2(data_dir, model_dir):
    namespace = globals().get("_S2_COLLISION_NAMESPACE")
    if not isinstance(namespace, dict):
        return _S010_BASE_PREDICT_STAGE2(data_dir, model_dir)

    motion_predict = namespace.get(_S010_MOTION_KEY)
    if not callable(motion_predict):
        return _S010_BASE_PREDICT_STAGE2(data_dir, model_dir)

    captured = {}

    def capture_motion(*args, **kwargs):
        table = motion_predict(*args, **kwargs)
        captured["table"] = table
        return table

    namespace[_S010_MOTION_KEY] = capture_motion
    try:
        champion = _S010_BASE_PREDICT_STAGE2(data_dir, model_dir).copy()
    finally:
        namespace[_S010_MOTION_KEY] = motion_predict

    motion = captured.get("table")
    if motion is None or "entry_frame" not in motion.columns:
        return champion
    entry_by_id = dict(
        zip(motion["ID"].astype(str), motion["entry_frame"].astype(int))
    )
    champion["entry_frame"] = [
        entry_by_id.get(str(sample_id), int(old_value))
        for sample_id, old_value in zip(
            champion["ID"], champion["entry_frame"]
        )
    ]
    return champion

# CONTROLLED CANDIDATE OVERRIDE
"""S012: replace only Stage 2 evasion_space from cached target detections.

S010's S008 layer already runs SSDLite on a bounded set of frames and anchors
the likely collision target.  This wrapper captures that in-memory target call,
without a second model pass, and asks whether a vehicle-width lateral corridor
remains around the target and other credible near-field vehicles.

The rule is label-free and scale-free.  All geometry is normalized by image
width.  If the expected S008 internals are absent, the target is weak, or the
decision is close to its boundary, the S010 value is preserved.  No field other
than evasion_space is assigned.
"""

from pathlib import Path as _S012Path


_S012_BASE_PREDICT_STAGE2 = predict_stage2
_S012_ROAD_LEFT = 0.05
_S012_ROAD_RIGHT = 0.95
_S012_SAFETY_PAD = 0.03
_S012_MIN_SCORE = 0.25
_S012_RELATIVE_SCORE = 0.50
_S012_MIN_BOTTOM = 0.45
_S012_MIN_MARGIN = 0.035
_S012_MIN_PASSAGE = 0.22
_S012_MAX_PASSAGE = 0.32
_S012_WIDTH_SCALE = 0.55
_S012_WIDTH_OFFSET = 0.08
_S012_LAST_DIAGNOSTICS = {}


def _s012_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return intersection / union if union > 1e-9 else 0.0


def _s012_merge_intervals(intervals):
    merged = []
    for left, right in sorted(intervals):
        left = max(_S012_ROAD_LEFT, float(left))
        right = min(_S012_ROAD_RIGHT, float(right))
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def _s012_clearance_evidence(per_frame, numbers, collision_frame):
    """Return an independent target-aware decision and its diagnostics."""

    anchor_function = globals().get("_s008_anchor")
    if not callable(anchor_function):
        return {"decisive": False, "reason": "missing_anchor_function"}
    anchor = anchor_function(per_frame, numbers, collision_frame)
    if anchor is None:
        return {"decisive": False, "reason": "no_target_anchor"}

    anchor_index, target = anchor
    if anchor_index < 0 or anchor_index >= len(per_frame):
        return {"decisive": False, "reason": "invalid_anchor_index"}
    target_box = tuple(float(value) for value in target["box"])
    x1, y1, x2, y2 = target_box
    target_score = float(target["score"])
    target_width = max(0.0, x2 - x1)
    if target_score < _S012_MIN_SCORE or y2 < _S012_MIN_BOTTOM:
        return {
            "decisive": False,
            "reason": "weak_or_distant_target",
            "target_score": round(target_score, 6),
            "target_bottom": round(y2, 6),
        }

    # Keep the anchored target plus independently located, credible vehicles at
    # comparable depth.  Duplicate COCO class hypotheses for the target are
    # suppressed by overlap before they can consume clearance twice.
    intervals = [
        (x1 - _S012_SAFETY_PAD, x2 + _S012_SAFETY_PAD)
    ]
    blockers = [target_box]
    score_cutoff = max(_S012_MIN_SCORE, _S012_RELATIVE_SCORE * target_score)
    depth_cutoff = max(_S012_MIN_BOTTOM, y1 + 0.04)
    for detection in per_frame[anchor_index]:
        box = tuple(float(value) for value in detection["box"])
        if detection is target or _s012_iou(box, target_box) >= 0.50:
            continue
        if float(detection["score"]) < score_cutoff or float(box[3]) < depth_cutoff:
            continue
        intervals.append(
            (box[0] - _S012_SAFETY_PAD, box[2] + _S012_SAFETY_PAD)
        )
        blockers.append(box)

    occupied = _s012_merge_intervals(intervals)
    cursor = _S012_ROAD_LEFT
    gaps = []
    for left, right in occupied:
        if left > cursor:
            gaps.append(left - cursor)
        cursor = max(cursor, right)
    if cursor < _S012_ROAD_RIGHT:
        gaps.append(_S012_ROAD_RIGHT - cursor)
    largest_gap = max(gaps) if gaps else 0.0

    # A nearer/wider target needs a wider angular corridor for the ego vehicle
    # to pass safely.  Bounds prevent tiny distant boxes or partial close boxes
    # from making the required passage implausibly small or large.
    required = _S012_WIDTH_OFFSET + _S012_WIDTH_SCALE * target_width
    required = min(_S012_MAX_PASSAGE, max(_S012_MIN_PASSAGE, required))
    margin = largest_gap - required
    decisive = abs(margin) >= _S012_MIN_MARGIN
    return {
        "decisive": bool(decisive),
        "reason": "clear_margin" if decisive else "boundary_fallback",
        "prediction": int(margin >= 0.0),
        "anchor_frame": int(numbers[anchor_index]),
        "collision_frame": int(collision_frame),
        "target_box": [round(value, 6) for value in target_box],
        "target_score": round(target_score, 6),
        "target_width": round(target_width, 6),
        "target_bottom": round(y2, 6),
        "blocker_count": len(blockers),
        "occupied_intervals": [
            [round(left, 6), round(right, 6)] for left, right in occupied
        ],
        "largest_gap": round(largest_gap, 6),
        "required_gap": round(required, 6),
        "margin": round(margin, 6),
    }


def _s012_capture_predict_sides(original_predict_sides, evidence_by_id):
    """Wrap S008's pass and capture each folder's target evidence in memory."""

    def capture_predict_sides(data_dir, model_dir, baseline):
        original_tensor = globals().get("_s008_tensor")
        original_track = globals().get("_s008_track_origin")
        if not callable(original_tensor) or not callable(original_track):
            return original_predict_sides(data_dir, model_dir, baseline)

        state = {"sample_id": None}

        def capture_tensor(path):
            tensor, shape = original_tensor(path)
            if tensor is not None:
                state["sample_id"] = _S012Path(path).parent.name
            return tensor, shape

        def capture_track(per_frame, numbers, collision_frame):
            side = original_track(per_frame, numbers, collision_frame)
            sample_id = state.get("sample_id")
            if sample_id is not None:
                try:
                    evidence_by_id[str(sample_id)] = _s012_clearance_evidence(
                        per_frame, numbers, collision_frame
                    )
                except Exception as error:  # preserve S010 on diagnostic failure
                    evidence_by_id[str(sample_id)] = {
                        "decisive": False,
                        "reason": f"capture_error:{type(error).__name__}",
                    }
            return side

        globals()["_s008_tensor"] = capture_tensor
        globals()["_s008_track_origin"] = capture_track
        try:
            return original_predict_sides(data_dir, model_dir, baseline)
        finally:
            globals()["_s008_tensor"] = original_tensor
            globals()["_s008_track_origin"] = original_track

    return capture_predict_sides


def predict_stage2(data_dir, model_dir):
    original_predict_sides = globals().get("_s008_predict_sides")
    if not callable(original_predict_sides):
        return _S012_BASE_PREDICT_STAGE2(data_dir, model_dir)

    evidence_by_id = {}
    globals()["_s008_predict_sides"] = _s012_capture_predict_sides(
        original_predict_sides, evidence_by_id
    )
    try:
        champion = _S012_BASE_PREDICT_STAGE2(data_dir, model_dir).copy()
    finally:
        globals()["_s008_predict_sides"] = original_predict_sides

    global _S012_LAST_DIAGNOSTICS
    _S012_LAST_DIAGNOSTICS = evidence_by_id
    champion["evasion_space"] = [
        int(evidence_by_id[str(sample_id)]["prediction"])
        if evidence_by_id.get(str(sample_id), {}).get("decisive", False)
        else int(old_value)
        for sample_id, old_value in zip(champion["ID"], champion["evasion_space"])
    ]
    return champion

# CONTROLLED CANDIDATE OVERRIDE
"""S011: supervised Stage 3 challenger (LK features + numpy MLP heads).

Self-contained drop-in replacement for Baseline/inference.py::predict_stage3.
Dependencies: numpy, pandas, opencv-python (all pinned by
Baseline/requirements.txt). No torch, no sklearn, no network.

Contract (identical to baseline and s003)
-----------------------------------------
predict_stage3(data_dir, model_dir) -> DataFrame with columns
["ID", "sample_index", "accel_label", "steer_label"], one row per decoded
frame of every video under <data_dir>/videos, sample_index = decoded frame
index. On the officially stated hidden 10 Hz cadence this is exactly the
required sample_index series.

Model (hybrid)
--------------
Per-frame sparse-LK ego-motion features (same geometry as s003) are summed
into per-sample 14-dim window features. Acceleration uses the s003 threshold
classifier (best accel head in route-grouped CV); steering uses the MLP head
when weights are found (macro-F1 0.43 CV vs 0.26 thresholds), forced STRAIGHT
on STOPPED frames. Weights were trained on comma2k19 Chunk_1 CAN-derived
labels calibrated against the 50 official public rows (accel 48/50, steer
49/50); see candidates/s011_stage3_supervised/INTEGRATION.md.

Weight lookup order: model_dir, then next to this file, then the candidate
artifacts directory. If no weights are found the module falls back to the
s003 threshold classifier so it never fails open.
"""


from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]

HIDDEN_DT_SECONDS = 0.1
PUBLIC_DT_SECONDS = 0.05
MODEL_FILENAME = "stage3_supervised_model.npz"


def video_paths(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXT
    )


def _prepare_gray(frame, work_width):
    height, width = frame.shape[:2]
    if width > work_width:
        scale = work_width / float(width)
        frame = cv2.resize(
            frame, (work_width, max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _feature_mask(shape):
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    top = round(height * 0.10)
    bottom = height - round(height * 0.16)
    if bottom <= top:
        top, bottom = 0, height
    mask[top:bottom, :] = 255
    return mask


def extract_video_features(path):
    """Per-decoded-frame (speed, yaw, expansion, tracked), s003 geometry."""

    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {Path(path).name}")
    work_width = 320
    speed, yaw, expansion, tracked = [], [], [], []
    previous = None
    mask = None
    center = None
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            decoded += 1
            gray = _prepare_gray(frame, work_width)
            if previous is None:
                previous = gray
                mask = _feature_mask(gray.shape[:2])
                center = np.array(
                    [gray.shape[1] / 2.0, gray.shape[0] / 2.0], dtype=np.float64
                )
                continue
            height, width = gray.shape[:2]
            corners = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=150,
                qualityLevel=0.01,
                minDistance=8,
                mask=mask,
                blockSize=7,
            )
            frame_speed = frame_yaw = frame_expansion = frame_tracked = 0.0
            if corners is not None and len(corners) >= 3:
                lk = {
                    "winSize": (15, 15),
                    "maxLevel": 3,
                    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                }
                forward, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous, gray, corners, None, **lk
                )
                if forward is not None:
                    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
                        gray, previous, forward, None, **lk
                    )
                    good = status.reshape(-1).astype(bool)
                    if backward is not None and status_back is not None:
                        good &= status_back.reshape(-1).astype(bool)
                        error = np.linalg.norm(
                            corners.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
                        )
                        good &= error <= 1.0
                    source = corners.reshape(-1, 2)[good].astype(np.float64)
                    target = forward.reshape(-1, 2)[good].astype(np.float64)
                    frame_tracked = float(len(source))
                    if len(source) >= 12:
                        displacement = np.linalg.norm(target - source, axis=1)
                        frame_speed = float(np.median(displacement)) / float(height)
                        fit = _fit_similarity(source, target)
                        if fit is not None:
                            scale, theta, translation = fit
                            rotation = np.array(
                                [
                                    [np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)],
                                ]
                            )
                            moved = (scale * rotation) @ center + translation
                            frame_yaw = float(moved[0] - center[0]) / float(width)
                            frame_expansion = float(scale) - 1.0
            speed.append(frame_speed)
            yaw.append(frame_yaw)
            expansion.append(frame_expansion)
            tracked.append(frame_tracked)
            previous = gray
    finally:
        capture.release()

    if decoded == 0:
        raise ValueError(f"cannot decode video: {Path(path).name}")
    if not speed:
        zeros = np.zeros(decoded, dtype=np.float64)
        return {name: zeros.copy() for name in ("speed", "yaw", "expansion", "tracked")}

    def head_padded(values):
        array = np.asarray(values, dtype=np.float64)
        return np.concatenate(([array[0]], array))

    return {
        "speed": head_padded(speed)[:decoded],
        "yaw": head_padded(yaw)[:decoded],
        "expansion": head_padded(expansion)[:decoded],
        "tracked": head_padded(tracked)[:decoded],
    }


def _fit_similarity(source, target):
    def fit(p, q):
        if len(p) < 3:
            return None
        p_mean = p.mean(axis=0)
        q_mean = q.mean(axis=0)
        dp = p - p_mean
        dq = q - q_mean
        denominator = float(np.sum(dp * dp))
        if denominator <= 1e-9:
            return None
        a = float(np.sum(dp * dq))
        b = float(np.sum(dp[:, 0] * dq[:, 1] - dp[:, 1] * dq[:, 0]))
        scale = float(np.hypot(a, b) / denominator)
        theta = float(np.arctan2(b, a))
        if not np.isfinite(scale) or scale <= 1e-6:
            return None
        matrix = scale * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return scale, theta, q_mean - matrix @ p_mean, matrix

    first = fit(source, target)
    if first is None:
        return None
    _, _, translation, matrix = first
    residual = np.linalg.norm(target - (source @ matrix.T + translation), axis=1)
    median = float(np.median(residual))
    scale_mad = float(np.median(np.abs(residual - median))) * 1.4826
    if scale_mad > 1e-6:
        keep = residual <= median + 2.5 * scale_mad
        if int(keep.sum()) >= max(3, len(source) // 4):
            second = fit(source[keep], target[keep])
            if second is not None:
                first = second
    return first[0], first[1], first[2]


def build_window_features(feat, dt):
    """14-dim per-sample features; sample s aligns to frame s/(10*dt)."""

    speed = feat["speed"].astype(np.float64) / max(dt, 1e-9)
    yaw = feat["yaw"].astype(np.float64) / max(dt, 1e-9)
    expansion = feat["expansion"].astype(np.float64)
    tracked = feat["tracked"].astype(np.float64)
    n_frames = len(speed)
    step = int(round(0.1 / dt))
    n_samples = max(0, n_frames // step)
    half = int(round(1.0 / dt))
    back075 = int(round(0.75 / dt))
    back05 = int(round(0.5 / dt))
    rows = np.empty((n_samples, 14), dtype=np.float64)
    for s in range(n_samples):
        f = min(s * step, n_frames - 1)
        lo, hi = max(0, f - half), min(n_frames, f + half + 1)
        w_speed = speed[lo:hi]
        fwd = min(n_frames - 1, f + back075)
        back = max(0, f - back075)
        span = max((fwd - back) * dt, 1e-9)
        s_slope = (speed[fwd] - speed[back]) / span
        s_mean = max(float(np.mean(w_speed)), 1e-3)
        rows[s] = (
            float(np.mean(w_speed)),
            float(np.std(w_speed)),
            float(np.min(w_speed)),
            float(np.max(w_speed)),
            float(speed[f]),
            s_slope,
            float(np.mean(yaw[lo:hi])),
            float(np.std(yaw[lo:hi])),
            float(np.mean(np.abs(yaw[lo:hi]))),
            float(np.median(yaw[max(0, f - back05): f + 1])),
            float(np.mean(expansion[lo:hi])),
            float(np.mean(tracked[lo:hi])),
            s_slope / s_mean,
            s_mean,
        )
    return rows


def _load_weights(model_dir):
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir) / MODEL_FILENAME)
    here = Path(__file__).resolve().parent
    candidates.append(here / MODEL_FILENAME)
    candidates.append(here / "artifacts" / "stage3_model.npz")
    for path in candidates:
        if path.is_file():
            return np.load(path)
    return None


def _mlp_predict(X, weights):
    mu = weights["mu"].astype(np.float64)
    sd = weights["sd"].astype(np.float64)
    hidden = np.maximum((X - mu) / sd @ weights["w1"].T.astype(np.float64) + weights["b1"], 0.0)
    accel_logits = hidden @ weights["wa"].T.astype(np.float64) + weights["ba"]
    steer_logits = hidden @ weights["ws"].T.astype(np.float64) + weights["bs"]
    return accel_logits.argmax(axis=1), steer_logits.argmax(axis=1)


def _threshold_classify(feat, dt):
    """s003 fallback: default thresholds on the same per-frame features."""

    raw_speed = feat["speed"] / max(dt, 1e-9)
    raw_yaw = feat["yaw"] / max(dt, 1e-9)
    count = len(raw_speed)
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def median_filter(values, window):
        window = max(1, window if window % 2 == 1 else window + 1)
        half = window // 2
        padded = np.pad(values, half, mode="edge")
        return np.median(
            np.lib.stride_tricks.sliding_window_view(padded, window), axis=1
        )

    def moving_average(values, window):
        window = max(1, window if window % 2 == 1 else window + 1)
        half = window // 2
        padded = np.pad(values, half, mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")

    wf = lambda seconds: max(1, round(seconds / dt))  # noqa: E731
    speed = moving_average(median_filter(raw_speed, wf(1.5)), wf(1.5))
    yaw_s = moving_average(median_filter(raw_yaw, wf(1.5)), wf(1.5))
    lag = wf(1.0)
    fwd = np.clip(np.arange(count) + lag, 0, count - 1)
    back = np.clip(np.arange(count) - lag, 0, count - 1)
    span = np.maximum((fwd - back) * dt, dt)
    slope = (speed[fwd] - speed[back]) / span
    relative = slope / np.maximum((speed[fwd] + speed[back]) / 2.0, 0.01)
    accel = np.full(count, 2, dtype=np.int64)
    accel[relative > 0.20] = 0
    accel[relative < -0.12] = 1
    accel[speed < 0.005] = 3
    steer = np.full(count, 1, dtype=np.int64)
    steer[yaw_s > 0.014] = 0
    steer[yaw_s < -0.014] = 2

    def mode_filter(labels, window, classes):
        window = max(1, window if window % 2 == 1 else window + 1)
        half = window // 2
        padded = np.pad(labels, half, mode="edge")
        strided = np.lib.stride_tricks.sliding_window_view(padded, window)
        counts = np.stack([(strided == v).sum(axis=1) for v in range(classes)], axis=1)
        return counts.argmax(axis=1)

    accel = mode_filter(accel, wf(0.9), 4)
    steer = mode_filter(steer, wf(0.9), 3)
    return accel, steer


def classify_video(feat, dt, weights=None):
    """Hybrid per-video labels at cadence dt.

    accel: s003 threshold classifier (route-grouped CV best accel head).
    steer: MLP head when weights available, else thresholds; forced
    STRAIGHT wherever accel says STOPPED. Returns indices + sample count.
    """
    X = build_window_features(feat, dt)
    accel_idx, steer_idx = _threshold_classify(feat, dt)
    step = int(round(0.1 / dt))
    accel_idx = accel_idx[::step][: len(X)]
    steer_idx = steer_idx[::step][: len(X)]
    if weights is not None:
        _, mlp_steer = _mlp_predict(X, weights)
        steer_idx = np.where(accel_idx == 3, 1, mlp_steer)
    return accel_idx, steer_idx, len(X)


def _fallback_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while capture.grab():
            decoded += 1
    finally:
        capture.release()
    return decoded


def predict_stage3(data_dir, model_dir=None) -> pd.DataFrame:
    """One row per decoded frame per video, ordered by video then sample_index."""

    weights = _load_weights(model_dir)
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames: list[pd.DataFrame] = []
    for path in videos:
        try:
            feat = extract_video_features(path)
            accel_idx, steer_idx, count = classify_video(
                feat, HIDDEN_DT_SECONDS, weights
            )
            if count == 0:
                continue
        except Exception:  # noqa: BLE001 - one bad video must not abort a submission
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel_idx = np.full(count, 2, dtype=np.int64)
            steer_idx = np.ones(count, dtype=np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(v)] for v in accel_idx],
                    "steer_label": [STEER[int(v)] for v in steer_idx],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["ID", "sample_index", "accel_label", "steer_label"])
    return pd.concat(frames, ignore_index=True)[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ]

# CONTROLLED CANDIDATE OVERRIDE
"""S015 Stage 3 override: five-seed steering ensemble + temporal decode.

This file is appended to the validated S014 submission.  It deliberately
reuses S014's optical-flow feature extractor and threshold acceleration head,
changing only the supervised steering decision path.
"""

_S015_MODEL_FILENAME = "stage3_temporal_ensemble.npz"
_S015_STEER_WINDOW_SECONDS = 2.0
_S015_STEER_LOG_BIAS = (-0.5, 0.0, -0.5)


def _s015_load_weights(model_dir):
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir) / _S015_MODEL_FILENAME)
    candidates.append(Path(__file__).resolve().parent / _S015_MODEL_FILENAME)
    for path in candidates:
        if path.is_file():
            weights = np.load(path)
            required = {"w1", "b1", "ws", "bs", "mu", "sd"}
            if required.issubset(weights.files):
                return weights
    return None


def _s015_softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _s015_steer_probabilities(features, weights):
    mu = weights["mu"].astype(np.float64)
    sd = weights["sd"].astype(np.float64)
    normalized = (features - mu) / sd
    probabilities = np.zeros((len(features), 3), dtype=np.float64)
    w1 = weights["w1"].astype(np.float64)
    b1 = weights["b1"].astype(np.float64)
    ws = weights["ws"].astype(np.float64)
    bs = weights["bs"].astype(np.float64)
    for seed in range(w1.shape[0]):
        hidden = np.maximum(normalized @ w1[seed].T + b1[seed], 0.0)
        probabilities += _s015_softmax(hidden @ ws[seed].T + bs[seed])
    return probabilities / float(w1.shape[0])


def _s015_smooth_probabilities(probabilities, window):
    if len(probabilities) == 0 or window <= 1:
        return probabilities
    window = int(window)
    if window % 2 == 0:
        window += 1
    half = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    output = np.empty_like(probabilities, dtype=np.float64)
    for class_index in range(probabilities.shape[1]):
        padded = np.pad(probabilities[:, class_index], half, mode="edge")
        output[:, class_index] = np.convolve(padded, kernel, mode="valid")
    return output


def _s015_classify_video(features, dt, weights):
    sample_features = build_window_features(features, dt)
    threshold_accel, _threshold_steer = _threshold_classify(features, dt)
    step = int(round(0.1 / dt))
    accel = threshold_accel[::step][: len(sample_features)]
    if weights is None:
        old_weights = _load_weights(None)
        return classify_video(features, dt, old_weights)

    probabilities = _s015_steer_probabilities(sample_features, weights)
    window = int(round(_S015_STEER_WINDOW_SECONDS / 0.1))
    probabilities = _s015_smooth_probabilities(probabilities, window)
    logits = np.log(np.maximum(probabilities, 1e-9))
    logits += np.asarray(_S015_STEER_LOG_BIAS, dtype=np.float64)
    steer = logits.argmax(axis=1).astype(np.int64)
    # Deliberately do not force STRAIGHT on predicted STOPPED rows.  Official
    # steering is masked by the *true* acceleration label, and this choice won
    # on every route-disjoint fold.
    return accel, steer, len(sample_features)


def predict_stage3(data_dir, model_dir=None):
    weights = _s015_load_weights(model_dir)
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames = []
    for path in videos:
        try:
            features = extract_video_features(path)
            accel_idx, steer_idx, count = _s015_classify_video(
                features, HIDDEN_DT_SECONDS, weights
            )
            if count == 0:
                continue
        except Exception:
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel_idx = np.full(count, 2, dtype=np.int64)
            steer_idx = np.ones(count, dtype=np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(value)] for value in accel_idx],
                    "steer_label": [STEER[int(value)] for value in steer_idx],
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["ID", "sample_index", "accel_label", "steer_label"]
        )
    return pd.concat(frames, ignore_index=True)[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ]

# CONTROLLED CANDIDATE OVERRIDE
"""Rich temporal features for the s019 acceleration head.

All features are computed from the same per-frame LK channels that the
packaged Stage 3 extractor already produces (speed / yaw / expansion /
tracked), so hidden-set inference needs no new extractor.  Every horizon
is expressed in seconds and converted with the active dt, keeping the
module correct at both the public 20 fps cadence and the hidden 10 Hz
cadence.

Output cadence matches s011 build_window_features: one 10 Hz row per
sample, row s aligned to frame s * round(0.1 / dt).
"""


import numpy as np

RICH_FEATURE_NAMES = [
    "fwd_slope_025",
    "fwd_slope_050",
    "fwd_slope_100",
    "fwd_slope_200",
    "fwd_slope_300",
    "bwd_slope_025",
    "bwd_slope_050",
    "bwd_slope_100",
    "bwd_slope_200",
    "bwd_slope_300",
    "ctr_slope_100",
    "ctr_slope_200",
    "rel_fwd_slope_100",
    "speed_med_100",
    "speed_std_200",
    "expansion_mean_050",
    "expansion_mean_100",
    "expansion_slope_100",
    "tracked_mean_100",
    "yaw_mean_100",
    "yaw_absmean_100",
    "speed_level_wide",
]

_HORIZONS_S = (0.25, 0.5, 1.0, 2.0, 3.0)


def _frame_at(value_s, dt, n_frames):
    return int(round(value_s / dt))


def build_rich_features(feat, dt):
    """Return (n_samples, len(RICH_FEATURE_NAMES)) on the 10 Hz grid."""

    speed = np.asarray(feat["speed"], dtype=np.float64) / max(dt, 1e-9)
    yaw = np.asarray(feat["yaw"], dtype=np.float64) / max(dt, 1e-9)
    expansion = np.asarray(feat["expansion"], dtype=np.float64)
    tracked = np.asarray(feat["tracked"], dtype=np.float64)
    n_frames = len(speed)
    if n_frames == 0:
        return np.zeros((0, len(RICH_FEATURE_NAMES)), dtype=np.float64)
    step = int(round(0.1 / dt))
    n_samples = max(0, n_frames // step)
    rows = np.empty((n_samples, len(RICH_FEATURE_NAMES)), dtype=np.float64)

    fwd_frames = [_frame_at(h, dt, n_frames) for h in _HORIZONS_S]
    for s in range(n_samples):
        f = min(s * step, n_frames - 1)
        out = []
        for h_f in fwd_frames:
            span = max(h_f * dt, 1e-9)
            out.append((speed[min(f + h_f, n_frames - 1)] - speed[f]) / span)
        for h_f in fwd_frames:
            span = max(h_f * dt, 1e-9)
            out.append((speed[f] - speed[max(f - h_f, 0)]) / span)
        for h_s in (1.0, 2.0):
            h_f = _frame_at(h_s, dt, n_frames)
            span = max(2 * h_f * dt, 1e-9)
            out.append(
                (speed[min(f + h_f, n_frames - 1)] - speed[max(f - h_f, 0)])
                / span
            )
        h1 = _frame_at(1.0, dt, n_frames)
        h2 = _frame_at(2.0, dt, n_frames)
        h05 = _frame_at(0.5, dt, n_frames)
        w1 = speed[max(f - h1, 0): min(f + h1, n_frames - 1) + 1]
        w2 = speed[max(f - h2, 0): min(f + h2, n_frames - 1) + 1]
        w05 = expansion[max(f - h05, 0): min(f + h05, n_frames - 1) + 1]
        w1_exp = expansion[max(f - h1, 0): min(f + h1, n_frames - 1) + 1]
        w1_yaw = yaw[max(f - h1, 0): min(f + h1, n_frames - 1) + 1]
        w1_trk = tracked[max(f - h1, 0): min(f + h1, n_frames - 1) + 1]
        speed_med_1 = max(float(np.median(w1)), 1e-3)
        fwd_slope_1 = out[2]
        out.append(fwd_slope_1 / speed_med_1)
        out.append(float(np.median(w1)))
        out.append(float(np.std(w2)))
        out.append(float(np.mean(w05)))
        out.append(float(np.mean(w1_exp)))
        exp_span = max(2 * h1 * dt, 1e-9)
        out.append(
            (expansion[min(f + h1, n_frames - 1)] - expansion[max(f - h1, 0)])
            / exp_span
        )
        out.append(float(np.mean(w1_trk)))
        out.append(float(np.mean(w1_yaw)))
        out.append(float(np.mean(np.abs(w1_yaw))))
        wide_mean = max(float(np.mean(w2)), 1e-3)
        out.append(speed[f] / wide_mean)
        rows[s] = out
    return rows


def build_threshold_indicators(feat, dt, classify_features, params):
    """4-dim one-hot of the s003 threshold head on the 10 Hz grid."""

    frame_accel, _ = classify_features(feat, dt, params)
    step = int(round(0.1 / dt))
    sampled = frame_accel[::step]
    n = len(sampled)
    out = np.zeros((n, 4), dtype=np.float64)
    if n:
        out[np.arange(n), sampled.astype(np.int64)] = 1.0
    return out


# CONTROLLED CANDIDATE OVERRIDE
"""S019 Stage 3 override: supervised acceleration plus exact S015 steering.

Append this module after S015 and ``features_rich.py`` in the submission's
``inference.py``.  The optical-flow extraction pass is unchanged; S019 only
adds temporal feature engineering and a small three-seed NumPy MLP.
"""

_S019_ACCEL_MODEL_FILENAME = "stage3_accel_mlp.npz"
_S019_ACCEL_WINDOW_SECONDS = 1.0
_S019_ACCEL_LOG_BIAS = (-0.25, -0.5, 0.0, -0.75)


def _s019_load_accel_weights(model_dir):
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir) / _S019_ACCEL_MODEL_FILENAME)
    candidates.append(Path(__file__).resolve().parent / _S019_ACCEL_MODEL_FILENAME)
    for path in candidates:
        if path.is_file():
            weights = np.load(path)
            required = {"w1", "b1", "w2", "b2", "w3", "b3", "mu", "sd"}
            if required.issubset(weights.files):
                return weights
            weights.close()
    return None


def _s019_accel_probabilities(features, weights):
    mu = weights["mu"].astype(np.float64)
    sd = weights["sd"].astype(np.float64)
    normalized = (features - mu) / np.maximum(sd, 1e-8)
    probabilities = np.zeros((len(features), 4), dtype=np.float64)
    w1 = weights["w1"].astype(np.float64)
    b1 = weights["b1"].astype(np.float64)
    w2 = weights["w2"].astype(np.float64)
    b2 = weights["b2"].astype(np.float64)
    w3 = weights["w3"].astype(np.float64)
    b3 = weights["b3"].astype(np.float64)
    for seed in range(w1.shape[0]):
        hidden1 = np.maximum(normalized @ w1[seed].T + b1[seed], 0.0)
        hidden2 = np.maximum(hidden1 @ w2[seed].T + b2[seed], 0.0)
        probabilities += _s015_softmax(hidden2 @ w3[seed].T + b3[seed])
    return probabilities / float(w1.shape[0])


def _s019_feature_matrix(features, dt):
    sparse = build_window_features(features, dt)
    rich = build_rich_features(features, dt)
    threshold_accel, _ = _threshold_classify(features, dt)
    step = max(1, int(round(0.1 / dt)))
    threshold_accel = threshold_accel[::step]
    count = min(len(sparse), len(rich), len(threshold_accel))
    sparse = sparse[:count]
    rich = rich[:count]
    threshold_onehot = np.zeros((count, 4), dtype=np.float64)
    if count:
        threshold_onehot[np.arange(count), threshold_accel[:count].astype(np.int64)] = 1.0
    return np.concatenate([sparse, rich, threshold_onehot], axis=1), sparse


def _s019_classify_video(features, dt, accel_weights, steer_weights):
    combined, sparse = _s019_feature_matrix(features, dt)
    count = len(combined)
    if count == 0 or accel_weights is None or steer_weights is None:
        return _s015_classify_video(features, dt, steer_weights)

    accel_probs = _s019_accel_probabilities(combined, accel_weights)
    accel_window = int(round(_S019_ACCEL_WINDOW_SECONDS / 0.1))
    accel_probs = _s015_smooth_probabilities(accel_probs, accel_window)
    accel_logits = np.log(np.maximum(accel_probs, 1e-9))
    accel_logits += np.asarray(_S019_ACCEL_LOG_BIAS, dtype=np.float64)
    accel = accel_logits.argmax(axis=1).astype(np.int64)

    steer_probs = _s015_steer_probabilities(sparse, steer_weights)
    steer_window = int(round(_S015_STEER_WINDOW_SECONDS / 0.1))
    steer_probs = _s015_smooth_probabilities(steer_probs, steer_window)
    steer_logits = np.log(np.maximum(steer_probs, 1e-9))
    steer_logits += np.asarray(_S015_STEER_LOG_BIAS, dtype=np.float64)
    steer = steer_logits.argmax(axis=1).astype(np.int64)
    return accel, steer, count


def predict_stage3(data_dir, model_dir=None):
    accel_weights = _s019_load_accel_weights(model_dir)
    steer_weights = _s015_load_weights(model_dir)
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames = []
    for path in videos:
        try:
            features = extract_video_features(path)
            accel_idx, steer_idx, count = _s019_classify_video(
                features, HIDDEN_DT_SECONDS, accel_weights, steer_weights
            )
            if count == 0:
                continue
        except Exception:
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel_idx = np.full(count, 2, dtype=np.int64)
            steer_idx = np.ones(count, dtype=np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(value)] for value in accel_idx],
                    "steer_label": [STEER[int(value)] for value in steer_idx],
                }
            )
        )
    if accel_weights is not None:
        accel_weights.close()
    if steer_weights is not None:
        steer_weights.close()
    if not frames:
        return pd.DataFrame(
            columns=["ID", "sample_index", "accel_label", "steer_label"]
        )
    return pd.concat(frames, ignore_index=True)[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ]

# CONTROLLED CANDIDATE OVERRIDE
"""S036 Stage-1 inference override: compact log-polar/radial recapture model.

Appendable override for the official ZIP via tools/build_candidate_submission.py
(pass this file as --override ... --stage stage1 plus --extra-file
<local-model.npz> model/stage1/s036_model.npz).

Contract: predict_stage1(data_dir, model_dir) returns one ID/answer row per
video under data_dir/videos with columns [ID, answer] drawn from
ORIGINAL/RERECORDED. The trained artifact is required for learned evidence;
any artifact problem or undecodable individual video falls back to the
already-integrated answer captured at append time (the existing S002/S019
chain) and finally to the conservative ORIGINAL label, so the call never
drops a row.

Deployment rule (explicit, env-tunable): the anchored poly2 model is primary;
when its probability sits within S036_AMBIGUITY_MARGIN (default 0.15) of the
frozen threshold, the S002 forensic answer is used as the fallback. The
S036_DEPLOYMENT_MODE environment variable selects hybrid (default), learned,
or s002.

Self-contained: only numpy/opencv/pandas plus the standard library. Never
imports sibling repository files, never touches absolute D: paths, performs
no dynamic source imports. The feature operator, grid sampler and classifier
maths mirror features.py/model.py (training-time sources of truth). No
ID/path/codec/cross-file statistics are used as model features: every score
comes from pixels alone.

The bundled model is a compact poly2 artifact; it has no dependency on the
training tree or on absolute local paths at submission time.
"""


import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    _S036_BASE_PREDICT_STAGE1 = predict_stage1  # noqa: F821 -- captured at append time
except NameError:  # standalone import (tests/dev without a prior override)
    _S036_BASE_PREDICT_STAGE1 = None

_S036_CROP_SIZE = 512
_S036_SHORT_SIDE = 512
_S036_MAX_VIDEO_FRAMES = 8
_S036_GRID_CROPS_PER_FRAME = 4
_S036_MAX_VIDEO_PATCHES = 32
_S036_EPS = 1e-9
_S036_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
_S036_PATCH_FEATURE_NAMES = (
    "ring_peak_db", "ring_density", "axis_peak_db", "radial_concentration",
    "high_ratio", "mid_high_ratio", "spectral_slope", "angular_anisotropy",
    "row_periodicity", "col_periodicity", "residual_mad", "residual_energy",
)
_S036_FEATURE_NAMES = tuple(s + "_" + f for s in ("mean", "std") for f in _S036_PATCH_FEATURE_NAMES)
_S036_VALID_ANSWERS = ("ORIGINAL", "RERECORDED")


def _s036_deterministic_indices(count: int, limit: int = _S036_MAX_VIDEO_FRAMES) -> list:
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    return [int(v) for v in sorted({round(v) for v in np.linspace(0, count - 1, limit)})]


def _s036_normalize_short_side(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if min(h, w) >= _S036_SHORT_SIDE:
        return gray
    scale = _S036_SHORT_SIDE / float(min(h, w))
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def _s036_crop_gray(gray: np.ndarray, x: int, y: int, size: int = _S036_CROP_SIZE) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("expected grayscale image, got shape " + str(gray.shape))
    h, w = gray.shape
    x, y = int(x), int(y)
    if x < 0 or y < 0 or x >= w or y >= h:
        raise ValueError("crop origin outside image")
    patch = gray[y:min(y + size, h), x:min(x + size, w)]
    pb, pr = size - patch.shape[0], size - patch.shape[1]
    if pb or pr:
        patch = cv2.copyMakeBorder(patch, 0, pb, 0, pr, cv2.BORDER_REFLECT_101)
    return patch.astype(np.float64, copy=False)


def _s036_radial_grid(shape) -> np.ndarray:
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    return np.hypot(fy, fx) / 0.5


def _s036_periodicity(signal2d: np.ndarray, axis: int) -> float:
    profile = np.mean(np.abs(signal2d), axis=axis)
    profile = profile - np.mean(profile)
    if float(np.std(profile)) < _S036_EPS:
        return 0.0
    spec = np.abs(np.fft.rfft(profile * np.hanning(profile.size))) ** 2
    if spec.size < 3:
        return 0.0
    return float(np.max(spec[2:]) / (np.sum(spec[1:]) + _S036_EPS))


def _s036_anisotropy(residual: np.ndarray, radius: np.ndarray, sectors: int = 8) -> float:
    h, w = residual.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    band = (radius > 0.12) & (radius < 0.75)
    if not band.any():
        return 0.0
    means = []
    for index in range(sectors):
        mask = band & (angle >= index * np.pi / sectors) & (angle < (index + 1) * np.pi / sectors)
        means.append(float(np.mean(np.abs(residual[mask]))) if mask.any() else 0.0)
    return float(np.std(np.asarray(means)))


def _s036_logpolar_concentration(power: np.ndarray) -> float:
    h, w = power.shape
    total = float(power.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    normed = (power / total).astype(np.float32)
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    magnitude = w / math.log(math.hypot(w, h) / 2.0 + _S036_EPS)
    warped = cv2.logPolar(normed, center, magnitude, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    if not np.all(np.isfinite(warped)):
        return 0.0
    profile = warped.mean(axis=0)
    return float(profile[w // 2:].sum() / (profile.sum() + _S036_EPS))


def _s036_patch_features(gray_patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(gray_patch, dtype=np.float64)
    if patch.shape != (_S036_CROP_SIZE, _S036_CROP_SIZE):
        patch = cv2.resize(patch, (_S036_CROP_SIZE, _S036_CROP_SIZE), interpolation=cv2.INTER_AREA)
    patch = patch - cv2.GaussianBlur(patch, (0, 0), 7.0)
    patch = patch / (np.median(np.abs(patch)) * 1.4826 + 1e-3)
    window = np.outer(np.hanning(_S036_CROP_SIZE), np.hanning(_S036_CROP_SIZE))
    power = np.abs(np.fft.fftshift(np.fft.fft2(patch * window))) ** 2
    radius = _s036_radial_grid(patch.shape)
    log_power = np.log10(power + _S036_EPS)
    residual = np.zeros_like(log_power)
    edges = np.linspace(0.05, 0.95, 25)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        if mask.any():
            residual[mask] = log_power[mask] - np.median(log_power[mask])
    valid = (radius >= 0.10) & (radius <= 0.92)
    values = residual[valid]
    peak = float(np.percentile(values, 99.8)) * 10.0
    density = float(np.mean(values > 1.0))
    centre = _S036_CROP_SIZE // 2
    rr, cc = np.indices(radius.shape)
    axis_mask = (((np.abs(rr - centre) <= 2) | (np.abs(cc - centre) <= 2)) & valid)
    axis_peak = float(np.percentile(residual[axis_mask], 99.5)) * 10.0 if axis_mask.any() else 0.0
    total = float(power[(radius > 0.03) & (radius < 1.0)].sum()) + _S036_EPS
    high = float(power[(radius > 0.48) & (radius < 0.92)].sum()) / total
    mid_high = float(power[(radius > 0.30) & (radius <= 0.48)].sum()) / total
    radial_means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        radial_means.append(float(np.mean(log_power[mask])) if mask.any() else 0.0)
    slope = float(np.polyfit(np.arange(len(radial_means), dtype=float), np.asarray(radial_means), 1)[0])
    out = np.array([peak, density, axis_peak, _s036_logpolar_concentration(power),
                    high, mid_high, slope, _s036_anisotropy(residual, radius),
                    _s036_periodicity(patch, 0), _s036_periodicity(patch, 1),
                    float(np.median(np.abs(values))), float(np.mean(values * values))],
                   dtype=np.float64)
    if out.shape != (len(_S036_PATCH_FEATURE_NAMES),) or not np.all(np.isfinite(out)):
        raise ValueError("non-finite S036 patch features")
    return out


def _s036_aggregate_patches(patches: list) -> np.ndarray:
    if not patches:
        raise ValueError("no image crops available for feature extraction")
    matrix = np.stack([_s036_patch_features(p) for p in patches])
    return np.concatenate([matrix.mean(axis=0), matrix.std(axis=0)])


def _s036_frame_grid_origins(height: int, width: int, count: int, size: int = _S036_CROP_SIZE) -> list:
    if count <= 0:
        raise ValueError("grid crop count must be positive")
    if count == 1:
        return [(max(0, (width - size) // 2), max(0, (height - size) // 2))]
    side = math.ceil(math.sqrt(count))
    xs = np.linspace(0, max(0, width - size), side).round().astype(int)
    ys = np.linspace(0, max(0, height - size), side).round().astype(int)
    origins = [(int(x), int(y)) for y in ys for x in xs][:count]
    if len(origins) < count:
        origins += [origins[-1]] * (count - len(origins))
    return origins


def _s036_read_video_frames(path) -> list:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError("cannot open video: " + str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = set(_s036_deterministic_indices(total, _S036_MAX_VIDEO_FRAMES)) if total > 0 else None
        frames, index = [], 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if picks is None or index in picks:
                frames.append(frame)
            index += 1
            if picks is not None and index > max(picks):
                break
            if picks is None and len(frames) >= _S036_MAX_VIDEO_FRAMES:
                break
    finally:
        capture.release()
    if not frames:
        raise ValueError("cannot decode any frame: " + str(path))
    return frames


def _s036_video_feature_vector(path) -> np.ndarray:
    frames = _s036_read_video_frames(Path(path))
    per_frame = max(1, min(_S036_GRID_CROPS_PER_FRAME,
                           _S036_MAX_VIDEO_PATCHES // max(1, len(frames))))
    patches = []
    for frame in frames:
        arr = np.asarray(frame)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
        gray = _s036_normalize_short_side(gray)
        height, width = gray.shape[:2]
        for x, y in _s036_frame_grid_origins(height, width, per_frame):
            patches.append(_s036_crop_gray(gray, x, y, _S036_CROP_SIZE))
            if len(patches) >= _S036_MAX_VIDEO_PATCHES:
                break
        if len(patches) >= _S036_MAX_VIDEO_PATCHES:
            break
    return _s036_aggregate_patches(patches)

# --- S036 anchored nonlinear deployment ---
# The S035 poly2 artifact is fitted with the ten DACON fixture examples as
# explicit training anchors. They are calibration anchors, never validation.
_S036_POLY_FEATURE_COUNT = 24
_S036_POLY_DIM = 24 + 24 * 25 // 2
_S036_ARTIFACT_KEYS = ("mean", "scale", "weights", "bias", "threshold", "family", "feature_names")

def _s036_poly2_expand(z):
    z = np.asarray(z, dtype=np.float64)
    cols = [z]
    for i in range(z.shape[1]):
        cols.append(z[:, i:i + 1] * z[:, i:])
    return np.concatenate(cols, axis=1)

class _S036AnchoredClassifier:
    def __init__(self, data):
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.bias = float(data["bias"])
        self.threshold = float(data["threshold"])
        self.family = str(np.asarray(data["family"]).item()) if "family" in data else ""
        if self.mean.shape != (24,) or self.scale.shape != (24,) or self.weights.shape != (_S036_POLY_DIM,):
            raise ValueError("malformed S036 anchored poly2 artifact")
        if self.family != "poly2_logistic":
            raise ValueError("unexpected S036 family: " + self.family)
        if (not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale))
                or not np.all(np.isfinite(self.weights)) or not np.isfinite(self.bias)
                or not np.isfinite(self.threshold) or np.any(self.scale <= 0.0)
                or not 0.0 <= self.threshold <= 1.0):
            raise ValueError("non-finite or invalid S036 artifact")
        if "feature_names" not in data:
            raise ValueError("S036 artifact is missing feature_names")
        self.feature_names = tuple(str(v) for v in np.asarray(data["feature_names"]).tolist())
        if self.feature_names != _S036_FEATURE_NAMES:
            raise ValueError("S036 artifact feature_names order does not match the feature implementation")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("S036 artifact missing: " + str(path))
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in _S036_ARTIFACT_KEYS if key not in data]
            if missing:
                raise ValueError("S036 artifact missing keys: " + str(missing))
            return cls({key: data[key] for key in data.files})

    def probabilities(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 24:
            raise ValueError("S036 feature width mismatch")
        z = (x - self.mean) / self.scale
        design = _s036_poly2_expand(z)
        logit = np.clip(design @ self.weights + self.bias, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logit))

def _s036_model_path(model_dir):
    candidates = []
    env = os.environ.get("S036_MODEL_PATH")
    if env:
        candidates.append(Path(env))
    if model_dir is not None:
        base = Path(model_dir)
        if base.suffix.lower() == ".npz":
            candidates.append(base)
        else:
            candidates.extend((base / "model.npz", base / "s036_model.npz",
                               base / "stage1" / "model.npz",
                               base / "stage1" / "s036_model.npz",
                               base / "stage1" / "s035_target_anchored_poly2.npz"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("S036 anchored artifact not found")

def _s036_base_answers(data_dir, model_dir):
    base = globals().get("_S036_BASE_PREDICT_STAGE1")
    if base is None or base is predict_stage1:
        return {}
    try:
        frame = base(data_dir, model_dir)
        return {str(k): str(v) if str(v) in _S036_VALID_ANSWERS else "ORIGINAL"
                for k, v in zip(frame["ID"].astype(str), frame["answer"].astype(str))}
    except Exception:
        return {}

def _s036_s002_answer(path):
    score = _s036_s002_score_video(path)
    if not score.get("decoded_frames"):
        return "ORIGINAL"
    peak = score["spec_peak_db"]
    if 24.0 <= peak <= 27.0:
        corroboration = int(score["sharp_lap_log"] >= 2.11) + int(score["spec_high_ratio"] >= 0.025)
        return "RERECORDED" if corroboration >= 1 else "ORIGINAL"
    return "RERECORDED" if peak >= 25.37 else "ORIGINAL"

def _s036_s002_score_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < 32:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return {"decoded_frames": 0.0}
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    picks = sorted(set([0, len(grays) // 2, len(grays) - 1]))
    peaks, highs, sharps = [], [], []
    for index in picks:
        gray = grays[index].astype(np.float64)
        h, w = gray.shape
        side = min(512, h, w)
        top, left = (h - side) // 2, (w - side) // 2
        crop = gray[top:top + side, left:left + side]
        window = np.outer(np.hanning(side), np.hanning(side))
        power = np.abs(np.fft.fftshift(np.fft.fft2((crop - crop.mean()) * window))) ** 2
        fy = np.fft.fftshift(np.fft.fftfreq(side))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(side))[None, :]
        radius = np.hypot(fy, fx) / 0.5
        total = power[(radius > 0.02) & (radius <= 1.0)].sum() + 1e-9
        log_power = np.log10(power + 1e-9)
        whitened = np.zeros_like(log_power)
        edges = np.linspace(0.05, 0.85, 25)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (radius >= lo) & (radius < hi)
            if mask.any():
                values = log_power[mask]
                whitened[mask] = values - float(np.median(values))
        residual = whitened[(radius >= 0.2) & (radius <= 0.9)]
        peaks.append(float(np.percentile(residual, 99.99)) * 10.0)
        high = power[(radius > 0.45) & (radius <= 0.95)].sum()
        highs.append(float(high / total))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharps.append(float(math.log10(lap.var() + 1e-9)))
    return {"decoded_frames": float(len(frames)), "spec_peak_db": float(np.mean(peaks)),
            "spec_high_ratio": float(np.mean(highs)), "sharp_lap_log": float(np.mean(sharps))}

def _s036_deployment_config():
    mode = os.environ.get("S036_DEPLOYMENT_MODE", "hybrid").lower()
    if mode not in ("hybrid", "learned", "s002"):
        raise ValueError("unknown S036_DEPLOYMENT_MODE: " + str(mode))
    try:
        margin = float(os.environ.get("S036_AMBIGUITY_MARGIN", "0.15"))
    except ValueError as exc:
        raise ValueError("S036_AMBIGUITY_MARGIN must be numeric") from exc
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("S036_AMBIGUITY_MARGIN must be finite and non-negative")
    return mode, margin

def _s036_deployment_rule(learned_probability, threshold, fallback_answer,
                          mode="hybrid", margin=0.15):
    learned = "RERECORDED" if float(learned_probability) >= float(threshold) else "ORIGINAL"
    if mode == "learned":
        return learned
    if mode == "s002":
        return fallback_answer if fallback_answer in _S036_VALID_ANSWERS else "ORIGINAL"
    if mode != "hybrid":
        raise ValueError("unknown S036_DEPLOYMENT_MODE: " + str(mode))
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("S036_AMBIGUITY_MARGIN must be finite and non-negative")
    if abs(float(learned_probability) - float(threshold)) < margin:
        return fallback_answer if fallback_answer in _S036_VALID_ANSWERS else "ORIGINAL"
    return learned

def predict_stage1(data_dir, model_dir=None):
    cv2.setNumThreads(1)
    mode, margin = _s036_deployment_config()
    try:
        model = _S036AnchoredClassifier.load(_s036_model_path(model_dir))
    except (FileNotFoundError, ValueError, OSError, RuntimeError):
        model = None
    fallback = _s036_base_answers(data_dir, model_dir)
    root = Path(data_dir) / "videos"
    videos = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _S036_VIDEO_EXT),
                    key=lambda p: p.relative_to(root).as_posix())
    stems = [p.stem for p in videos]
    if len(stems) != len(set(stems)):
        raise ValueError("duplicate Stage-1 video IDs from file stems: " + str(sorted({s for s in stems if stems.count(s) > 1})))
    rows = []
    for path in videos:
        base_answer = fallback.get(path.stem)
        if base_answer not in _S036_VALID_ANSWERS:
            try:
                base_answer = _s036_s002_answer(path)
            except (OSError, ValueError, RuntimeError, cv2.error):
                base_answer = "ORIGINAL"
        answer = base_answer
        if model is not None:
            try:
                probability = float(model.probabilities(_s036_video_feature_vector(path)[None])[0])
                answer = _s036_deployment_rule(probability, model.threshold, base_answer, mode, margin)
            except (OSError, ValueError, RuntimeError, cv2.error):
                answer = base_answer
        rows.append({"ID": path.stem, "answer": answer})
    return pd.DataFrame(rows, columns=["ID", "answer"])

# CONTROLLED CANDIDATE OVERRIDE
"""S022: camera-scale-augmented acceleration model, exact S019 steering."""


_S022_MODEL_FILENAME = "stage3_accel_robust.npz"


def _s022_load_accel_weights(model_dir):
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir) / _S022_MODEL_FILENAME)
    candidates.append(Path(__file__).resolve().parent / _S022_MODEL_FILENAME)
    for path in candidates:
        if path.is_file():
            weights = np.load(path)
            required = {"w1", "b1", "w2", "b2", "w3", "b3", "mu", "sd"}
            if required.issubset(weights.files):
                return weights
            weights.close()
    return None


def predict_stage3(data_dir, model_dir=None):
    accel_weights = _s022_load_accel_weights(model_dir)
    steer_weights = _s015_load_weights(model_dir)
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames = []
    for path in videos:
        try:
            features = extract_video_features(path)
            accel_idx, steer_idx, count = _s019_classify_video(
                features, HIDDEN_DT_SECONDS, accel_weights, steer_weights
            )
            if count == 0:
                continue
        except Exception:
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel_idx = np.full(count, 2, dtype=np.int64)
            steer_idx = np.ones(count, dtype=np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(value)] for value in accel_idx],
                    "steer_label": [STEER[int(value)] for value in steer_idx],
                }
            )
        )
    if accel_weights is not None:
        accel_weights.close()
    if steer_weights is not None:
        steer_weights.close()
    if not frames:
        return pd.DataFrame(
            columns=["ID", "sample_index", "accel_label", "steer_label"]
        )
    return pd.concat(frames, ignore_index=True)[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ]


# CONTROLLED CANDIDATE OVERRIDE
"""S024: blend S019 native and S022 camera-scale-robust acceleration."""


_S024_ROBUST_ALPHA = 0.50


def _s024_classify_video(
    features, dt, original_accel_weights, robust_accel_weights, steer_weights
):
    combined, sparse = _s019_feature_matrix(features, dt)
    count = len(combined)
    if (
        count == 0
        or original_accel_weights is None
        or robust_accel_weights is None
        or steer_weights is None
    ):
        return _s019_classify_video(
            features, dt, original_accel_weights, steer_weights
        )

    original_prob = _s019_accel_probabilities(combined, original_accel_weights)
    robust_prob = _s019_accel_probabilities(combined, robust_accel_weights)
    accel_prob = (
        (1.0 - _S024_ROBUST_ALPHA) * original_prob
        + _S024_ROBUST_ALPHA * robust_prob
    )
    accel_window = int(round(_S019_ACCEL_WINDOW_SECONDS / 0.1))
    accel_prob = _s015_smooth_probabilities(accel_prob, accel_window)
    accel_logits = np.log(np.maximum(accel_prob, 1e-9))
    accel_logits += np.asarray(_S019_ACCEL_LOG_BIAS, dtype=np.float64)
    accel = accel_logits.argmax(axis=1).astype(np.int64)

    steer_prob = _s015_steer_probabilities(sparse, steer_weights)
    steer_window = int(round(_S015_STEER_WINDOW_SECONDS / 0.1))
    steer_prob = _s015_smooth_probabilities(steer_prob, steer_window)
    steer_logits = np.log(np.maximum(steer_prob, 1e-9))
    steer_logits += np.asarray(_S015_STEER_LOG_BIAS, dtype=np.float64)
    steer = steer_logits.argmax(axis=1).astype(np.int64)
    return accel, steer, count


def predict_stage3(data_dir, model_dir=None):
    original_accel_weights = _s019_load_accel_weights(model_dir)
    robust_accel_weights = _s022_load_accel_weights(model_dir)
    steer_weights = _s015_load_weights(model_dir)
    cv2.setNumThreads(1)
    videos = video_paths(Path(data_dir) / "videos")
    frames = []
    for path in videos:
        try:
            features = extract_video_features(path)
            accel_idx, steer_idx, count = _s024_classify_video(
                features,
                HIDDEN_DT_SECONDS,
                original_accel_weights,
                robust_accel_weights,
                steer_weights,
            )
            if count == 0:
                continue
        except Exception:
            count = _fallback_frame_count(path)
            if count <= 0:
                continue
            accel_idx = np.full(count, 2, dtype=np.int64)
            steer_idx = np.ones(count, dtype=np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ID": path.stem,
                    "sample_index": np.arange(count, dtype=np.int64),
                    "accel_label": [ACCEL[int(value)] for value in accel_idx],
                    "steer_label": [STEER[int(value)] for value in steer_idx],
                }
            )
        )
    for weights in (original_accel_weights, robust_accel_weights, steer_weights):
        if weights is not None:
            weights.close()
    if not frames:
        return pd.DataFrame(
            columns=["ID", "sample_index", "accel_label", "steer_label"]
        )
    return pd.concat(frames, ignore_index=True)[
        ["ID", "sample_index", "accel_label", "steer_label"]
    ]


# S048 CONTROLLED TEMPORAL CONTEXT OVERRIDE
"""S048 ordered within-video context; appended to the exact S038 source.

The original final predict_stage3 still owns decoding, features, steering,
probability blending, label decoding, and file-local output assembly.
"""

_S048_CONTEXT_OFFSETS = (-20, -10, -5, -2, 0, 2, 5, 10, 20)
_s048_base_accel_probabilities = _s019_accel_probabilities


def _s048_ordered_context(values):
    if values.ndim != 2 or values.shape[1] != 40:
        raise ValueError('S048 expects a single-video N x 40 feature matrix')
    if len(values) == 0:
        return np.empty((0, 360), dtype=values.dtype)
    indices = np.arange(len(values))[:, None] + np.asarray(_S048_CONTEXT_OFFSETS)[None, :]
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices].reshape(len(values), 360)


def _s019_accel_probabilities(features, weights):
    if weights['mu'].shape != (360,):
        raise ValueError('S048 temporal weight contract requires 360 features')
    contextual = _s048_ordered_context(features)
    return _s048_base_accel_probabilities(contextual, weights)


# S051 ROBUST-ALPHA 0.75 OVERRIDE
"""S051 robust-alpha 0.75; appended to the exact S048 source.

Captures the final predict_stage3 and runs it with _S024_ROBUST_ALPHA = 0.75,
restoring the previous value afterwards (S045 pattern). Steering, decoder,
features and weights are untouched.
"""
_S051_BASE_PREDICT_STAGE3 = predict_stage3
_S051_ROBUST_ALPHA = 1.00


def predict_stage3(data_dir, model_dir=None):
    """Run the S048 decoder with robust alpha 0.75."""
    previous = globals().get("_S024_ROBUST_ALPHA", 0.50)
    globals()["_S024_ROBUST_ALPHA"] = _S051_ROBUST_ALPHA
    try:
        return _S051_BASE_PREDICT_STAGE3(data_dir, model_dir)
    finally:
        globals()["_S024_ROBUST_ALPHA"] = previous


# S071 CANONICAL720 STAGE1 OVERRIDE
"""S071 canonical720 shared-head override; append after unchanged S053 source.

Only Stage1 is replaced. Stage2/3 names and implementations remain untouched.
This is not a ready-to-submit artifact or permission to promote the model.
"""
import importlib as _s071_importlib
import sys as _s071_sys
from pathlib import Path as _s071_Path

_S071_VIDEO_EXTENSIONS = frozenset(('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.3gp', '.3gpp', '.wmv'))
_S071_PREDICTORS = {}


def _s071_model_root(model_dir):
    root = _s071_Path(model_dir).resolve() if model_dir is not None else _s071_Path(__file__).resolve().parent / 'model'
    if (root / 'stage1_dino').is_dir():
        return root
    if (root / 'model' / 'stage1_dino').is_dir():
        return root / 'model'
    if root.name.lower() == 'stage1' and root.is_dir() and (root.parent / 'stage1_dino').is_dir():
        return root.parent
    raise FileNotFoundError('S071 expects model/stage1, a model root, or a candidate root containing stage1_dino')


def _s071_load_predictor(model_root):
    identity = str(model_root.resolve())
    if identity in _S071_PREDICTORS:
        return _S071_PREDICTORS[identity]
    package_parent = model_root / 'stage1_dino_code'
    expected_runtime = (package_parent / 's071_runtime' / 'runtime.py').resolve()
    if not expected_runtime.is_file():
        raise FileNotFoundError(expected_runtime)
    for name, module in tuple(_s071_sys.modules.items()):
        if name == 's071_runtime' or name.startswith('s071_runtime.'):
            location = getattr(module, '__file__', None)
            if location is None or not _s071_Path(location).resolve().is_relative_to(expected_runtime.parent):
                raise RuntimeError('A different S071/S072 runtime root is already imported; use a fresh process')
    previous_path = list(_s071_sys.path)
    previous_bytecode = _s071_sys.dont_write_bytecode
    try:
        _s071_sys.path.insert(0, str(package_parent))
        _s071_sys.dont_write_bytecode = True
        runtime = _s071_importlib.import_module('s071_runtime.runtime')
    finally:
        _s071_sys.path[:] = previous_path
        _s071_sys.dont_write_bytecode = previous_bytecode
    if _s071_Path(runtime.__file__).resolve() != expected_runtime:
        raise RuntimeError('A different S071/S072 runtime root is already imported; use a fresh process')
    predictor = runtime.Predictor(model_root / 'stage1_dino' / 'backbone',
                                  model_root / 'stage1_dino' / 'head.bin', model_root / 'stage1_dino' / 'runtime_config.json', device='cuda:0')
    _S071_PREDICTORS[identity] = predictor
    return predictor


def predict_stage1(data_dir, model_dir=None):
    import pandas as _s071_pd
    videos_root = _s071_Path(data_dir) / 'videos'
    if not videos_root.is_dir():
        raise FileNotFoundError('S071 expects input videos/ directory')
    videos = sorted((p for p in videos_root.rglob('*') if p.is_file() and p.suffix.lower() in _S071_VIDEO_EXTENSIONS),
                    key=lambda p: p.relative_to(videos_root).as_posix())
    identifiers = [p.stem for p in videos]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate Stage1 file-stem IDs')
    if not videos:
        return _s071_pd.DataFrame([], columns=['ID', 'answer'])
    predictor = _s071_load_predictor(_s071_model_root(model_dir))
    rows = []
    for path in videos:
        result = predictor.predict_video(path)
        if result['prediction'] not in (0, 1):
            raise ValueError('Invalid S071 binary decision')
        rows.append({'ID': path.stem, 'answer': 'RERECORDED' if result['prediction'] else 'ORIGINAL'})
    return _s071_pd.DataFrame(rows, columns=['ID', 'answer'])


# S108 acceleration-only dispatch appended after the exact scored S105 bytes.
# Works when inference.py itself is absent from sys.modules.
import importlib.util as _s108_importlib
import sys as _s108_sys
import threading as _s108_threading
from pathlib import Path as _S108Path

_S108_LEGACY_PREDICT_STAGE3 = globals()['predict_stage3']
_S108_LAST_DIAGNOSTICS = {}
_S108_LOCK = _s108_threading.RLock()


def _s108_runtime():
    folder = _S108Path(__file__).resolve().parent / 'model/stage3/s108_runtime'
    # Path-bound package namespace prevents one loaded artifact from resolving
    # imports against another artifact's vendor code in the same process.
    import hashlib as _s108_hashlib
    name = '_s108_runtime_' + _s108_hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    if name not in _s108_sys.modules:
        spec = _s108_importlib.spec_from_file_location(
            name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        package = _s108_importlib.module_from_spec(spec)
        _s108_sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            _s108_sys.modules.pop(name, None)
            raise
    from importlib import import_module as _s108_import_module
    return _s108_import_module(name + '.runtime')


def predict_stage3(data_dir, model_dir=None):
    folder = (_S108Path(model_dir) if model_dir is not None
              else _S108Path(__file__).resolve().parent / 'model/stage3')
    with _S108_LOCK:
        return _s108_runtime().predict(globals(), _S108_LEGACY_PREDICT_STAGE3,
                                       data_dir, folder)


# S109 deployment adapter; original S108/S105 source above remains intact.
def _s109_adapter():
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parent / 'model/stage3/s109'
    decision_bytes = (root / 'selection.json').read_bytes()
    if hashlib.sha256(decision_bytes).hexdigest() != '8262b94fbb89cc2c4abfda1b216fb695739d84cdf40c9a935cc863489eadc011':
        raise RuntimeError('S109 deployment decision identity mismatch')
    decision = json.loads(decision_bytes)
    source = root / 'adapter.py'
    raw = source.read_bytes()
    if (decision['recipe'] != 'base75_vjepa25_incumbent_decoder'
            or decision['video_weight'] != 0.25
            or decision['adapter'] != {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}):
        raise RuntimeError('S109 deployment adapter/recipe mismatch')
    name = '_s109_' + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def predict_stage3(data_dir, model_dir=None):
    from pathlib import Path
    folder = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent / 'model/stage3'
    with globals()['_S108_LOCK']:
        return _s109_adapter().predict(globals(), data_dir, folder)


# S118: rule-driven Stage1/Stage2/Stage3 leaf overrides; preserve all S109 source.
# A stage whose rule file is absent returns the S109 result object untouched.
_S118_BASE_PREDICT_STAGE1 = predict_stage1
_S118_BASE_PREDICT_STAGE2 = predict_stage2
_S118_BASE_PREDICT_STAGE3 = predict_stage3
_S118_STAGE1_DIAGNOSTICS = {}
_S118_LAST_DIAGNOSTICS = {}
_S118_STAGE3_DIAGNOSTICS = {}


def _s118_adapter():
    import hashlib
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parent / 'model/stage2/s118'
    name = '_s118_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'adapter.py', submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def predict_stage1(data_dir, model_dir=None):
    from pathlib import Path
    base = _S118_BASE_PREDICT_STAGE1(data_dir, model_dir)
    rule_path = Path(__file__).resolve().parent / 'model/stage1/s118/rule1.json'
    if not rule_path.is_file():
        return base
    diag = {'rule_error': None, 'videos': {}}
    globals()['_S118_STAGE1_DIAGNOSTICS'] = diag
    try:
        adapter = _s118_adapter()
        rule = adapter.load_rule1(rule_path)
        return adapter.apply_stage1(base, data_dir, rule, diag['videos'])
    except Exception as exc:
        diag['rule_error'] = '%s: %s' % (type(exc).__name__, exc)
        return base


def predict_stage2(data_dir, model_dir):
    from pathlib import Path
    if not (Path(__file__).resolve().parent / 'model/stage2/s118/rule.json').is_file():
        return _S118_BASE_PREDICT_STAGE2(data_dir, model_dir)
    try:
        adapter = _s118_adapter()
    except Exception:
        return _S118_BASE_PREDICT_STAGE2(data_dir, model_dir)
    return adapter.predict(globals(), data_dir, model_dir)


def predict_stage3(data_dir, model_dir=None):
    from pathlib import Path
    rule = Path(__file__).resolve().parent / 'model/stage3/s118/rule3.json'
    if not rule.is_file():
        return _S118_BASE_PREDICT_STAGE3(data_dir, model_dir)
    try:
        adapter = _s118_adapter()
        selected = adapter.load_rule3(rule)
    except Exception:
        return _S118_BASE_PREDICT_STAGE3(data_dir, model_dir)
    if 'steer_regression_package' in selected:
        return adapter.predict_stage3_regression(globals(), data_dir, model_dir, selected)
    if 'steer_video_mix' in selected:
        return adapter.predict_stage3_mix(globals(), data_dir, model_dir, selected)
    if 'accel_log_bias' in selected:
        base = adapter.predict_stage3_bias(globals(), data_dir, model_dir, selected)
        if 'steer_label_const' in selected:
            try:
                return adapter.apply_stage3(base, rule)
            except Exception:
                return base
        return base
    base = _S118_BASE_PREDICT_STAGE3(data_dir, model_dir)
    try:
        return adapter.apply_stage3(base, rule)
    except Exception:
        return base


# S163 pixel-only positive recapture gate.
"""Append to S156 inference.py; changes Stage1 answers only."""
_S163_BASE_PREDICT_STAGE1 = predict_stage1
_S163_DIAGNOSTICS = {}


def predict_stage1(data_dir, model_dir=None):
    import importlib.util
    import json
    from pathlib import Path
    import time

    base = _S163_BASE_PREDICT_STAGE1(data_dir, model_dir)
    diag = {'videos': {}, 'model_error': None}
    globals()['_S163_DIAGNOSTICS'] = diag
    root = Path(__file__).resolve().parent / 'model/stage1/s163'
    try:
        spec = importlib.util.spec_from_file_location('_s163_cue', root / 'cue.py')
        cue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cue)
        model = json.loads((root / 'model.json').read_text())
        if model['version'] != cue.VERSION or model['direction'] != 'ORIGINAL_to_RERECORDED':
            raise ValueError('Unexpected S163 model contract')
    except Exception as exc:
        diag['model_error'] = type(exc).__name__ + ': ' + str(exc)
        return base
    paths = {p.stem: p for p in (Path(data_dir) / 'videos').rglob('*')
             if p.is_file() and p.suffix.lower() in _S071_VIDEO_EXTENSIONS}
    result = base.copy()
    for index, row in base.iterrows():
        if row['answer'] != 'ORIGINAL':
            continue
        start = time.perf_counter()
        record = {'base_answer': 'ORIGINAL', 'flipped': False, 'error': None}
        try:
            description = cue.describe(paths[row['ID']])
            score = cue.predict(description['features'], model)
            record.update(probability=score, threshold=model['threshold'], frames=description['frames'])
            if score >= model['threshold']:
                result.at[index, 'answer'] = 'RERECORDED'
                record['flipped'] = True
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        record['seconds'] = time.perf_counter() - start
        diag['videos'][row['ID']] = record
    return result

# S164: execution-only acceleration. Append after all stage/field adapters.
_S164_BASE_STAGE2 = predict_stage2
_S164_BASE_STAGE3 = predict_stage3


def _s164_runtime():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parent / 'model/runtime_s164/runtime.py'
    spec = importlib.util.spec_from_file_location('_s164_runtime', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_stage2(data_dir, model_dir):
    return _s164_runtime().predict(globals(), _S164_BASE_STAGE2, data_dir, model_dir, 2)


def predict_stage3(data_dir, model_dir=None):
    return _s164_runtime().predict(globals(), _S164_BASE_STAGE3, data_dir, model_dir, 3)

# S172 clocks the visual extension from module initialization.
import time as _s172_time
_S172_PROCESS_START = _s172_time.perf_counter()

# S172 applies after complete S171 entry reconstruction; only collision and its clamp change.
_S172_BASE_STAGE2 = predict_stage2

def _s172_runtime():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parent / 'model/stage2/s172/runtime.py'
    spec = importlib.util.spec_from_file_location('_s172_runtime', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def predict_stage2(data_dir, model_dir):
    from pathlib import Path
    base = _S172_BASE_STAGE2(data_dir, model_dir)
    try:
        runtime = _s172_runtime()
        return runtime.apply(globals(), base, data_dir,
                             Path(__file__).resolve().parent / 'model/stage2/s172')
    except Exception as exc:
        globals()['_S172_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}
        return base


# S175: S172 collision, then S174 entry on that final anchor, then clamp.
_S175_BASE_STAGE2 = predict_stage2

def predict_stage2(data_dir, model_dir):
    import importlib.util
    from pathlib import Path
    base = _S175_BASE_STAGE2(data_dir, model_dir)
    try:
        source = Path(__file__).resolve().parent / 'model/stage2/s175_entry.py'
        spec = importlib.util.spec_from_file_location('_s175_entry', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.apply(globals(), base, data_dir, model_dir)
    except Exception as exc:
        globals()['_S175_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}
        return base


# S177: S175 rows, then only S176 abstention buckets on final S172 collision.
_S177_BASE_STAGE2 = predict_stage2

def predict_stage2(data_dir, model_dir):
    import importlib.util
    from pathlib import Path
    base = _S177_BASE_STAGE2(data_dir, model_dir)
    try:
        source = Path(__file__).resolve().parent / 'model/stage2/s177_entry.py'
        spec = importlib.util.spec_from_file_location('_s177_entry', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.apply(globals(), base, data_dir, model_dir)
    except Exception as exc:
        globals()['_S177_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}
        return base


# S182: ordered chained rules on exact S177 output and final S172 collision.
_S182_BASE_STAGE2 = predict_stage2

def predict_stage2(data_dir, model_dir):
    import importlib.util
    from pathlib import Path
    base = _S182_BASE_STAGE2(data_dir, model_dir)
    try:
        source = Path(__file__).resolve().parent / 'model/stage2/s182_chain.py'
        spec = importlib.util.spec_from_file_location('_s182_chain', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.apply(globals(), base, data_dir, model_dir, full=True)
    except Exception as exc:
        globals()['_S182_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}
        return base
