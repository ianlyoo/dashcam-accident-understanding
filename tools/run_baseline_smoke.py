"""Run the generated baseline inference end-to-end on public fixtures."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch


def prepare_inputs(root: Path) -> Path:
    data = root / "data"
    smoke = root / "sample_evaluation_data"
    if smoke.exists():
        shutil.rmtree(smoke)
    (smoke / "stage1" / "videos").mkdir(parents=True)
    (smoke / "stage2" / "images").mkdir(parents=True)
    (smoke / "stage3" / "videos").mkdir(parents=True)

    for label, folder in (("O", "original"), ("R", "rerecorded")):
        for index, path in enumerate(sorted((data / "stage1" / folder).glob("*")), 1):
            target = (
                smoke
                / "stage1"
                / "videos"
                / f"SAMPLE_S1_{label}_{index:03d}{path.suffix.lower()}"
            )
            shutil.copy2(path, target)

    for index, video_path in enumerate(
        sorted((data / "stage2" / "videos").glob("*")), 1
    ):
        frame_dir = smoke / "stage2" / "images" / f"SAMPLE_S2_{index:03d}"
        frame_dir.mkdir()
        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(frame_dir / f"frame_{frame_index:06d}.jpg"), image):
                raise RuntimeError(
                    f"failed to write frame {frame_index} from {video_path}"
                )
            frame_index += 1
        capture.release()

    for path in sorted((data / "stage3" / "videos").glob("*")):
        shutil.copy2(path, smoke / "stage3" / "videos" / path.name)
    return smoke


def load_inference(path: Path):
    # Import by the real filename so Windows DataLoader workers can import the
    # Dataset classes again in spawned processes.
    sys.path.insert(0, str(path.parent))
    sys.modules.pop(path.stem, None)
    return importlib.import_module(path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("Baseline"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("stage1", "stage2", "stage3"),
        default=["stage1", "stage2", "stage3"],
    )
    args = parser.parse_args()
    root = args.baseline_dir.resolve()
    smoke = prepare_inputs(root)
    inference = load_inference(root / "inference.py")
    output = root / "output"
    output.mkdir(exist_ok=True)

    for stage in args.stages:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        frame = getattr(inference, f"predict_{stage}")(
            smoke / stage, root / "model" / stage
        )
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        frame.to_csv(output / f"{stage}_submission.csv", index=False, encoding="utf-8")
        print(
            f"{stage}: rows={len(frame)} elapsed_seconds={elapsed:.3f} "
            f"peak_cuda_gib={peak:.3f} columns={list(frame.columns)}"
        )


if __name__ == "__main__":
    main()
