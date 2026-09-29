"""Local implementation of the official DACON 236753 metric.

Stage 2 converts submitted frames with per-video frame-time metadata before
applying Accuracy@0.3 seconds. The local evaluator therefore requires that
mapping explicitly and never assumes a constant FPS.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

STAGE1_LABELS = ("ORIGINAL", "RERECORDED")
ENTRY_SIDE_LABELS = ("LEFT", "RIGHT")
EVASION_LABELS = (0, 1)
ACCEL_LABELS = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_LABELS = ("LEFT", "STRAIGHT", "RIGHT")
MAX_REPORTED_EXAMPLES = 5


class EvaluationError(ValueError):
    """Raised when truth or prediction tables violate the local contract."""


@dataclass(frozen=True)
class Stage2Scores:
    collision_accuracy_03s: float
    entry_accuracy_03s: float
    entry_side_macro_f1: float
    evasion_space_macro_f1: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage3Scores:
    accel_macro_f1: float
    steer_macro_f1: float
    score: float
    steer_evaluated_rows: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CompetitionScores:
    stage1: float
    stage2: float
    stage3: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EvaluationError(f"{name} is missing columns: {missing}")


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = (
            frame.loc[duplicated, list(keys)]
            .head(MAX_REPORTED_EXAMPLES)
            .to_dict("records")
        )
        raise EvaluationError(f"{name} has duplicate keys {list(keys)}: {examples}")


def _key_index(frame: pd.DataFrame, keys: Sequence[str], name: str) -> pd.MultiIndex:
    """Build a comparable key index, refusing missing key components."""

    key_frame = frame[list(keys)]
    if bool(key_frame.isna().any().any()):
        raise EvaluationError(f"{name} has missing values in key columns {list(keys)}")
    return pd.MultiIndex.from_frame(key_frame)


def _describe_keys(index: pd.Index) -> list[object]:
    examples = list(index[:MAX_REPORTED_EXAMPLES])
    return [value[0] if len(value) == 1 else value for value in examples]


def _require_exact_coverage(
    truth: pd.DataFrame, prediction: pd.DataFrame, keys: Sequence[str]
) -> None:
    """Require one prediction row per truth row and no unexpected extras."""

    truth_keys = _key_index(truth, keys, "truth")
    prediction_keys = _key_index(prediction, keys, "prediction")
    missing = truth_keys.difference(prediction_keys, sort=False)
    extra = prediction_keys.difference(truth_keys, sort=False)
    problems: list[str] = []
    if len(missing):
        problems.append(
            f"{len(missing)} truth rows have no prediction "
            f"(examples: {_describe_keys(missing)})"
        )
    if len(extra):
        problems.append(
            f"{len(extra)} prediction rows match no truth row "
            f"(examples: {_describe_keys(extra)})"
        )
    if problems:
        raise EvaluationError(
            f"prediction coverage does not match truth on {list(keys)}: "
            + "; ".join(problems)
        )


def _align(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    keys: Sequence[str],
    truth_columns: Sequence[str],
    prediction_columns: Sequence[str],
) -> pd.DataFrame:
    _require_columns(truth, [*keys, *truth_columns], "truth")
    _require_columns(prediction, [*keys, *prediction_columns], "prediction")
    _require_unique(truth, keys, "truth")
    _require_unique(prediction, keys, "prediction")
    _require_exact_coverage(truth, prediction, keys)
    return truth[[*keys, *truth_columns]].merge(
        prediction[[*keys, *prediction_columns]],
        on=list(keys),
        how="left",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )


def _coerce_evasion(value: object) -> int | None:
    """Map one legitimate evasion encoding to 0/1, or None when invalid."""

    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        # Only the canonical CSV encodings are legitimate; "1.0" or "1e0" are not.
        stripped = value.strip()
        return int(stripped) if stripped in ("0", "1") else None
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if not isfinite(numeric) or not numeric.is_integer():
            return None
        label = int(numeric)
        return label if label in EVASION_LABELS else None
    return None


def normalize_evasion_space(values: Iterable[object], name: str) -> list[int]:
    """Normalize evasion encodings to canonical 0/1, rejecting invalid values."""

    normalized: list[int] = []
    invalid: list[object] = []
    for value in values:
        label = _coerce_evasion(value)
        if label is None:
            if len(invalid) < MAX_REPORTED_EXAMPLES:
                invalid.append(value)
        else:
            normalized.append(label)
    if invalid:
        raise EvaluationError(
            f"{name} has values outside the evasion label set {list(EVASION_LABELS)}: "
            f"{invalid}"
        )
    return normalized


def macro_f1(
    y_true: Iterable[Hashable],
    y_pred: Iterable[Hashable],
    labels: Sequence[Hashable],
) -> float:
    """Macro-F1 over the fixed official label universe."""

    truth = np.asarray(list(y_true), dtype=object)
    prediction = np.asarray(list(y_pred), dtype=object)
    if truth.shape != prediction.shape:
        raise EvaluationError(
            f"truth/prediction length mismatch: {truth.size} != {prediction.size}"
        )
    if truth.ndim != 1:
        raise EvaluationError("macro_f1 expects one-dimensional inputs")

    scores: list[float] = []
    for label in labels:
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def score_stage1(truth: pd.DataFrame, prediction: pd.DataFrame) -> float:
    aligned = _align(truth, prediction, ["ID"], ["answer"], ["answer"])
    return macro_f1(aligned["answer_true"], aligned["answer_pred"], STAGE1_LABELS)


def _frame_time_lookup(frame_times: pd.DataFrame) -> dict[tuple[object, int], float]:
    _require_columns(frame_times, ["ID", "frame", "time_seconds"], "frame_times")
    _require_unique(frame_times, ["ID", "frame"], "frame_times")
    lookup: dict[tuple[object, int], float] = {}
    for row in frame_times[["ID", "frame", "time_seconds"]].itertuples(index=False):
        try:
            frame_number = float(row.frame)
            time_seconds = float(row.time_seconds)
        except (TypeError, ValueError):
            continue
        if (
            isfinite(frame_number)
            and frame_number >= 0
            and frame_number.is_integer()
            and isfinite(time_seconds)
        ):
            lookup[(row.ID, int(frame_number))] = time_seconds
    return lookup


def _map_frames_to_seconds(
    ids: pd.Series, frames: pd.Series, lookup: dict[tuple[object, int], float]
) -> np.ndarray:
    mapped = np.full(len(ids), np.nan, dtype=float)
    for index, (sample_id, raw_frame) in enumerate(zip(ids, frames)):
        try:
            frame = float(raw_frame)
        except (TypeError, ValueError):
            continue
        if not isfinite(frame) or frame < 0 or not frame.is_integer():
            continue
        mapped[index] = lookup.get((sample_id, int(frame)), np.nan)
    return mapped


def _accuracy_at_seconds(
    truth_seconds: pd.Series, prediction_seconds: np.ndarray, tolerance: float
) -> float:
    truth = pd.to_numeric(truth_seconds, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction_seconds)
    correct = valid & (np.abs(truth - prediction_seconds) <= tolerance + 1e-12)
    return float(np.mean(correct))


def score_stage2(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    frame_times: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.3,
) -> Stage2Scores:
    """Score Stage 2 with official weights and per-video frame-time maps."""

    if tolerance_seconds < 0:
        raise EvaluationError("tolerance_seconds must be non-negative")
    aligned = _align(
        truth,
        prediction,
        ["ID"],
        ["collision_time_seconds", "entry_time_seconds", "evasion_space", "entry_side"],
        ["collision_frame", "entry_frame", "evasion_space", "entry_side"],
    )
    lookup = _frame_time_lookup(frame_times)
    collision_seconds = _map_frames_to_seconds(
        aligned["ID"], aligned["collision_frame"], lookup
    )
    entry_seconds = _map_frames_to_seconds(
        aligned["ID"], aligned["entry_frame"], lookup
    )
    collision = _accuracy_at_seconds(
        aligned["collision_time_seconds"], collision_seconds, tolerance_seconds
    )
    entry = _accuracy_at_seconds(
        aligned["entry_time_seconds"], entry_seconds, tolerance_seconds
    )
    direction = macro_f1(
        aligned["entry_side_true"], aligned["entry_side_pred"], ENTRY_SIDE_LABELS
    )
    evasion_true = normalize_evasion_space(
        aligned["evasion_space_true"], "truth evasion_space"
    )
    evasion_pred = normalize_evasion_space(
        aligned["evasion_space_pred"], "prediction evasion_space"
    )
    evasion = macro_f1(evasion_true, evasion_pred, EVASION_LABELS)
    score = 0.35 * collision + 0.35 * entry + 0.15 * direction + 0.15 * evasion
    return Stage2Scores(collision, entry, direction, evasion, score)


def score_stage3(truth: pd.DataFrame, prediction: pd.DataFrame) -> Stage3Scores:
    aligned = _align(
        truth,
        prediction,
        ["ID", "sample_index"],
        ["accel_label", "steer_label"],
        ["accel_label", "steer_label"],
    )
    accel = macro_f1(
        aligned["accel_label_true"], aligned["accel_label_pred"], ACCEL_LABELS
    )
    moving = aligned["accel_label_true"] != "STOPPED"
    steer = macro_f1(
        aligned.loc[moving, "steer_label_true"],
        aligned.loc[moving, "steer_label_pred"],
        STEER_LABELS,
    )
    return Stage3Scores(accel, steer, 0.7 * accel + 0.3 * steer, int(moving.sum()))


def competition_score(stage1: float, stage2: float, stage3: float) -> CompetitionScores:
    values = (float(stage1), float(stage2), float(stage3))
    if not all(isfinite(value) for value in values):
        raise EvaluationError("stage scores must be finite")
    total = 0.2 * values[0] + 0.4 * values[1] + 0.4 * values[2]
    return CompetitionScores(*values, total)
