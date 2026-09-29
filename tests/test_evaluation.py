import math

import pandas as pd
import pytest

from videohackathon.evaluation import (
    EvaluationError,
    competition_score,
    normalize_evasion_space,
    score_stage1,
    score_stage2,
    score_stage3,
)


def _stage2_frames():
    truth = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "collision_time_seconds": [0.7, 2.3],
            "entry_time_seconds": [0.2, 1.5],
            "evasion_space": [0, 1],
            "entry_side": ["LEFT", "RIGHT"],
        }
    )
    prediction = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "collision_frame": [7, 23],
            "entry_frame": [2, 15],
            "evasion_space": [0, 1],
            "entry_side": ["LEFT", "RIGHT"],
        }
    )
    frame_times = pd.DataFrame(
        {
            "ID": ["a", "a", "b", "b"],
            "frame": [7, 2, 23, 15],
            "time_seconds": [0.7, 0.2, 2.31, 1.5],
        }
    )
    return truth, prediction, frame_times


def test_stage1_uses_fixed_two_class_macro_f1():
    truth = pd.DataFrame({"ID": ["a", "b"], "answer": ["ORIGINAL", "RERECORDED"]})
    prediction = pd.DataFrame({"ID": ["a", "b"], "answer": ["ORIGINAL", "ORIGINAL"]})
    assert math.isclose(score_stage1(truth, prediction), 1 / 3)


def test_stage2_matches_official_weights_and_inclusive_point_three_seconds():
    truth = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "collision_time_seconds": [1.0, 2.0],
            "entry_time_seconds": [0.5, 1.5],
            "evasion_space": [0, 1],
            "entry_side": ["LEFT", "RIGHT"],
        }
    )
    prediction = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "collision_frame": [7, 23],
            "entry_frame": [2, 15],
            "evasion_space": [0, 1],
            "entry_side": ["LEFT", "LEFT"],
        }
    )
    frame_times = pd.DataFrame(
        {
            "ID": ["a", "a", "b", "b"],
            "frame": [7, 2, 23, 15],
            "time_seconds": [0.7, 0.2, 2.31, 1.5],
        }
    )
    scores = score_stage2(truth, prediction, frame_times)
    assert scores.collision_accuracy_03s == 0.5
    assert scores.entry_accuracy_03s == 1.0
    assert math.isclose(scores.entry_side_macro_f1, 1 / 3)
    assert scores.evasion_space_macro_f1 == 1.0
    assert math.isclose(scores.score, 0.725)


def test_stage3_excludes_true_stopped_rows_from_steering():
    truth = pd.DataFrame(
        {
            "ID": ["v"] * 4,
            "sample_index": [0, 1, 2, 3],
            "accel_label": [
                "ACCELERATING",
                "DECELERATING",
                "CONSTANT",
                "STOPPED",
            ],
            "steer_label": ["LEFT", "STRAIGHT", "RIGHT", "LEFT"],
        }
    )
    prediction = truth.copy()
    prediction.loc[3, "steer_label"] = "RIGHT"
    scores = score_stage3(truth, prediction)
    assert scores.accel_macro_f1 == 1.0
    assert scores.steer_macro_f1 == 1.0
    assert scores.steer_evaluated_rows == 3
    assert scores.score == 1.0


def test_competition_score_uses_official_stage_weights():
    scores = competition_score(0.5, 0.75, 1.0)
    assert math.isclose(scores.total, 0.8)


def test_stage1_rejects_missing_predictions():
    truth = pd.DataFrame({"ID": ["a", "b"], "answer": ["ORIGINAL", "RERECORDED"]})
    prediction = pd.DataFrame({"ID": ["a"], "answer": ["ORIGINAL"]})
    with pytest.raises(EvaluationError, match="no prediction"):
        score_stage1(truth, prediction)


def test_stage1_rejects_extra_predictions():
    truth = pd.DataFrame({"ID": ["a"], "answer": ["ORIGINAL"]})
    prediction = pd.DataFrame(
        {"ID": ["a", "ghost"], "answer": ["ORIGINAL", "RERECORDED"]}
    )
    with pytest.raises(EvaluationError, match="match no truth row"):
        score_stage1(truth, prediction)


def test_stage1_rejects_duplicate_prediction_ids():
    truth = pd.DataFrame({"ID": ["a"], "answer": ["ORIGINAL"]})
    prediction = pd.DataFrame({"ID": ["a", "a"], "answer": ["ORIGINAL", "RERECORDED"]})
    with pytest.raises(EvaluationError, match="duplicate keys"):
        score_stage1(truth, prediction)


def test_stage1_rejects_missing_key_values():
    truth = pd.DataFrame({"ID": ["a"], "answer": ["ORIGINAL"]})
    prediction = pd.DataFrame({"ID": [None], "answer": ["ORIGINAL"]})
    with pytest.raises(EvaluationError, match="missing values in key columns"):
        score_stage1(truth, prediction)


def test_stage2_normalizes_csv_string_evasion_values():
    truth, prediction, frame_times = _stage2_frames()
    prediction["evasion_space"] = ["0", "1"]
    scores = score_stage2(truth, prediction, frame_times)
    assert scores.evasion_space_macro_f1 == 1.0


def test_stage2_normalizes_boolean_and_float_evasion_values():
    truth, prediction, frame_times = _stage2_frames()
    truth["evasion_space"] = [False, True]
    prediction["evasion_space"] = [0.0, " 1 "]
    scores = score_stage2(truth, prediction, frame_times)
    assert scores.evasion_space_macro_f1 == 1.0


def test_stage2_rejects_invalid_prediction_evasion_values():
    truth, prediction, frame_times = _stage2_frames()
    prediction["evasion_space"] = ["yes", "1"]
    with pytest.raises(EvaluationError, match="prediction evasion_space"):
        score_stage2(truth, prediction, frame_times)


def test_stage2_rejects_non_canonical_numeric_evasion_strings():
    truth, prediction, frame_times = _stage2_frames()
    prediction["evasion_space"] = ["1.0", "1e0"]
    with pytest.raises(EvaluationError, match="prediction evasion_space"):
        score_stage2(truth, prediction, frame_times)


def test_stage2_rejects_invalid_truth_evasion_values():
    truth, prediction, frame_times = _stage2_frames()
    truth["evasion_space"] = [-1, 1]
    with pytest.raises(EvaluationError, match="truth evasion_space"):
        score_stage2(truth, prediction, frame_times)


def test_stage2_rejects_extra_prediction_ids():
    truth, prediction, frame_times = _stage2_frames()
    extra = prediction.iloc[[0]].copy()
    extra["ID"] = "ghost"
    with pytest.raises(EvaluationError, match="match no truth row"):
        score_stage2(truth, pd.concat([prediction, extra]), frame_times)


def test_stage3_rejects_partial_sample_coverage():
    truth = pd.DataFrame(
        {
            "ID": ["v", "v"],
            "sample_index": [0, 1],
            "accel_label": ["ACCELERATING", "CONSTANT"],
            "steer_label": ["LEFT", "STRAIGHT"],
        }
    )
    with pytest.raises(EvaluationError, match="no prediction"):
        score_stage3(truth, truth.iloc[[0]])


def test_stage3_rejects_duplicate_sample_rows():
    truth = pd.DataFrame(
        {
            "ID": ["v"],
            "sample_index": [0],
            "accel_label": ["ACCELERATING"],
            "steer_label": ["LEFT"],
        }
    )
    prediction = pd.concat([truth, truth])
    with pytest.raises(EvaluationError, match="duplicate keys"):
        score_stage3(truth, prediction)


def test_normalize_evasion_space_reports_invalid_values():
    assert normalize_evasion_space(["0", 1, True], "truth evasion_space") == [0, 1, 1]
    with pytest.raises(EvaluationError, match="evasion label set"):
        normalize_evasion_space([0, 2], "truth evasion_space")


def test_normalize_evasion_space_rejects_non_canonical_strings():
    for value in ("1.0", "1e0", "", "true", "01"):
        with pytest.raises(EvaluationError, match="evasion label set"):
            normalize_evasion_space([value], "prediction evasion_space")
