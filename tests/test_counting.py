from pathlib import Path

import pytest

from evaluator_service.counting import (
    CountMeasurement,
    load_count_file,
    validate_count_submission,
)
from evaluator_service.errors import FormatValidationError
from evaluator_service.metrics import evaluate_counting
from evaluator_service.scoring import _score


def test_evaluate_counting_computes_mean_squared_error():
    ground_truth = {
        "one.png": CountMeasurement("one.png", 1.0),
        "two.png": CountMeasurement("two.png", 3.0),
    }
    predictions = {
        "one.png": CountMeasurement("one.png", 2.0),
        "two.png": CountMeasurement("two.png", 1.0),
    }

    result = evaluate_counting(ground_truth, predictions)

    assert result.score == pytest.approx(2.5)
    assert result.metrics == {"mse": 2.5, "score": 2.5}


def test_q4_repository_ground_truth_files_are_valid():
    project_root = Path(__file__).resolve().parents[1]

    validation = load_count_file(project_root / "q4" / "val.txt")
    test = load_count_file(project_root / "q4_test" / "test.txt")

    assert len(validation) == 460
    assert len(test) == 963


def test_q4_ground_truth_as_prediction_scores_zero():
    project_root = Path(__file__).resolve().parents[1]
    validation_path = project_root / "q4" / "val.txt"
    test_path = project_root / "q4_test" / "test.txt"

    validation_result = _score(4, validation_path)
    test_result = _score(4, test_path, lambda topic_id: test_path)

    assert validation_result.metrics == {"mse": 0.0, "score": 0.0}
    assert test_result.metrics == {"mse": 0.0, "score": 0.0}


def test_count_submission_matches_rows_by_image_id_not_order(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.txt"
    prediction_path = tmp_path / "prediction.txt"
    ground_truth_path.write_text("one.png 1\ntwo.png 2\n", encoding="utf-8")
    prediction_path.write_text("two.png 2.5\none.png 1.5\n", encoding="utf-8")

    validate_count_submission(prediction_path, ground_truth_path)


def test_count_submission_rejects_missing_image(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.txt"
    prediction_path = tmp_path / "prediction.txt"
    ground_truth_path.write_text("one.png 1\ntwo.png 2\n", encoding="utf-8")
    prediction_path.write_text("one.png 1\n", encoding="utf-8")

    with pytest.raises(FormatValidationError, match="缺少 1 个图片: two.png"):
        validate_count_submission(prediction_path, ground_truth_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "not-a-number"])
def test_count_file_rejects_invalid_count(tmp_path, value):
    path = tmp_path / "prediction.txt"
    path.write_text(f"one.png {value}\n", encoding="utf-8")

    with pytest.raises(FormatValidationError):
        load_count_file(path)
