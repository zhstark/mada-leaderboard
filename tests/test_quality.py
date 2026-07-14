from pathlib import Path

import pytest
from openpyxl import Workbook

from evaluator_service.errors import FormatValidationError
from evaluator_service.metrics import evaluate_quality
from evaluator_service.quality import (
    QualityMeasurement,
    load_quality_workbook,
    validate_quality_submission,
)


def test_evaluate_quality_uses_current_dataset_denominators():
    ground_truth = {
        1: QualityMeasurement(1, sugar=10.0, acid=1.0),
        2: QualityMeasurement(2, sugar=30.0, acid=3.0),
    }
    predictions = {
        1: QualityMeasurement(1, sugar=11.0, acid=1.1),
        2: QualityMeasurement(2, sugar=33.0, acid=3.3),
    }

    result = evaluate_quality(ground_truth, predictions)

    assert result.score == pytest.approx(0.1)
    assert result.metrics == {
        "sugar_error": 0.1,
        "acid_error": 0.1,
        "error": 0.1,
        "score": 0.1,
    }


def test_evaluate_quality_perfect_prediction_has_zero_error():
    ground_truth = {1: QualityMeasurement(1, sugar=10.0, acid=1.0)}

    result = evaluate_quality(ground_truth, ground_truth)

    assert result.score == 0.0
    assert result.metrics["error"] == 0.0


def test_repository_ground_truth_workbooks_are_valid_and_independent():
    project_root = Path(__file__).resolve().parents[1]

    validation = load_quality_workbook(project_root / "q3" / "val_result.xlsx")
    test = load_quality_workbook(project_root / "q3_test" / "test_result.xlsx")

    assert len(validation) == 30
    assert len(test) == 30
    assert sum(item.sugar for item in validation.values()) == pytest.approx(386.4166666666667)
    assert sum(item.sugar for item in test.values()) == pytest.approx(396.23333333333335)


def test_quality_submission_matches_rows_by_sample_id_not_order(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.xlsx"
    prediction_path = tmp_path / "prediction.xlsx"
    _write_workbook(ground_truth_path, [(1, 10.0, 1.0), (2, 20.0, 2.0)])
    _write_workbook(prediction_path, [(2, 19.0, 1.9), (1, 11.0, 1.1)])

    validate_quality_submission(prediction_path, ground_truth_path)


def test_quality_submission_rejects_missing_sample_id(tmp_path):
    ground_truth_path = tmp_path / "ground_truth.xlsx"
    prediction_path = tmp_path / "prediction.xlsx"
    _write_workbook(ground_truth_path, [(1, 10.0, 1.0), (2, 20.0, 2.0)])
    _write_workbook(prediction_path, [(1, 11.0, 1.1)])

    with pytest.raises(FormatValidationError, match="缺少 1 个序号: 2"):
        validate_quality_submission(prediction_path, ground_truth_path)


def test_quality_workbook_rejects_formula(tmp_path):
    path = tmp_path / "prediction.xlsx"
    _write_workbook(path, [(1, "=10+1", 1.0)])

    with pytest.raises(FormatValidationError, match="不允许使用公式"):
        load_quality_workbook(path)


def _write_workbook(path: Path, rows: list[tuple[object, object, object]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["序号", "糖度", "酸度"])
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
