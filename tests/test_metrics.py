import pytest

from evaluator_service.labels import Box
from evaluator_service.metrics import MAP_50_95_THRESHOLDS, evaluate_detection, ultralytics_map

PERFECT_ULTRALYTICS_AP = 0.995


def test_perfect_prediction_uses_ultralytics_ap_value():
    ground_truth = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2)]
    predictions = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2, 0.9)]

    map50, map50_95 = ultralytics_map(ground_truth, predictions, MAP_50_95_THRESHOLDS)

    assert map50 == pytest.approx(PERFECT_ULTRALYTICS_AP)
    assert map50_95 == pytest.approx(PERFECT_ULTRALYTICS_AP)


def test_duplicate_prediction_uses_ultralytics_ap_matching():
    ground_truth = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2)]
    predictions = [
        Box("image-1", 0, 0.5, 0.5, 0.2, 0.2, 0.9),
        Box("image-1", 0, 0.5, 0.5, 0.2, 0.2, 0.8),
    ]

    map50, map50_95 = ultralytics_map(ground_truth, predictions, MAP_50_95_THRESHOLDS)

    assert map50 == pytest.approx(PERFECT_ULTRALYTICS_AP)
    assert map50_95 == pytest.approx(PERFECT_ULTRALYTICS_AP)


def test_false_prediction_scores_zero():
    ground_truth = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2)]
    predictions = [Box("image-1", 0, 0.1, 0.1, 0.1, 0.1, 0.9)]

    map50, map50_95 = ultralytics_map(ground_truth, predictions, MAP_50_95_THRESHOLDS)

    assert map50 == 0.0
    assert map50_95 == 0.0


def test_topic_scores_use_expected_formula():
    ground_truth = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2)]
    predictions = [Box("image-1", 0, 0.5, 0.5, 0.2, 0.2, 0.9)]

    q1 = evaluate_detection(ground_truth, predictions, topic_id=1)
    q2 = evaluate_detection(ground_truth, predictions, topic_id=2)

    assert q1.metrics == {"mAP50": 0.995, "mAP50_95": 0.995, "score": 0.995}
    assert q1.score == 0.995
    assert q2.metrics == {"mAP50_95": 0.995, "score": 0.995}
    assert q2.score == 0.995


def test_map_50_95_thresholds_include_95():
    assert MAP_50_95_THRESHOLDS == pytest.approx((0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95))
