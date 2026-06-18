from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from .labels import Box


MAP_50_90_THRESHOLDS = tuple(round(value / 100, 2) for value in range(50, 91, 5))


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    score: float


def evaluate_detection(ground_truth: list[Box], predictions: list[Box], topic_id: int) -> EvaluationResult:
    map50, map50_90 = ultralytics_map(ground_truth, predictions, MAP_50_90_THRESHOLDS)

    if topic_id == 1:
        score = 0.5 * map50 + 0.5 * map50_90
        metrics = {
            "mAP50": round(map50, 6),
            "mAP50_90": round(map50_90, 6),
            "score": round(score, 6),
        }
    elif topic_id == 2:
        score = map50_90
        metrics = {
            "mAP50_90": round(map50_90, 6),
            "score": round(score, 6),
        }
    else:
        raise ValueError(f"Unsupported topic_id: {topic_id}")

    return EvaluationResult(metrics=metrics, score=round(score, 6))


def ultralytics_map(
    ground_truth: list[Box],
    predictions: list[Box],
    iou_thresholds: tuple[float, ...],
) -> tuple[float, float]:
    """Compute mAP using Ultralytics' AP implementation for already-parsed YOLO labels."""
    target_classes = np.array([box.class_id for box in ground_truth], dtype=np.int64)
    if target_classes.size == 0:
        return 0.0, 0.0

    stats = _build_detection_stats(ground_truth, predictions, iou_thresholds)
    if stats.confidence.size == 0:
        return 0.0, 0.0

    _, _, _, _, _, ap, _, _, _, _, _, _ = ap_per_class(
        stats.true_positive,
        stats.confidence,
        stats.predicted_classes,
        target_classes,
        plot=False,
    )
    if ap.size == 0:
        return 0.0, 0.0

    map50 = float(ap[:, 0].mean())
    map50_90 = float(ap.mean())
    return map50, map50_90


@dataclass(frozen=True)
class _DetectionStats:
    true_positive: np.ndarray
    confidence: np.ndarray
    predicted_classes: np.ndarray


def _build_detection_stats(
    ground_truth: list[Box],
    predictions: list[Box],
    iou_thresholds: tuple[float, ...],
) -> _DetectionStats:
    gt_by_image = _group_by_image(ground_truth)
    predictions_by_image = _group_by_image(predictions)
    all_image_ids = sorted(set(gt_by_image) | set(predictions_by_image))

    true_positive: list[np.ndarray] = []
    confidence: list[float] = []
    predicted_classes: list[int] = []

    for image_id in all_image_ids:
        image_predictions = predictions_by_image.get(image_id, [])
        if not image_predictions:
            continue

        image_ground_truth = gt_by_image.get(image_id, [])
        if image_ground_truth:
            gt_boxes = _boxes_to_tensor(image_ground_truth)
            pred_boxes = _boxes_to_tensor(image_predictions)
            true_classes = torch.tensor([box.class_id for box in image_ground_truth], dtype=torch.int64)
            pred_classes = torch.tensor([box.class_id for box in image_predictions], dtype=torch.int64)
            image_iou = box_iou(gt_boxes, pred_boxes)
            image_tp = _match_predictions(pred_classes, true_classes, image_iou, iou_thresholds)
        else:
            image_tp = np.zeros((len(image_predictions), len(iou_thresholds)), dtype=bool)

        true_positive.append(image_tp)
        confidence.extend(box.confidence for box in image_predictions)
        predicted_classes.extend(box.class_id for box in image_predictions)

    if not true_positive:
        return _DetectionStats(
            true_positive=np.zeros((0, len(iou_thresholds)), dtype=bool),
            confidence=np.array([], dtype=np.float64),
            predicted_classes=np.array([], dtype=np.int64),
        )

    return _DetectionStats(
        true_positive=np.concatenate(true_positive, axis=0),
        confidence=np.array(confidence, dtype=np.float64),
        predicted_classes=np.array(predicted_classes, dtype=np.int64),
    )


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iou_thresholds: tuple[float, ...],
) -> np.ndarray:
    # Mirrors Ultralytics BaseValidator.match_predictions without requiring a model validator instance.
    correct = np.zeros((pred_classes.shape[0], len(iou_thresholds)), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    class_iou = (iou * correct_class).cpu().numpy()

    for threshold_index, threshold in enumerate(iou_thresholds):
        matches = np.array(np.nonzero(class_iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True

    return correct


def _group_by_image(boxes: list[Box]) -> dict[str, list[Box]]:
    by_image: dict[str, list[Box]] = defaultdict(list)
    for box in boxes:
        by_image[box.image_id].append(box)
    return by_image


def _boxes_to_tensor(boxes: list[Box]) -> torch.Tensor:
    return torch.tensor([box.xyxy for box in boxes], dtype=torch.float32)
