from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable
from uuid import UUID

from .config import ground_truth_dir_for_topic, test_ground_truth_dir_for_topic
from .database import SubmissionRepository
from .labels import load_ground_truth, load_predictions, validate_prediction_files
from .metrics import evaluate_detection


logger = logging.getLogger(__name__)


async def run_scoring_job(
    *,
    submission_id: UUID,
    topic_id: int,
    prediction_dir: Path,
    repository: SubmissionRepository,
) -> None:
    await _run_scoring_job(
        submission_id=submission_id,
        topic_id=topic_id,
        prediction_dir=prediction_dir,
        repository=repository,
        ground_truth_dir_resolver=ground_truth_dir_for_topic,
    )


async def run_test_scoring_job(
    *,
    submission_id: UUID,
    topic_id: int,
    prediction_dir: Path,
    repository: SubmissionRepository,
) -> None:
    await _run_scoring_job(
        submission_id=submission_id,
        topic_id=topic_id,
        prediction_dir=prediction_dir,
        repository=repository,
        ground_truth_dir_resolver=test_ground_truth_dir_for_topic,
    )


async def _run_scoring_job(
    *,
    submission_id: UUID,
    topic_id: int,
    prediction_dir: Path,
    repository: SubmissionRepository,
    ground_truth_dir_resolver: Callable[[int], Path],
) -> None:
    try:
        await repository.mark_scoring(submission_id)
        result = await asyncio.to_thread(_score, topic_id, prediction_dir, ground_truth_dir_resolver)
        await repository.mark_succeeded(submission_id, result.score, result.metrics)
    except Exception as exc:  # noqa: BLE001 - background task must persist any failure state.
        logger.exception(
            "Scoring job failed submission_id=%s topic_id=%s prediction_dir=%s",
            submission_id,
            topic_id,
            prediction_dir,
        )
        try:
            await repository.mark_failed(submission_id, str(exc))
        except Exception:  # noqa: BLE001 - original scoring exception is already logged.
            logger.exception(
                "Failed to persist scoring failure submission_id=%s topic_id=%s prediction_dir=%s scoring_error=%s",
                submission_id,
                topic_id,
                prediction_dir,
                exc,
            )


def _score(
    topic_id: int,
    prediction_dir: Path,
    ground_truth_dir_resolver: Callable[[int], Path] = ground_truth_dir_for_topic,
):
    ground_truth_dir = ground_truth_dir_resolver(topic_id)
    prediction_files = validate_prediction_files(prediction_dir, ground_truth_dir)
    ground_truth = load_ground_truth(ground_truth_dir)
    predictions = load_predictions(prediction_files)
    return evaluate_detection(ground_truth, predictions, topic_id)
