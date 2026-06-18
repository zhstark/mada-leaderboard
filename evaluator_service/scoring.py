from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from .config import ground_truth_dir_for_topic
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
    try:
        await repository.mark_scoring(submission_id)
        result = await asyncio.to_thread(_score, topic_id, prediction_dir)
        await repository.mark_succeeded(submission_id, result.score, result.metrics)
    except Exception as exc:  # noqa: BLE001 - background task must persist any failure state.
        logger.exception("Scoring job failed for submission %s", submission_id)
        try:
            await repository.mark_failed(submission_id, str(exc))
        except Exception:  # noqa: BLE001 - original scoring exception is already logged.
            logger.exception("Failed to persist scoring failure for submission %s", submission_id)


def _score(topic_id: int, prediction_dir: Path):
    ground_truth_dir = ground_truth_dir_for_topic(topic_id)
    prediction_files = validate_prediction_files(prediction_dir, ground_truth_dir)
    ground_truth = load_ground_truth(ground_truth_dir)
    predictions = load_predictions(prediction_files)
    return evaluate_detection(ground_truth, predictions, topic_id)
