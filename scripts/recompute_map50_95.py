from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from evaluator_service.archive import _find_label_root, extract_zip_safely
from evaluator_service.config import Settings, ground_truth_dir_for_topic
from evaluator_service.database import normalize_database_url
from evaluator_service.labels import load_ground_truth, load_predictions, validate_prediction_files
from evaluator_service.metrics import evaluate_detection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    id: UUID
    topic_id: int
    evaluator_job_id: str


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = Settings.from_env()
    database_url = normalize_database_url(args.database_url or settings.database_url)
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    work_dir = Path(args.work_dir or settings.work_dir)
    submissions = fetch_submissions(database_url, args.limit, args.all_succeeded, args.topic_column)

    updated = 0
    skipped = 0
    failed = 0
    for submission in submissions:
        try:
            result = rescore_submission(submission, work_dir, settings)
        except Exception as exc:  # noqa: BLE001 - continue through a batch and report all failures.
            failed += 1
            logger.exception("Failed to rescore submission_id=%s evaluator_job_id=%s: %s", submission.id, submission.evaluator_job_id, exc)
            continue

        logger.info(
            "%s submission_id=%s topic_id=%s evaluator_job_id=%s score=%s metrics=%s",
            "Would update" if not args.apply else "Updating",
            submission.id,
            submission.topic_id,
            submission.evaluator_job_id,
            result.score,
            json.dumps(result.metrics, ensure_ascii=False, sort_keys=True),
        )
        if args.apply:
            update_submission(database_url, submission.id, result.score, result.metrics)
            updated += 1
        else:
            skipped += 1

    logger.info("Done total=%s updated=%s dry_run=%s failed=%s", len(submissions), updated, skipped, failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute historical mAP50_90 submission metrics as mAP50_95 using retained evaluator jobs."
    )
    parser.add_argument("--apply", action="store_true", help="Write recomputed score and metrics back to the database.")
    parser.add_argument("--all-succeeded", action="store_true", help="Recompute all succeeded submissions, not only rows with mAP50_90.")
    parser.add_argument("--database-url", help="Override DATABASE_URL.")
    parser.add_argument("--work-dir", help="Override EVALUATOR_WORK_DIR.")
    parser.add_argument("--limit", type=int, help="Maximum number of submissions to process.")
    parser.add_argument("--topic-column", default="topic_id", help="Topic id column name in competition_prediction_submissions.")
    return parser.parse_args()


def fetch_submissions(
    database_url: str,
    limit: int | None,
    all_succeeded: bool,
    topic_column: str,
) -> list[Submission]:
    if not topic_column.isidentifier():
        raise ValueError(f"Unsafe topic column name: {topic_column}")

    filters = ["status = 'succeeded'", "evaluator_job_id is not null"]
    if not all_succeeded:
        filters.append("metrics ? 'mAP50_90'")

    sql = f"""
        select id, {topic_column} as topic_id, evaluator_job_id
        from public.competition_prediction_submissions
        where {" and ".join(filters)}
        order by evaluated_at nulls last, id
    """
    params: tuple[object, ...] = ()
    if limit is not None:
        sql += " limit %s"
        params = (limit,)

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    return [
        Submission(
            id=row["id"],
            topic_id=int(row["topic_id"]),
            evaluator_job_id=str(row["evaluator_job_id"]),
        )
        for row in rows
    ]


def rescore_submission(submission: Submission, work_dir: Path, settings: Settings):
    prediction_dir = prediction_dir_for_job(work_dir / submission.evaluator_job_id, settings)
    ground_truth_dir = ground_truth_dir_for_topic(submission.topic_id)
    prediction_files = validate_prediction_files(prediction_dir, ground_truth_dir)
    ground_truth = load_ground_truth(ground_truth_dir)
    predictions = load_predictions(prediction_files)
    return evaluate_detection(ground_truth, predictions, submission.topic_id)


def prediction_dir_for_job(job_dir: Path, settings: Settings) -> Path:
    extracted_dir = job_dir / "extracted"
    if extracted_dir.exists():
        return _find_label_root(extracted_dir)

    zip_files = sorted(path for path in job_dir.glob("*.zip") if path.is_file())
    if len(zip_files) == 1:
        return extract_zip_safely(zip_files[0], extracted_dir, settings)

    if not job_dir.exists():
        raise FileNotFoundError(f"Evaluator job directory does not exist: {job_dir}")
    raise FileNotFoundError(f"Expected retained extracted labels or one zip file in {job_dir}")


def update_submission(database_url: str, submission_id: UUID, score: float, metrics: dict[str, float]) -> None:
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                update public.competition_prediction_submissions
                set score = %s,
                    metrics = %s::jsonb,
                    evaluated_at = now(),
                    validation_error = null
                where id = %s
                """,
                (score, json.dumps(metrics, ensure_ascii=False), submission_id),
            )


if __name__ == "__main__":
    main()
