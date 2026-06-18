from __future__ import annotations

import json
import logging
from uuid import UUID

import psycopg


logger = logging.getLogger(__name__)


class SubmissionRepository:
    def __init__(self, database_url: str | None):
        self.database_url = normalize_database_url(database_url)

    async def mark_queued(self, submission_id: UUID, job_id: str) -> None:
        await self._execute(
            """
            update public.competition_prediction_submissions
            set status = 'queued',
                evaluator_job_id = %s,
                validation_error = null
            where id = %s
            """,
            (job_id, submission_id),
        )

    async def mark_scoring(self, submission_id: UUID) -> None:
        await self._execute(
            """
            update public.competition_prediction_submissions
            set status = 'scoring',
                validation_error = null
            where id = %s
            """,
            (submission_id,),
        )

    async def mark_succeeded(self, submission_id: UUID, score: float, metrics: dict[str, float]) -> None:
        await self._execute(
            """
            update public.competition_prediction_submissions
            set status = 'succeeded',
                score = %s,
                metrics = %s::jsonb,
                evaluated_at = now(),
                validation_error = null
            where id = %s
            """,
            (score, json.dumps(metrics, ensure_ascii=False), submission_id),
        )

    async def mark_failed(self, submission_id: UUID, error: str) -> None:
        await self._execute(
            """
            update public.competition_prediction_submissions
            set status = 'failed',
                validation_error = %s,
                evaluated_at = now()
            where id = %s
            """,
            (error[:2000], submission_id),
        )

    async def _execute(self, sql: str, params: tuple[object, ...]) -> None:
        if not self.database_url:
            logger.warning("DATABASE_URL is not configured; skipping submission status update")
            return

        async with await psycopg.AsyncConnection.connect(self.database_url) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)


def normalize_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return database_url

    scheme_separator = "://"
    if scheme_separator not in database_url:
        return database_url

    scheme, rest = database_url.split(scheme_separator, 1)
    if scheme.startswith("postgresql+") or scheme.startswith("postgres+"):
        return f"postgresql{scheme_separator}{rest}"
    return database_url
