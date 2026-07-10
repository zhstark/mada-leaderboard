from psycopg.conninfo import conninfo_to_dict

import asyncio
from uuid import UUID

from evaluator_service.database import TestSubmissionRepository, normalize_database_url


def test_normalize_database_url_accepts_sqlalchemy_asyncpg_scheme():
    normalized = normalize_database_url("postgresql+asyncpg://postgres:postgres@localhost:54322/postgres")

    assert normalized == "postgresql://postgres:postgres@localhost:54322/postgres"
    assert conninfo_to_dict(normalized)["dbname"] == "postgres"


def test_normalize_database_url_keeps_native_psycopg_url():
    url = "postgresql://postgres:postgres@localhost:54322/postgres"

    assert normalize_database_url(url) == url


def test_normalize_database_url_keeps_libpq_key_value_string():
    conninfo = "host=localhost port=54322 dbname=postgres user=postgres password=postgres"

    assert normalize_database_url(conninfo) == conninfo
    assert conninfo_to_dict(conninfo)["dbname"] == "postgres"


def test_test_submission_repository_updates_test_table(monkeypatch):
    captured = []

    async def capture_execute(self, sql, params):
        captured.append((sql, params))

    monkeypatch.setattr(TestSubmissionRepository, "_execute", capture_execute)

    repository = TestSubmissionRepository("postgresql://postgres:postgres@localhost:54322/postgres")
    submission_id = UUID("44444444-4444-4444-4444-444444444444")

    asyncio.run(repository.mark_queued(submission_id, "job-1"))

    assert "public.competition_test_prediction_submissions" in captured[0][0]
    assert captured[0][1] == ("job-1", submission_id)
