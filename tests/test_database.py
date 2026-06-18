from psycopg.conninfo import conninfo_to_dict

from evaluator_service.database import normalize_database_url


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
