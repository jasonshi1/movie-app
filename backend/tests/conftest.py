import psycopg
import pytest

from app.config import settings
from pipeline.load_db import apply_schema

TEST_DATABASE = "movie_app_test"


@pytest.fixture(scope="session")
def test_database_url():
    """Create a throwaway database so tests never touch the loaded catalog."""
    admin_url = settings.database_url
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {TEST_DATABASE}")

    url = f"{admin_url.rsplit('/', 1)[0]}/{TEST_DATABASE}"
    with psycopg.connect(url) as conn:
        apply_schema(conn)

    yield url

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)")


@pytest.fixture
def empty_catalog(test_database_url, monkeypatch):
    monkeypatch.setattr(settings, "database_url", test_database_url)
    with psycopg.connect(test_database_url) as conn:
        conn.execute("TRUNCATE films, genome_tags")
    return test_database_url
