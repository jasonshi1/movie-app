import psycopg

from app.config import settings


def vector_extension_version() -> str | None:
    """Return the installed pgvector version, or None if the extension is missing."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    return row[0] if row else None


def newest_film_year() -> int | None:
    """The release year of the newest film in the catalog, or None if it's empty."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        return conn.execute("SELECT max(year) FROM films").fetchone()[0]
