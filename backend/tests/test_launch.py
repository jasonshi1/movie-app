import uuid

import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from app import conversations, main
from pipeline.posters import POSTER_HOSTS


def test_api_responses_carry_security_headers(empty_catalog):
    headers = TestClient(main.app).get("/api/health").headers

    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    image_sources = next(
        part for part in headers["Content-Security-Policy"].split(";") if "img-src" in part
    ).split()[1:]
    assert sorted(image_sources) == sorted(
        ["'self'", "data:", *(f"https://{host}" for host in POSTER_HOSTS)]
    )


def test_the_built_app_is_served_alongside_the_api(tmp_path):
    application = FastAPI()
    application.get("/api/health")(lambda: {"status": "ok"})
    (tmp_path / "index.html").write_text("<title>Tonight's Pick</title>", encoding="utf-8")

    assert main.serve_frontend(application, tmp_path)
    client = TestClient(application)
    assert "Tonight's Pick" in client.get("/").text
    assert client.get("/api/health").json() == {"status": "ok"}


def test_nothing_is_served_before_the_app_is_built(tmp_path):
    assert not main.serve_frontend(FastAPI(), tmp_path)


def test_old_conversations_are_deleted(empty_catalog):
    with psycopg.connect(empty_catalog) as conn:
        conn.execute("TRUNCATE conversations")
        for age in (40, 5):
            conn.execute(
                "INSERT INTO conversations (id, version, messages, state, updated_at)"
                " VALUES (%s, 1, %s, %s, now() - make_interval(days => %s))",
                (uuid.uuid4(), Jsonb([]), Jsonb({}), age),
            )

    assert conversations.delete_older_than(30) == 1
    with psycopg.connect(empty_catalog) as conn:
        assert conn.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1
