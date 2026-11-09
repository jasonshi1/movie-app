import psycopg
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_health_reports_pgvector_from_real_database():
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["pgvector"]


def test_health_is_503_when_database_is_down(monkeypatch):
    def refuse_connection():
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main, "vector_extension_version", refuse_connection)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": False, "pgvector": None}


def test_health_is_503_when_pgvector_is_missing(monkeypatch):
    monkeypatch.setattr(main, "vector_extension_version", lambda: None)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": True, "pgvector": None}
