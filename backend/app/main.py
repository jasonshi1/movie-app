import psycopg
from fastapi import FastAPI, Response

from app.db import vector_extension_version

app = FastAPI(title="Tonight's Pick API", version="0.1.0")


@app.get("/api/health")
def health(response: Response) -> dict[str, str | bool | None]:
    try:
        pgvector = vector_extension_version()
    except psycopg.OperationalError:
        response.status_code = 503
        return {"status": "unavailable", "database": False, "pgvector": None}

    if pgvector is None:
        response.status_code = 503
        return {"status": "unavailable", "database": True, "pgvector": None}

    return {"status": "ok", "database": True, "pgvector": pgvector}
