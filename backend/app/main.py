from typing import Annotated

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.db import vector_extension_version
from app.recommend import pick_film, search_films

app = FastAPI(title="Tonight's Pick API", version="0.1.0")


class FilmMatch(BaseModel):
    movie_id: int
    title: str
    year: int | None
    directors: list[str]


class PickRequest(BaseModel):
    liked: list[int] = Field(min_length=1, max_length=50)
    rejected: list[int] = Field(default_factory=list, max_length=500)


class PickedFilm(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    directors: list[str]
    runtime: int | None
    certification: str | None
    wikipedia_url: str | None
    where_to_watch_url: str


class PickReason(BaseModel):
    similar_to: str
    themes: list[str]


class PickResponse(BaseModel):
    film: PickedFilm
    reason: PickReason


@app.exception_handler(psycopg.OperationalError)
def database_unavailable(_: Request, __: psycopg.OperationalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "The database isn't available."})


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


@app.get("/api/films/search")
def search(
    q: Annotated[str, Query(max_length=100)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> list[FilmMatch]:
    return search_films(q, limit)


@app.post("/api/pick")
def pick(request: PickRequest) -> PickResponse:
    result = pick_film(request.liked, request.rejected)
    if result is None:
        raise HTTPException(status_code=404, detail="No film is available to pick.")
    return result
