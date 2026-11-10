"""Search the catalog and pick a film from a person's favourites.

Picks use the Tag Genome + popularity method. results/stage1_baselines.md shows how it and
the popularity weight were chosen, and pipeline/evaluate.py computes the same score.
"""

import json
from urllib.parse import quote

import psycopg

from app.config import settings

POPULARITY_WEIGHT = 1.0
THEME_COUNT = 3

# The columns film_card() reads, in order, from a query that aliases films as "film".
FILM_CARD_COLUMNS = (
    "film.movie_id, film.title, film.year, film.genres, film.directors, film.runtime,"
    " film.certification, film.wikipedia_title"
)

_SEARCH = """
SELECT movie_id, title, year, directors
FROM films
WHERE title ILIKE %(contains)s OR title %% %(query)s
ORDER BY title ILIKE %(prefix)s DESC,
         similarity(title, %(query)s) + 0.25 * popularity DESC,
         movie_id
LIMIT %(limit)s
"""

_PICK = f"""
WITH taste AS (SELECT avg(genome) AS vector FROM films WHERE movie_id = ANY(%(liked)s))
SELECT {FILM_CARD_COLUMNS}, film.genome::text, taste.vector::text
FROM films AS film, taste
WHERE taste.vector IS NOT NULL AND film.movie_id <> ALL(%(excluded)s)
ORDER BY (1 - (film.genome <=> taste.vector)) + %(weight)s * film.popularity DESC, film.movie_id
LIMIT 1
"""

_CLOSEST_FAVOURITE = """
SELECT title FROM films
WHERE movie_id = ANY(%(liked)s)
ORDER BY genome <=> %(film)s::vector, movie_id
LIMIT 1
"""

_DISPLAYABLE_TAGS = "SELECT position, tag FROM genome_tags WHERE displayable"


def _like_pattern(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_films(query: str, limit: int) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    pattern = _like_pattern(query)
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        rows = conn.execute(
            _SEARCH,
            {"query": query, "contains": f"%{pattern}%", "prefix": f"{pattern}%", "limit": limit},
        ).fetchall()
    return [
        {"movie_id": movie_id, "title": title, "year": year, "directors": directors}
        for movie_id, title, year, directors in rows
    ]


def shared_themes(taste: list[float], film: list[float], tags: list[tuple[int, str]]) -> list[str]:
    """Return the displayable tags that both the favourites and the film score highly on."""
    ranked = sorted(tags, key=lambda tag: min(taste[tag[0]], film[tag[0]]), reverse=True)
    return [name for _, name in ranked[:THEME_COUNT]]


def film_card(row: tuple) -> dict:
    """Describe a film for display, from a row starting with FILM_CARD_COLUMNS."""
    movie_id, title, year, genres, directors, runtime, certification, wikipedia_title = row[:8]
    return {
        "movie_id": movie_id,
        "title": title,
        "year": year,
        "genres": genres,
        "directors": directors,
        "runtime": runtime,
        "certification": certification,
        "wikipedia_url": (
            f"https://en.wikipedia.org/wiki/{quote(wikipedia_title.replace(' ', '_'))}"
            if wikipedia_title
            else None
        ),
        "where_to_watch_url": f"https://www.justwatch.com/ca/search?q={quote(title)}",
    }


def pick_film(liked: list[int], rejected: list[int]) -> dict | None:
    """Return the best film the person hasn't named or rejected, or None if there isn't one."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        row = conn.execute(
            _PICK,
            {"liked": liked, "excluded": liked + rejected, "weight": POPULARITY_WEIGHT},
        ).fetchone()
        if row is None:
            return None

        film_vector = row[8]
        similar_to = conn.execute(
            _CLOSEST_FAVOURITE, {"liked": liked, "film": film_vector}
        ).fetchone()[0]
        tags = conn.execute(_DISPLAYABLE_TAGS).fetchall()

    return {
        "film": film_card(row),
        "reason": {
            "similar_to": similar_to,
            "themes": shared_themes(json.loads(row[9]), json.loads(film_vector), tags),
        },
    }
