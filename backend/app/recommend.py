"""Search the catalog and pick a film from a person's favourites.

Picks use the trained ranker when the catalog includes one (results/stage3_ranker.md), and
otherwise the Stage 1 Tag Genome + popularity method (results/stage1_baselines.md). Either way
the app computes the same score as the evaluation. The reason shown with a pick always comes
from Tag Genome themes, because people can read those.
"""

import json
from urllib.parse import quote

import numpy as np
import psycopg

from app.config import settings
from app.ranker import encoder_from_bytes, taste_vectors

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

def _pick_query(order_by: str) -> str:
    """Find the best film first, then fetch its details.

    Selecting the details while sorting made PostgreSQL turn every film's genome into text,
    which took about 0.6 seconds instead of 0.01.
    """
    return f"""
WITH taste AS (SELECT avg(genome) AS vector FROM films WHERE movie_id = ANY(%(liked)s)),
best AS (
    SELECT film.movie_id FROM films AS film, taste
    WHERE taste.vector IS NOT NULL AND film.movie_id <> ALL(%(excluded)s)
    ORDER BY {order_by}, film.movie_id
    LIMIT 1
)
SELECT {FILM_CARD_COLUMNS}, film.genome::text, taste.vector::text
FROM best JOIN films AS film USING (movie_id), taste
"""


_PICK = _pick_query(
    "(1 - (film.genome <=> taste.vector)) + %(weight)s * film.popularity DESC"
)
# ranker_output ends with the film's bias and the taste vector ends with 1, so their inner
# product is the ranker's score. pgvector's <#> is the negative inner product.
_PICK_RANKED = _pick_query("film.ranker_output <#> %(ranker_taste)s::vector")

_RANKER = "SELECT encoder FROM ranker"
_FAVOURITE_INPUTS = """
SELECT ranker_input::text FROM films
WHERE movie_id = ANY(%(liked)s) AND ranker_input IS NOT NULL
"""

_CLOSEST_FAVOURITE = """
SELECT title FROM films
WHERE movie_id = ANY(%(liked)s)
ORDER BY genome <=> %(film)s::vector, movie_id
LIMIT 1
"""

_DISPLAYABLE_TAGS = "SELECT position, tag FROM genome_tags WHERE displayable"


def vector_literal(vector: np.ndarray) -> str:
    """Format a vector the way pgvector reads it."""
    return "[" + ",".join(f"{value:.6f}" for value in np.asarray(vector).tolist()) + "]"


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


def ranker_taste(conn: psycopg.Connection, liked: list[int]) -> np.ndarray | None:
    """The ranker's taste vector with a 1 appended, or None if there's no ranker to use."""
    encoder = conn.execute(_RANKER).fetchone()
    if encoder is None:
        return None
    rows = conn.execute(_FAVOURITE_INPUTS, {"liked": liked}).fetchall()
    if not rows:
        return None
    vectors = np.array([json.loads(text) for (text,) in rows], dtype=np.float32)
    taste = taste_vectors([vectors], encoder_from_bytes(bytes(encoder[0])))[0]
    return np.append(taste, 1.0)


def pick_film(liked: list[int], rejected: list[int]) -> dict | None:
    """Return the best film the person hasn't named or rejected, or None if there isn't one."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        parameters = {"liked": liked, "excluded": liked + rejected, "weight": POPULARITY_WEIGHT}
        taste = ranker_taste(conn, liked)
        if taste is None:
            row = conn.execute(_PICK, parameters).fetchone()
        else:
            row = conn.execute(
                _PICK_RANKED, parameters | {"ranker_taste": vector_literal(taste)}
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
