"""Search the catalog and pick films from a person's favourites.

Picks use the trained ranker when the catalog includes one (results/stage3_ranker.md), and
otherwise the Stage 1 Tag Genome + popularity method (results/stage1_baselines.md). Either way
the app computes the same score as the evaluation. The reason shown with a pick always comes
from Tag Genome themes, because people can read those.
"""

import json
import re
from collections.abc import Sequence
from urllib.parse import quote

import numpy as np
import psycopg

from app.config import settings
from app.ranker import encoder_from_bytes, taste_vectors

POPULARITY_WEIGHT = 1.0
THEME_COUNT = 3

# Picks shown together should be a real choice, so a film is passed over when it's from the same
# series as one already chosen. Sequels' genomes are 0.82 to 0.98 similar, and different films
# rarely pass 0.90 (Seven and The Silence of the Lambs), so similarity catches most sequels and
# the title catches the rest, such as Shrek 2 at 0.895.
SAME_SERIES_SIMILARITY = 0.92
# How many of the best films to look through for each pick asked for.
CANDIDATES_PER_PICK = 10
_ARTICLES = {"the", "a", "an"}

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

def _pick_query(cost: str) -> str:
    """Find the best films first, then fetch their details, best first. Lower cost is better.

    Selecting the details while sorting made PostgreSQL turn every film's genome into text,
    which took about 0.6 seconds instead of 0.01.
    """
    return f"""
WITH taste AS (SELECT avg(genome) AS vector FROM films WHERE movie_id = ANY(%(liked)s)),
best AS (
    SELECT film.movie_id, {cost} AS cost FROM films AS film, taste
    WHERE taste.vector IS NOT NULL AND film.movie_id <> ALL(%(excluded)s)
    ORDER BY cost, film.movie_id
    LIMIT %(count)s
)
SELECT {FILM_CARD_COLUMNS}, film.genome::text, taste.vector::text
FROM best JOIN films AS film USING (movie_id), taste
ORDER BY best.cost, best.movie_id
"""


_PICK = _pick_query("-((1 - (film.genome <=> taste.vector)) + %(weight)s * film.popularity)")
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

_SHOWING = "SELECT title, genome::text FROM films WHERE movie_id = ANY(%(showing)s)"


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


def _title_words(title: str) -> list[str]:
    """A title's words for comparing series: lower case, without a leading article or an
    "(a.k.a. ...)" alternative title."""
    words = re.findall(r"[\w']+", re.sub(r"\(.*?\)", " ", title.lower()))
    return words[1:] if words and words[0] in _ARTICLES else words


def same_series(a: tuple[str, np.ndarray], b: tuple[str, np.ndarray]) -> bool:
    """Whether two (title, genome) films look like parts of one series: one title starts the
    other (Shrek, Shrek 2), they share their first two words (Star Wars: Episode IV and V), or
    their themes are almost identical (Alien, Aliens)."""
    words_a, words_b = _title_words(a[0]), _title_words(b[0])
    shorter, longer = sorted((words_a, words_b), key=len)
    if shorter and longer[: len(shorter)] == shorter:
        return True
    if len(shorter) >= 2 and words_a[:2] == words_b[:2]:
        return True
    norms = float(np.linalg.norm(a[1]) * np.linalg.norm(b[1]))
    return norms > 0 and float(np.dot(a[1], b[1])) / norms >= SAME_SERIES_SIMILARITY


def choose_distinct(
    candidates: Sequence[tuple[str, np.ndarray]],
    count: int,
    showing: Sequence[tuple[str, np.ndarray]] = (),
) -> list[int]:
    """Indexes of up to count candidates, taken best first and skipping any in the same series
    as one already chosen or showing. If too few are left, the skipped ones fill the gap."""
    chosen: list[int] = []
    skipped: list[int] = []
    kept = list(showing)
    for index, candidate in enumerate(candidates):
        if len(chosen) == count:
            break
        if any(same_series(candidate, other) for other in kept):
            skipped.append(index)
            continue
        chosen.append(index)
        kept.append(candidate)
    return chosen + skipped[: count - len(chosen)]


def pick_films(
    liked: list[int], rejected: list[int], count: int, showing: Sequence[int] = ()
) -> list[dict]:
    """Return up to count of the best films the person hasn't named or rejected, best first,
    each with the reason: the closest favourite and the themes they share.

    The first is always the ranker's best film. The rest skip films from the same series as
    one already picked, or as the films in showing: picks still on screen, which are left out.
    """
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        parameters = {
            "liked": liked,
            "excluded": liked + rejected + list(showing),
            "weight": POPULARITY_WEIGHT,
            "count": count * CANDIDATES_PER_PICK,
        }
        taste = ranker_taste(conn, liked)
        if taste is None:
            rows = conn.execute(_PICK, parameters).fetchall()
        else:
            rows = conn.execute(
                _PICK_RANKED, parameters | {"ranker_taste": vector_literal(taste)}
            ).fetchall()
        if not rows:
            return []

        def genome(text: str) -> np.ndarray:
            return np.array(json.loads(text), dtype=np.float32)

        on_screen = [
            (title, genome(vector))
            for title, vector in conn.execute(_SHOWING, {"showing": list(showing)}).fetchall()
        ]
        candidates = [(row[1], genome(row[8])) for row in rows]
        rows = [rows[index] for index in choose_distinct(candidates, count, on_screen)]

        tags = conn.execute(_DISPLAYABLE_TAGS).fetchall()
        picks = []
        for row in rows:
            film_vector = row[8]
            similar_to = conn.execute(
                _CLOSEST_FAVOURITE, {"liked": liked, "film": film_vector}
            ).fetchone()[0]
            picks.append(
                {
                    "film": film_card(row),
                    "reason": {
                        "similar_to": similar_to,
                        "themes": shared_themes(json.loads(row[9]), json.loads(film_vector), tags),
                    },
                }
            )
    return picks


def pick_film(liked: list[int], rejected: list[int]) -> dict | None:
    """Return the best film the person hasn't named or rejected, or None if there isn't one."""
    picks = pick_films(liked, rejected, 1)
    return picks[0] if picks else None
