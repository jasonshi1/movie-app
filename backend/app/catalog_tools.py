"""Catalog queries behind the Claude conversation's tools.

Every hard limit a person sets (runtime, years, genres, ratings, films already seen or rejected)
is enforced here in SQL, so Claude can only choose among films that already satisfy them.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from app.config import settings
from app.recommend import (
    CARD_WIDTH,
    FILM_CARD_COLUMNS,
    POPULARITY_WEIGHT,
    film_card,
    vector_literal,
)

SEARCH_LIMIT = 8
# How much the described mood counts against long-term taste. Mood similarities vary little
# from film to film, so a weight of 1 barely changed results; 4 is where theme matches stop
# gaining much for the taste they cost. See results/stage2_mood_weight.md.
MOOD_WEIGHT = 4.0
THEMES_PER_FILM = 5
DESCRIPTION_CHARS = 1200

GENRES = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
)
CERTIFICATIONS = ("G", "PG", "PG-13", "R", "NC-17")


@dataclass(frozen=True)
class SearchFilters:
    query: str = ""
    genres: tuple[str, ...] = ()
    exclude_genres: tuple[str, ...] = ()
    max_runtime: int | None = None
    min_year: int | None = None
    max_year: int | None = None
    certifications: tuple[str, ...] | None = None


# Ranks first and fetches details for the winners only. Selecting every film's genome as text
# while sorting made unfiltered searches take about 0.5 seconds.
_SEARCH = f"""
WITH taste AS (SELECT avg(genome) AS vector FROM films WHERE movie_id = ANY(%(liked)s)),
best AS (
    SELECT film.movie_id,
           coalesce(1 - (film.genome <=> taste.vector), 0)
           + %(popularity_weight)s * film.popularity
           + coalesce(%(mood_weight)s * (1 - (film.embedding <=> %(mood)s::vector)), 0) AS score
    FROM films AS film, taste
    WHERE film.movie_id <> ALL(%(excluded)s)
      AND (%(max_runtime)s::integer IS NULL OR film.runtime <= %(max_runtime)s)
      AND (%(min_year)s::integer IS NULL OR film.year >= %(min_year)s)
      AND (%(max_year)s::integer IS NULL OR film.year <= %(max_year)s)
      AND (cardinality(%(genres)s::text[]) = 0 OR film.genres && %(genres)s::text[])
      AND NOT (film.genres && %(exclude_genres)s::text[])
      AND (%(certifications)s::text[] IS NULL OR film.certification = ANY(%(certifications)s))
    ORDER BY score DESC, film.movie_id
    LIMIT %(limit)s
)
SELECT {FILM_CARD_COLUMNS}, film.genome::text
FROM best JOIN films AS film USING (movie_id)
ORDER BY best.score DESC, film.movie_id
"""

_DETAILS = f"""
SELECT {FILM_CARD_COLUMNS}, film.genome::text, film.cast_members, film.description
FROM films AS film
WHERE film.movie_id = %(movie_id)s
"""

_DISPLAYABLE_TAGS = "SELECT position, tag FROM genome_tags WHERE displayable"


def _top_themes(genome_text: str, tags: list[tuple[int, str]]) -> list[str]:
    genome = json.loads(genome_text)
    ranked = sorted(tags, key=lambda tag: genome[tag[0]], reverse=True)
    return [name for _, name in ranked[:THEMES_PER_FILM]]


def _summary(row: tuple, tags: list[tuple[int, str]]) -> dict:
    card = film_card(row)
    return {
        "movie_id": card["movie_id"],
        "title": card["title"],
        "year": card["year"],
        "runtime_minutes": card["runtime"],
        "certification": card["certification"],
        "genres": card["genres"],
        "directors": card["directors"],
        "themes": _top_themes(row[CARD_WIDTH], tags),
    }


def search_candidates(
    filters: SearchFilters,
    liked: list[int],
    excluded: list[int],
    embed: Callable[[str], list[float]],
) -> list[dict]:
    """Return the best films that pass every filter, ranked by taste, popularity and mood."""
    mood = embed(filters.query) if filters.query.strip() else None
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        rows = conn.execute(
            _SEARCH,
            {
                "liked": liked,
                "excluded": excluded,
                "max_runtime": filters.max_runtime,
                "min_year": filters.min_year,
                "max_year": filters.max_year,
                "genres": list(filters.genres),
                "exclude_genres": list(filters.exclude_genres),
                "certifications": (
                    list(filters.certifications) if filters.certifications is not None else None
                ),
                "popularity_weight": POPULARITY_WEIGHT,
                "mood_weight": MOOD_WEIGHT,
                "mood": vector_literal(mood) if mood else None,
                "limit": SEARCH_LIMIT,
            },
        ).fetchall()
        tags = conn.execute(_DISPLAYABLE_TAGS).fetchall()
    return [_summary(row, tags) for row in rows]


def film_details(movie_id: int) -> dict | None:
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        row = conn.execute(_DETAILS, {"movie_id": movie_id}).fetchone()
        if row is None:
            return None
        tags = conn.execute(_DISPLAYABLE_TAGS).fetchall()
    return {
        **_summary(row, tags),
        "cast": row[CARD_WIDTH + 1][:6],
        "description": row[CARD_WIDTH + 2][:DESCRIPTION_CHARS],
    }


def film_cards(movie_ids: list[int]) -> dict[int, dict]:
    """Display details for each film that exists, keyed by ID."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        rows = conn.execute(
            f"SELECT {FILM_CARD_COLUMNS} FROM films AS film WHERE film.movie_id = ANY(%s)",
            (movie_ids,),
        ).fetchall()
    return {row[0]: film_card(row) for row in rows}
