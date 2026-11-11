"""Measure how strongly a described mood should count against long-term taste in search.

Each trial pairs a training user's first five favourites with a request for a random theme,
such as "heist" or "time travel". A result counts as a theme hit if the Tag Genome scores the
film at 0.6 or more for that theme. Taste fit is the results' similarity to the favourites,
relative to ignoring the mood.

Theme names also appear in the text each film was embedded from, so hit rates are optimistic.
Use them to compare weights, not as a measure of real quality.
"""

import json

import numpy as np
import polars as pl

from app.catalog_tools import MOOD_WEIGHT, SEARCH_LIMIT
from app.embedding import embed_query
from pipeline.catalog import PROCESSED_DIR, film_popularity
from pipeline.embed import normalize
from pipeline.evaluate import RESULTS_DIR, build_users
from pipeline.tag_policy import is_displayable

WEIGHTS = (0, 1, 2, 3, 4, 6, 8, 12)
TRIALS = 300
THEME_SCORE = 0.6
MIN_FILMS_WITH_THEME = 40
FAVOURITES = 5
SEED = 3


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    raw_genome = np.load(PROCESSED_DIR / "genome.npy")
    genome = normalize(raw_genome)
    text = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = film_popularity(films)
    tags = json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8"))

    themes = [
        index
        for index, tag in enumerate(tags)
        if is_displayable(tag) and (raw_genome[:, index] >= THEME_SCORE).sum() >= MIN_FILMS_WITH_THEME
    ]
    ratings = pl.read_parquet(PROCESSED_DIR / "ratings.parquet").filter(pl.col("split") == "train")
    users = build_users(ratings, films["movie_id"].to_numpy(), TRIALS, seed=SEED)
    requests = np.random.default_rng(SEED).choice(themes, size=len(users))
    request_vectors = np.array([embed_query(tags[theme]) for theme in requests], dtype=np.float32)

    rows = []
    for weight in WEIGHTS:
        hits, fits = [], []
        for user, theme, request in zip(users, requests, request_vectors, strict=True):
            favourites = user.favourites[:FAVOURITES]
            taste = genome @ normalize(genome[favourites].mean(axis=0, keepdims=True))[0]
            scores = taste + popularity + weight * (text @ request)
            scores[favourites] = -np.inf
            top = np.argpartition(-scores, SEARCH_LIMIT)[:SEARCH_LIMIT]
            hits.append((raw_genome[top, theme] >= THEME_SCORE).mean())
            fits.append(taste[top].mean())
        rows.append({"weight": weight, "theme_hit_rate": float(np.mean(hits)), "taste_fit": float(np.mean(fits))})

    baseline = rows[0]["taste_fit"]
    table = "\n".join(
        f"| {row['weight']} | {row['theme_hit_rate']:.2f} | {row['taste_fit'] / baseline:.2f} |"
        for row in rows
    )
    report = (
        "# Mood weight\n\n"
        f"{len(users)} training users, each asking for one of {len(themes)} themes. "
        f"The app uses weight {MOOD_WEIGHT}.\n\n"
        f"| Mood weight | Theme hits in top {SEARCH_LIMIT} | Taste fit vs no mood |\n"
        "|---:|---:|---:|\n"
        f"{table}\n\n"
        "Theme names also appear in the embedded film text, so hit rates are optimistic.\n"
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "stage2_mood_weight.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
