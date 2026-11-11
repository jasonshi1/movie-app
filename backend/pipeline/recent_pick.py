"""Choose and check how the app's recent pick ranks films from the last three years.

The ranker learned from MovieLens ratings, which end in October 2023, so the 2024 to 2026 films
the app's last pick comes from have no ratings, and nobody's liking of them can be checked. This
plays the same situation out three years earlier instead, as the Stage 3 new-film check does:
the ranker is trained again as if the MovieLens films from 2021 on had no ratings. For each
validation user who liked one of those films after their favourites, it checks whether the one
picked for them from those films is one they liked, against simpler ways to choose.

On its own the ranker picks a liked film less often than taking the most-viewed one, because
a film without ratings gets only a rough guess at how widely liked it is. So the recent pick
adds the film's popularity from page views, as the app stores it, to the ranker's score. The
weight is chosen on training users, whose ratings of these films the retrained model never
saw, so the validation users still judge it fresh. Results go to
results/stage6_recent_pick.md.

Run it after pipeline.ranker, which trains the model the app uses.
"""

import json

import numpy as np
import polars as pl
import torch

from app.recommend import RECENT_POPULARITY_WEIGHT, RECENT_YEARS
from pipeline.catalog import PROCESSED_DIR, film_popularity
from pipeline.evaluate import (
    BATCH_USERS,
    RESULTS_DIR,
    VARIETY_PROFILE_SIZE,
    EvalUser,
    Scorer,
    build_users,
    popularity_scorer,
    ranker_scorer,
)
from pipeline.load_db import RANKER_PATH
from pipeline.ranker import (
    NEW_FILM_CHECK_FROM,
    TrainingSettings,
    load_catalog,
    paired_difference,
    train_as_if_unrated,
)

FAVOURITE_COUNTS = (1, 3, VARIETY_PROFILE_SIZE)
# Weights tried for adding a film's 0-1 popularity, or its standing in its year, to its ranker
# score. Ranker scores for one person spread about 9 times as widely as popularity.
WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def popularity_from_views(views: np.ndarray, popularity: np.ndarray, fit_on: np.ndarray) -> np.ndarray:
    """Popularity worked out from page views alone, as pipeline.expand does for films without
    ratings: a line from log views to the popularity of the films in fit_on that have views."""
    log_views = np.log1p(views)
    rows = fit_on & (views > 0)
    slope, intercept = np.polyfit(log_views[rows], popularity[rows], 1)
    return np.clip(intercept + slope * log_views, 0.0, 1.0).astype(np.float32)


def standing_within_year(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Each film's percent rank among the films from its year, as PostgreSQL's percent_rank()
    gives it: the share of the others with a lower value, so 0 for the lowest and 1 for the
    highest."""
    standing = np.zeros(len(values), np.float32)
    for year in np.unique(years):
        rows = np.flatnonzero(years == year)
        if rows.size > 1:
            ordered = np.sort(values[rows])
            standing[rows] = np.searchsorted(ordered, values[rows], side="left") / (rows.size - 1)
    return standing


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def plus(scorer: Scorer, values: np.ndarray, weight: float) -> Scorer:
    return lambda profiles: scorer(profiles) + weight * values


def pool_picks(scorer: Scorer, users: list[EvalUser], size: int, pool: np.ndarray) -> np.ndarray:
    """The pool film the scorer ranks first for each user, leaving out films they'd rated."""
    picks = [np.zeros(0, dtype=np.int64)]
    for start in range(0, len(users), BATCH_USERS):
        batch = users[start : start + BATCH_USERS]
        scores = scorer([user.favourites[:size] for user in batch]).astype(np.float64)
        scores[:, ~pool] = -np.inf
        for row, user in enumerate(batch):
            scores[row, user.seen] = -np.inf
        picks.append(scores.argmax(axis=1))
    return np.concatenate(picks)


def variety(picks: np.ndarray, films: pl.DataFrame) -> dict:
    """How spread out the picks are: distinct films, and the most common one's share."""
    ids, counts = np.unique(picks, return_counts=True)
    years = films["year"].to_numpy()[picks]
    return {
        "distinct": int(ids.size),
        "most_common": films["title"][int(ids[counts.argmax()])],
        "most_common_share": float(counts.max() / picks.size),
        "by_year": {int(year): float((years == year).mean()) for year in np.unique(years)},
    }


def likes_pool_film(user: EvalUser, pool: np.ndarray) -> bool:
    return bool(pool[user.hidden].any())


def pool_pick_hits(scorer: Scorer, users: list[EvalUser], size: int, pool: np.ndarray) -> np.ndarray:
    """For each user who liked a film from the pool after their favourites: 1 if the pool film
    the scorer ranks first for them, leaving out films they'd rated, is one they liked."""
    users = [user for user in users if likes_pool_film(user, pool)]
    picks = pool_picks(scorer, users, size, pool)
    return np.array([float(np.isin(pick, user.hidden)) for pick, user in zip(picks, users)])


def random_pick_hits(users: list[EvalUser], pool: np.ndarray) -> np.ndarray:
    """For the same users: the chance that a random pool film they hadn't rated is one they
    liked."""
    return np.array([
        pool[user.hidden].sum() / (pool.sum() - pool[user.seen].sum())
        for user in users
        if likes_pool_film(user, pool)
    ])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = load_catalog()
    saved = np.load(RANKER_PATH)
    if not np.array_equal(saved["movie_ids"], catalog.movie_ids):
        raise SystemExit("ranker.npz is for another catalog: run pipeline.ranker first")
    trained = {name: saved[name] for name in saved.files if name != "movie_ids"}

    ratings = pl.read_parquet(PROCESSED_DIR / "ratings.parquet")
    as_new, pool = train_as_if_unrated(catalog, ratings, TrainingSettings(), device)
    ranker_alone = ranker_scorer(as_new)
    years = catalog.films["year"].fill_null(0).to_numpy()
    popularity = popularity_from_views(
        catalog.monthly_views, film_popularity(catalog.films), catalog.rated & ~pool
    )
    extras = {"standing": standing_within_year(popularity, years), "popularity": popularity}

    tuning_users = build_users(ratings.filter(pl.col("split") == "train"), catalog.movie_ids, 10**9)
    tuning = {
        kind: {weight: float(pool_pick_hits(plus(ranker_alone, values, weight), tuning_users,
                                            VARIETY_PROFILE_SIZE, pool).mean())
               for weight in WEIGHTS}
        for kind, values in extras.items()
    }
    weights = {kind: max(by_weight, key=lambda w: by_weight[w]) for kind, by_weight in tuning.items()}
    tuning_count = sum(likes_pool_film(user, pool) for user in tuning_users)
    print(f"tuning on {tuning_count:,} training users: {tuning}", flush=True)
    if weights["standing"] != RECENT_POPULARITY_WEIGHT:
        print(f"The app uses {RECENT_POPULARITY_WEIGHT}: set RECENT_POPULARITY_WEIGHT in "
              f"app/recommend.py to {weights['standing']} so it matches this report.", flush=True)

    # Every validation user, not just Stage 3's 5,000, since only some liked a film this new.
    users = build_users(ratings.filter(pl.col("split") == "val"), catalog.movie_ids, 10**9)
    app_method = f"Ranker plus {weights['standing']:g} × standing in its year (the app's)"
    popularity_method = f"Ranker plus {weights['popularity']:g} × popularity"
    methods = {
        app_method: plus(ranker_alone, extras["standing"], weights["standing"]),
        popularity_method: plus(ranker_alone, popularity, weights["popularity"]),
        "Ranker alone": ranker_alone,
        "Most Wikipedia page views": popularity_scorer(catalog.monthly_views),
        "Ranker trained with their ratings (for reference)": ranker_scorer(trained),
    }
    hits = {
        name: {size: pool_pick_hits(scorer, users, size, pool) for size in FAVOURITE_COUNTS}
        for name, scorer in methods.items()
    }
    hits["Random film"] = {size: random_pick_hits(users, pool) for size in FAVOURITE_COUNTS}
    differences = {
        against: {size: paired_difference(hits[app_method][size], hits[against][size])
                  for size in FAVOURITE_COUNTS}
        for against in ("Most Wikipedia page views", "Ranker alone")
    }

    # What the app would show: the model it uses, choosing among its own recent films.
    app_pool = years >= years.max() - RECENT_YEARS + 1
    app_popularity = film_popularity(catalog.films)
    trained_scorer = ranker_scorer(trained)
    shown = {
        app_method: plus(trained_scorer, standing_within_year(app_popularity, years),
                         weights["standing"]),
        popularity_method: plus(trained_scorer, app_popularity, weights["popularity"]),
        "Ranker alone": trained_scorer,
    }
    spread = {
        name: variety(pool_picks(scorer, users, VARIETY_PROFILE_SIZE, app_pool), catalog.films)
        for name, scorer in shown.items()
    }
    write_report(hits, differences, tuning, weights, tuning_count, int(pool.sum()),
                 hits[app_method][FAVOURITE_COUNTS[0]].size, spread, len(users),
                 int(years.max()) - RECENT_YEARS + 1, int(years.max()))


def write_report(
    hits: dict[str, dict[int, np.ndarray]],
    differences: dict[str, dict[int, tuple[float, float]]],
    tuning: dict[str, dict[float, float]],
    weights: dict[str, float],
    tuning_count: int,
    pool_size: int,
    user_count: int,
    spread: dict[str, dict],
    shown_users: int,
    first_year: int,
    last_year: int,
) -> None:
    def interval(values: np.ndarray) -> str:
        half_width = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
        return f"{values.mean():.1%} ± {half_width:.1%}"

    sizes = " | ".join(f"{size} favourite{'s' if size > 1 else ''}" for size in FAVOURITE_COUNTS)
    against_names = list(differences)
    years = range(first_year, last_year + 1)
    report = "\n".join([
        "# Stage 6 recent pick",
        "",
        (f"The app's last pick is the best film for you from {first_year} to {last_year}. "
         "Those films came out after MovieLens ends, so nobody's liking of them can be checked. "
         f"Instead the ranker was trained again as if the {pool_size} MovieLens films from "
         f"{NEW_FILM_CHECK_FROM} on had no ratings, the last three years MovieLens has, and "
         "each way of choosing below picked one of those films for each of the "
         f"{user_count:,} validation users who liked at least one of them after their "
         "favourites. A hit is a film they liked."),
        "",
        ("Popularity is worked out from page views, as for the catalog's films without ratings. "
         "A film's standing in its year is its percent rank on popularity among the catalog's "
         "films from the same year: 0 for the least viewed, 1 for the most."),
        "",
        f"| Choice from the films since {NEW_FILM_CHECK_FROM} | {sizes} |",
        "|---|" + "---:|" * len(FAVOURITE_COUNTS),
        *[f"| {name} | " + " | ".join(interval(by_size[size]) for size in FAVOURITE_COUNTS) + " |"
          for name, by_size in hits.items()],
        "",
        "The app's recent pick against two of the others, paired over the same users:",
        "",
        (f"| Favourites | Minus {lower_first(against_names[0])} | "
         f"Minus {lower_first(against_names[1])} |"),
        "|---:|---:|---:|",
        *[f"| {size} | " + " | ".join(
            f"{differences[name][size][0]:+.1%} ± {differences[name][size][1]:.1%}"
            for name in against_names) + " |"
          for size in FAVOURITE_COUNTS],
        "",
        "## What the app shows",
        "",
        ("Page views run far higher in a film's first months, so the newest films look the most "
         "popular. Adding popularity itself scores well above, where every film is a few years "
         "old, but in the app it hands most people films from this year. Standing in its year "
         "compares a film only with films as new as it. These are the recent picks the trained "
         f"ranker makes from the catalog's {first_year} to {last_year} films for the "
         f"{shown_users:,} validation users at {VARIETY_PROFILE_SIZE} favourites:"),
        "",
        "| Choice | Different films | Most common pick | "
        + " | ".join(f"From {year}" for year in years) + " |",
        "|---|---:|---|" + "---:|" * len(years),
        *[f"| {name} | {s['distinct']} | {s['most_common']} ({s['most_common_share']:.0%}) | "
          + " | ".join(f"{s['by_year'].get(year, 0.0):.0%}" for year in years) + " |"
          for name, s in spread.items()],
        "",
        ("Either way, many people get the same film. How widely liked a film is counts for a "
         "lot when the model knows nothing else about who liked it, and the most-viewed films "
         "are the likeliest to be liked. The ranker alone spreads its picks further but finds "
         "a liked film far less often, and a smaller weight gives up hits long before the most "
         "common pick goes to fewer people."),
        "",
        "## Choosing the weights",
        "",
        (f"Both weights were chosen at {VARIETY_PROFILE_SIZE} favourites on the "
         f"{tuning_count:,} training users who liked one of these films. The retrained model "
         "never saw those ratings."),
        "",
        "| Weight | Plus standing in its year | Plus popularity |",
        "|---:|---:|---:|",
        *[f"| {w:g} | {tuning['standing'][w]:.1%}{' (used)' if w == weights['standing'] else ''}"
          f" | {tuning['popularity'][w]:.1%}{' (best)' if w == weights['popularity'] else ''} |"
          for w in WEIGHTS],
        "",
        ("Ranges are 95% confidence intervals. The users here rated films from these years, "
         "so they're likelier than most to want a new film."),
        "",
    ])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "stage6_recent_pick.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "stage6_recent_pick.json").write_text(
        json.dumps({
            "weights": weights,
            "tuning": {kind: {str(w): share for w, share in by_weight.items()}
                       for kind, by_weight in tuning.items()},
            "variety": spread,
            "pool_size": pool_size,
            "users": user_count,
            "hit_rate": {name: {size: float(values.mean()) for size, values in by_size.items()}
                         for name, by_size in hits.items()},
            "difference": {
                name: {size: {"mean": gain, "half_width": half}
                       for size, (gain, half) in by_size.items()}
                for name, by_size in differences.items()
            },
        }, indent=2),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
