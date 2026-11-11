"""Score recommendation baselines on how well a few favourites predict what a user liked later.

Each test user's first 20 liked films (by date) are the possible favourites. Everything they
rated after the 20th is hidden. Methods see only the first N favourites and must rank the
hidden liked films highly. The hidden set is the same at every N, so scores are comparable.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from app.ranker import taste_vectors
from pipeline.catalog import LIKED_RATING, PROCESSED_DIR, film_popularity
from pipeline.embed import normalize

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
PROFILE_SIZES = (1, 3, 5, 10, 20)
TOP_K = 10
MIN_HIDDEN_LIKES = 5
EVAL_USERS = 5000
TUNING_USERS = 2000
SEED = 20261109
BATCH_USERS = 256
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
# Prefer the least popularity-driven weight that scores close to the best. The score alone
# rewards blockbusters, because people rate films they've heard of.
NEAR_BEST_SHARE = 0.97
VARIETY_PROFILE_SIZE = 5


@dataclass(frozen=True)
class EvalUser:
    favourites: np.ndarray  # catalog rows of the first liked films, oldest first
    seen: np.ndarray  # every catalog row rated up to the cutoff, never recommended
    hidden: np.ndarray  # catalog rows of films liked after the cutoff


@dataclass(frozen=True)
class Scores:
    ndcg: np.ndarray
    recall: np.ndarray
    first_picks: np.ndarray  # the catalog row each user would be offered first


def build_users(
    ratings: pl.DataFrame,
    movie_ids: np.ndarray,
    limit: int,
    seed: int = SEED,
    profile_size: int = max(PROFILE_SIZES),
    min_hidden: int = MIN_HIDDEN_LIKES,
) -> list[EvalUser]:
    """Turn raw ratings into evaluation users, in random order, up to limit."""
    rows_by_movie = {movie_id: row for row, movie_id in enumerate(movie_ids.tolist())}
    users = []
    ordered = ratings.sort("user_id", "timestamp", "movie_id")
    for (_,), group in ordered.group_by(["user_id"], maintain_order=True):
        rows = np.array([rows_by_movie[movie_id] for movie_id in group["movie_id"].to_list()])
        liked = group["rating"].to_numpy() >= LIKED_RATING
        liked_positions = np.flatnonzero(liked)
        if liked_positions.size < profile_size + min_hidden:
            continue

        cutoff = liked_positions[profile_size - 1]
        hidden_mask = liked.copy()
        hidden_mask[: cutoff + 1] = False
        hidden = rows[hidden_mask]
        if hidden.size < min_hidden:
            continue

        users.append(
            EvalUser(
                favourites=rows[liked_positions[:profile_size]],
                seen=rows[: cutoff + 1],
                hidden=hidden,
            )
        )

    order = np.random.default_rng(seed).permutation(len(users))
    return [users[index] for index in order[:limit]]


def ndcg_and_recall(ranked: np.ndarray, hidden: np.ndarray, k: int = TOP_K) -> tuple[float, float]:
    """Binary NDCG@k and recall@k, with recall capped at k so a perfect list scores 1."""
    hits = np.isin(ranked[:k], hidden)
    ideal = min(k, hidden.size)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float((hits * discounts).sum() / discounts[:ideal].sum()), float(hits.sum() / ideal)


Scorer = Callable[[list[np.ndarray]], np.ndarray]


def popularity_scorer(popularity: np.ndarray) -> Scorer:
    return lambda profiles: np.tile(popularity, (len(profiles), 1))


def genre_scorer(genres: np.ndarray, popularity: np.ndarray) -> Scorer:
    """Rank by cosine similarity of genres, using popularity only to break ties."""
    film_genres = normalize(genres)

    def score(profiles: list[np.ndarray]) -> np.ndarray:
        taste = normalize(np.stack([genres[profile].mean(axis=0) for profile in profiles]))
        return taste @ film_genres.T + 1e-6 * popularity

    return score


def similarity_scorer(vectors: np.ndarray, popularity: np.ndarray, alpha: float = 0.0) -> Scorer:
    """Cosine similarity to the favourites' mean vector, plus alpha times popularity."""

    def score(profiles: list[np.ndarray]) -> np.ndarray:
        taste = normalize(np.stack([vectors[profile].mean(axis=0) for profile in profiles]))
        return taste @ vectors.T + alpha * popularity

    return score


def ranker_scorer(exported: dict[str, np.ndarray]) -> Scorer:
    """Score every film for each profile with the numpy code the app runs."""

    def score(profiles: list[np.ndarray]) -> np.ndarray:
        tastes = taste_vectors([exported["input_vectors"][profile] for profile in profiles], exported)
        return tastes @ exported["output_vectors"].T + exported["bias"]

    return score


def evaluate(
    scorer: Scorer, users: list[EvalUser], profile_size: int, k: int = TOP_K
) -> Scores:
    """Score one method at one profile size, per user."""
    ndcgs, recalls, first_picks = [], [], []
    for start in range(0, len(users), BATCH_USERS):
        batch = users[start : start + BATCH_USERS]
        scores = scorer([user.favourites[:profile_size] for user in batch]).astype(np.float64)
        for row, user in enumerate(batch):
            scores[row, user.seen] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        ranked = np.take_along_axis(top, order, axis=1)
        for row, user in enumerate(batch):
            ndcg, recall = ndcg_and_recall(ranked[row], user.hidden, k)
            ndcgs.append(ndcg)
            recalls.append(recall)
            first_picks.append(ranked[row, 0])
    return Scores(np.array(ndcgs), np.array(recalls), np.array(first_picks))


def choose_alpha(sweep: list[dict[str, float]]) -> float:
    """Return the smallest popularity weight scoring within NEAR_BEST_SHARE of the best."""
    threshold = NEAR_BEST_SHARE * max(entry["ndcg"] for entry in sweep)
    return min(entry["alpha"] for entry in sweep if entry["ndcg"] >= threshold)


def sweep_alpha(
    vectors: np.ndarray, popularity: np.ndarray, users: list[EvalUser]
) -> list[dict[str, float]]:
    sweep = []
    for alpha in ALPHAS:
        scorer = similarity_scorer(vectors, popularity, alpha)
        sweep.append(
            {
                "alpha": alpha,
                "ndcg": float(np.mean([evaluate(scorer, users, n).ndcg.mean() for n in PROFILE_SIZES])),
                "distinct_first_picks": int(
                    np.unique(evaluate(scorer, users, VARIETY_PROFILE_SIZE).first_picks).size
                ),
            }
        )
    return sweep


def genre_matrix(genre_lists: list[list[str]]) -> np.ndarray:
    names = sorted({genre for genres in genre_lists for genre in genres})
    column = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(genre_lists), len(names)), dtype=np.float32)
    for row, genres in enumerate(genre_lists):
        matrix[row, [column[genre] for genre in genres]] = 1.0
    return matrix


def markdown_table(
    results: dict[str, dict[int, dict[str, float]]], metric: str, number_format: str = ".4f"
) -> str:
    header = "| Method | " + " | ".join(f"{n} favourite{'s' * (n > 1)}" for n in PROFILE_SIZES) + " |"
    divider = "|---|" + "---:|" * len(PROFILE_SIZES)
    rows = [
        f"| {method} | "
        + " | ".join(format(by_size[n][metric], number_format) for n in PROFILE_SIZES)
        + " |"
        for method, by_size in results.items()
    ]
    return "\n".join([header, divider, *rows])


def sweep_table(sweep: list[dict[str, float]]) -> str:
    rows = [
        f"| {entry['alpha']} | {entry['ndcg']:.4f} | {entry['distinct_first_picks']} |"
        for entry in sweep
    ]
    return "\n".join(
        ["| Popularity weight | Mean NDCG@10 | Distinct first picks |", "|---:|---:|---:|", *rows]
    )


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    movie_ids = films["movie_id"].to_numpy()
    genome = normalize(np.load(PROCESSED_DIR / "genome.npy"))
    text = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = film_popularity(films)
    genres = genre_matrix(films["genres"].to_list())

    ratings = pl.read_parquet(PROCESSED_DIR / "ratings.parquet")
    val_users = build_users(ratings.filter(pl.col("split") == "val"), movie_ids, EVAL_USERS)
    tuning_users = build_users(ratings.filter(pl.col("split") == "train"), movie_ids, TUNING_USERS)
    print(f"evaluation users: {len(val_users):,} | tuning users: {len(tuning_users):,}", flush=True)

    # Tuned on training users so the reported validation scores stay untouched by the choice.
    genome_sweep = sweep_alpha(genome, popularity, tuning_users)
    text_sweep = sweep_alpha(text, popularity, tuning_users)
    genome_alpha = choose_alpha(genome_sweep)
    text_alpha = choose_alpha(text_sweep)
    print(f"alpha: genome {genome_alpha} | text {text_alpha}", flush=True)

    methods: dict[str, Scorer] = {
        "Popularity": popularity_scorer(popularity),
        "Genre match": genre_scorer(genres, popularity),
        "Text embedding": similarity_scorer(text, popularity),
        "Tag Genome": similarity_scorer(genome, popularity),
        "Text embedding + popularity": similarity_scorer(text, popularity, text_alpha),
        "Tag Genome + popularity": similarity_scorer(genome, popularity, genome_alpha),
    }

    results: dict[str, dict[int, dict[str, float]]] = {}
    for name, scorer in methods.items():
        results[name] = {}
        for size in PROFILE_SIZES:
            scores = evaluate(scorer, val_users, size)
            results[name][size] = {
                "ndcg": float(scores.ndcg.mean()),
                "ndcg_ci95": float(1.96 * scores.ndcg.std(ddof=1) / np.sqrt(scores.ndcg.size)),
                "recall": float(scores.recall.mean()),
                "distinct_first_picks": int(np.unique(scores.first_picks).size),
            }
        print(f"finished {name}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    ci = max(entry["ndcg_ci95"] for by_size in results.values() for entry in by_size.values())
    report = "\n".join(
        [
            "# Stage 1 baseline results",
            "",
            (f"Validation users: {len(val_users):,}. Films: {len(movie_ids):,}. "
            f"Largest 95% confidence half-width on any NDCG score: {ci:.4f}."),
            "",
            (f"Popularity weight, chosen on training users as the smallest within "
            f"{NEAR_BEST_SHARE:.0%} of the best score: "
            f"Tag Genome {genome_alpha}, text embedding {text_alpha}."),
            "",
            f"## NDCG@{TOP_K}",
            "",
            markdown_table(results, "ndcg"),
            "",
            f"## Recall@{TOP_K}",
            "",
            markdown_table(results, "recall"),
            "",
            f"## Distinct first picks across {len(val_users):,} users",
            "",
            "The app's first pick is each method's top film, so this shows how personal it is.",
            "",
            markdown_table(results, "distinct_first_picks", "d"),
            "",
            f"## Tag Genome popularity weight, on {len(tuning_users):,} training users",
            "",
            sweep_table(genome_sweep),
            "",
            ("Unrated films count as misses, so these scores understate real-world quality. "
            "They are for comparing methods."),
            "",
        ]
    )
    (RESULTS_DIR / "stage1_baselines.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "stage1_baselines.json").write_text(
        json.dumps(
            {
                "users": len(val_users),
                "genome_alpha": genome_alpha,
                "text_alpha": text_alpha,
                "genome_sweep": genome_sweep,
                "text_sweep": text_sweep,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
