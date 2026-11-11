"""Train a PyTorch model that turns a few favourite films into a taste vector.

Each film gets an input vector (used when it's a favourite), an output vector (used when it's
a candidate) and a bias that learns how widely liked it is. Both vectors start from the film's
Tag Genome scores and text embedding, plus a learned per-film correction. A person's favourites
are averaged and passed through a small network to make their taste vector, so someone new
needs no retraining: a film's score is its output vector dotted with the taste vector, plus
its bias.

Training mirrors pipeline/evaluate.py. Each training user's first 20 liked films are the
possible favourites, the model sees a random number of them, and it must rank the films the
user liked after the 20th above every film they hadn't rated by then. Early stopping uses
held-out training users. Validation users are split off by user ID and are only scored once,
in the final report.
"""

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

from app.ranker import TRAINED_MAX_FAVOURITES
from app.recommend import POPULARITY_WEIGHT
from pipeline import themes
from pipeline.catalog import PROCESSED_DIR, film_popularity
from pipeline.embed import normalize
from pipeline.evaluate import (
    BATCH_USERS,
    EVAL_USERS,
    PROFILE_SIZES,
    RESULTS_DIR,
    VARIETY_PROFILE_SIZE,
    EvalUser,
    Scorer,
    build_users,
    evaluate,
    popularity_scorer,
    ranker_scorer,
    similarity_scorer,
)
from pipeline.load_db import RANKER_PATH

MAX_FAVOURITES = TRAINED_MAX_FAVOURITES
assert MAX_FAVOURITES == max(PROFILE_SIZES), "training must cover every evaluated profile size"
STOPPING_USERS = 5000
SEED = 7
# Scores for films a person has already rated are pushed this far down before the softmax.
# A finite value keeps log-probabilities finite, so hidden films never multiply an infinity.
SEEN_PENALTY = -1e9


@dataclass(frozen=True)
class TrainingSettings:
    dim: int = 128
    hidden: int = 512
    dropout: float = 0.2
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 40
    # The held-out score dips for a few epochs at a time, more so with the Stage 6 catalog,
    # where training stopped at epoch 2 of a run that kept improving until about epoch 14.
    patience: int = 5


class Ranker(nn.Module):
    def __init__(
        self, features: torch.Tensor, settings: TrainingSettings, rated: torch.Tensor | None = None
    ):
        super().__init__()
        films, feature_count = features.shape
        self.register_buffer("features", features)
        # Films with no ratings never appear in training: always ranking them below what people
        # liked would teach the model they're bad. Their bias is predicted afterwards.
        self.register_buffer(
            "rated", rated if rated is not None else torch.ones(films, dtype=torch.bool)
        )
        self.input_projection = nn.Linear(feature_count, settings.dim)
        self.output_projection = nn.Linear(feature_count, settings.dim)
        # Corrections start at zero, so training begins from what the features alone predict.
        self.input_ids = nn.Embedding(films, settings.dim)
        self.output_ids = nn.Embedding(films, settings.dim)
        nn.init.zeros_(self.input_ids.weight)
        nn.init.zeros_(self.output_ids.weight)
        self.bias = nn.Parameter(torch.zeros(films))
        self.encoder = nn.Sequential(
            nn.Linear(settings.dim + 1, settings.hidden),
            nn.ReLU(),
            nn.Dropout(settings.dropout),
            nn.Linear(settings.hidden, settings.dim),
        )

    def input_vectors(self, rows: torch.Tensor) -> torch.Tensor:
        return self.input_projection(self.features[rows]) + self.input_ids(rows)

    def output_vectors(self) -> torch.Tensor:
        return self.output_projection(self.features) + self.output_ids.weight

    def taste(self, favourites: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average each person's favourites, then let the encoder adjust the average.

        favourites holds catalog rows, padded to the same length; mask marks the real ones.
        """
        weights = mask.unsqueeze(-1).to(self.features.dtype)
        counts = weights.sum(dim=1)
        means = (self.input_vectors(favourites) * weights).sum(dim=1) / counts
        return means + self.encoder(torch.cat([means, counts.log()], dim=1))

    def forward(self, favourites: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.taste(favourites, mask) @ self.output_vectors().T + self.bias


def film_features(genome: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Tag Genome scores and text embeddings, each column scaled to mean 0 and spread 1."""
    features = np.concatenate([genome, text], axis=1).astype(np.float32)
    spread = features.std(axis=0)
    return (features - features.mean(axis=0)) / np.where(spread == 0, 1, spread)


def _segments(
    arrays: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row and column indices that mark every array's entries in a batch-by-films matrix."""
    rows = np.repeat(np.arange(len(arrays)), [array.size for array in arrays])
    columns = np.concatenate(arrays)
    return torch.from_numpy(rows).to(device), torch.from_numpy(columns).to(device)


def _loss(
    model: Ranker, users: list[EvalUser], rng: np.random.Generator, device: torch.device
) -> torch.Tensor:
    """Average negative log-likelihood of each user's hidden films, with a random profile size."""
    films = model.features.shape[0]
    favourites = torch.from_numpy(np.stack([user.favourites for user in users])).to(device)
    sizes = rng.integers(1, MAX_FAVOURITES + 1, size=len(users))
    mask = torch.from_numpy(np.arange(MAX_FAVOURITES)[None, :] < sizes[:, None]).to(device)

    seen = torch.zeros(len(users), films, dtype=torch.bool, device=device)
    seen[_segments([user.seen for user in users], device)] = True
    hidden = torch.zeros(len(users), films, device=device)
    hidden[_segments([user.hidden for user in users], device)] = 1.0

    left_out = seen | ~model.rated
    log_probs = torch.log_softmax(model(favourites, mask).masked_fill(left_out, SEEN_PENALTY), dim=1)
    return (-(log_probs * hidden).sum(dim=1) / hidden.sum(dim=1)).mean()


def export(model: Ranker) -> dict[str, np.ndarray]:
    """The trained film vectors and encoder weights, as numpy arrays for the app."""
    model.eval()
    with torch.no_grad():
        rows = torch.arange(model.features.shape[0], device=model.features.device)
        arrays = {
            "input_vectors": model.input_vectors(rows),
            "output_vectors": model.output_vectors(),
            "bias": model.bias,
            "w1": model.encoder[0].weight,
            "b1": model.encoder[0].bias,
            "w2": model.encoder[3].weight,
            "b2": model.encoder[3].bias,
        }
        return {name: value.detach().cpu().numpy().astype(np.float32) for name, value in arrays.items()}


BIAS_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def fit_bias_model(x: np.ndarray, bias: np.ndarray, seed: int = SEED) -> tuple[tuple, float]:
    """Ridge regression from a film's predictors to its learned bias, with the strength chosen
    on a tenth of the films held out. Returns the model refitted on every film, and the
    held-out R²."""
    order = np.random.default_rng(seed).permutation(len(bias))
    held, fitted = order[: len(bias) // 10], order[len(bias) // 10 :]

    def r_squared(model: tuple, rows: np.ndarray) -> float:
        residual = bias[rows] - predict_bias(model, x[rows])
        return float(1 - residual.var() / bias[rows].var())

    scores = {alpha: r_squared(themes.fit_ridge(x[fitted], bias[fitted, None], alpha), held)
              for alpha in BIAS_ALPHAS}
    alpha = max(scores, key=scores.get)
    return themes.fit_ridge(x, bias[:, None], alpha), scores[alpha]


def predict_bias(model: tuple, x: np.ndarray) -> np.ndarray:
    weights, intercept = model
    return (x @ weights + intercept)[:, 0]


def with_unrated_bias(
    exported: dict[str, np.ndarray], rated: np.ndarray, bias_inputs: np.ndarray
) -> tuple[dict[str, np.ndarray], float | None]:
    """Give films without ratings a bias predicted from their page views and features, by a
    model fitted on the rated films' learned biases. Returns the held-out R² too."""
    if rated.all():
        return exported, None
    model, r2 = fit_bias_model(bias_inputs[rated], exported["bias"][rated])
    bias = exported["bias"].copy()
    bias[~rated] = predict_bias(model, bias_inputs[~rated])
    return exported | {"bias": bias.astype(np.float32)}, r2


def bias_inputs(features: np.ndarray, monthly_views: np.ndarray) -> np.ndarray:
    """What a film's bias is predicted from: its page views on a log scale, and its features."""
    views = np.log1p(monthly_views.astype(np.float32))[:, None]
    return np.concatenate([(views - views.mean()) / views.std(), features], axis=1)


def leaving_out(scorer: Scorer, films: np.ndarray) -> Scorer:
    """The same scores, with the given films never ranked."""
    if not films.any():
        return scorer

    def score(profiles: list[np.ndarray]) -> np.ndarray:
        scores = scorer(profiles).astype(np.float64)
        scores[:, films] = -np.inf
        return scores

    return score


def mean_ndcg(scorer: Scorer, users: list[EvalUser]) -> float:
    return float(np.mean([evaluate(scorer, users, size).ndcg.mean() for size in PROFILE_SIZES]))


def fit(
    features: np.ndarray,
    users: list[EvalUser],
    stopping_users: list[EvalUser],
    settings: TrainingSettings,
    device: torch.device,
    seed: int = SEED,
    log=print,
    rated: np.ndarray | None = None,
    unrated_bias_inputs: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Train until the held-out users stop improving, and return the best model's export.

    Films where rated is False are left out of training and of early stopping, which nobody's
    ratings can judge. The best model's biases for them are then predicted from
    unrated_bias_inputs.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rated = np.ones(features.shape[0], dtype=bool) if rated is None else rated
    model = Ranker(torch.from_numpy(features), settings, torch.from_numpy(rated)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    best, best_score, history, stale = None, -np.inf, [], 0
    for epoch in range(1, settings.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        order = rng.permutation(len(users))
        for start in range(0, len(users), settings.batch_size):
            batch = [users[index] for index in order[start : start + settings.batch_size]]
            loss = _loss(model, batch, rng, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        exported = export(model)
        score = mean_ndcg(leaving_out(ranker_scorer(exported), ~rated), stopping_users)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "stopping_ndcg": score,
                "seconds": time.perf_counter() - started,
            }
        )
        log(
            f"epoch {epoch}: loss {history[-1]['loss']:.4f} | held-out NDCG@10 {score:.4f} | "
            f"{history[-1]['seconds']:.0f}s"
        )
        if score > best_score:
            best, best_score, stale = exported, score, 0
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if not rated.all():
        best, _ = with_unrated_bias(best, rated, unrated_bias_inputs)
    return best, history


def summarize(scorer: Scorer, users: list[EvalUser]) -> dict[int, dict]:
    summary = {}
    for size in PROFILE_SIZES:
        scores = evaluate(scorer, users, size)
        summary[size] = {
            "ndcg": float(scores.ndcg.mean()),
            "ndcg_ci95": float(1.96 * scores.ndcg.std(ddof=1) / np.sqrt(scores.ndcg.size)),
            "recall": float(scores.recall.mean()),
            "distinct_first_picks": int(np.unique(scores.first_picks).size),
            "per_user_ndcg": scores.ndcg,
        }
    return summary


def paired_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Mean per-user difference and its 95% confidence half-width. Both runs score the same users."""
    difference = first - second
    return float(difference.mean()), float(1.96 * difference.std(ddof=1) / np.sqrt(difference.size))


def table(summaries: dict[str, dict[int, dict]], metric: str, number_format: str) -> str:
    header = "| Method | " + " | ".join(f"{n} favourite{'s' * (n > 1)}" for n in PROFILE_SIZES) + " |"
    rows = [
        f"| {name} | " + " | ".join(format(summary[n][metric], number_format) for n in PROFILE_SIZES) + " |"
        for name, summary in summaries.items()
    ]
    return "\n".join([header, "|---|" + "---:|" * len(PROFILE_SIZES), *rows])


# The new-film check retrains as if MovieLens films from this year on had no ratings.
NEW_FILM_CHECK_FROM = 2021
CHECK_TOP = 100
SHOWN_PICKS = 3


def hidden_ranks(
    scorer: Scorer, users: list[EvalUser], size: int, of_interest: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each film of interest a user liked later: its position among that user's candidates
    as a share (0 is first), and whether it's in their top CHECK_TOP."""
    shares, hits = [], []
    for start in range(0, len(users), BATCH_USERS):
        batch = users[start : start + BATCH_USERS]
        scores = scorer([user.favourites[:size] for user in batch]).astype(np.float64)
        for row, user in enumerate(batch):
            scores[row, user.seen] = -np.inf
            candidates = np.isfinite(scores[row]).sum()
            for target in user.hidden[of_interest[user.hidden]]:
                above = int((scores[row] > scores[row, target]).sum())
                shares.append(above / candidates)
                hits.append(above < CHECK_TOP)
    return np.array(shares), np.array(hits)


def share_shown(scorer: Scorer, users: list[EvalUser], size: int, films: np.ndarray) -> float:
    """The share of users who get at least one of the given films among their picks."""
    shown = 0
    for start in range(0, len(users), BATCH_USERS):
        batch = users[start : start + BATCH_USERS]
        scores = scorer([user.favourites[:size] for user in batch]).astype(np.float64)
        for row, user in enumerate(batch):
            scores[row, user.seen] = -np.inf
        top = np.argpartition(-scores, SHOWN_PICKS, axis=1)[:, :SHOWN_PICKS]
        shown += int(films[top].any(axis=1).sum())
    return shown / len(users)


def unrated_report(
    films: pl.DataFrame,
    ratings: pl.DataFrame,
    movie_ids: np.ndarray,
    features: np.ndarray,
    rated: np.ndarray,
    unrated_inputs: np.ndarray,
    exported: dict[str, np.ndarray],
    val_users: list[EvalUser],
    settings: TrainingSettings,
    device: torch.device,
    run_check: bool,
) -> list[str]:
    """Report lines on the films without ratings: how their bias was predicted, how often
    they're picked, and how well films are ranked when they have no ratings yet."""
    size = VARIETY_PROFILE_SIZE
    _, r2 = with_unrated_bias(exported, rated, unrated_inputs)
    shown = share_shown(ranker_scorer(exported), val_users, size, ~rated)
    lines = [
        "## Films without ratings",
        "",
        (f"{(~rated).sum():,} films from Wikidata, released after MovieLens ends, have no "
         f"ratings. They're left out of training, and their bias is predicted from Wikipedia "
         f"page views and their features by a ridge regression fitted on the rated films' "
         f"learned biases (R² {r2:.2f} on rated films held out). With {size} favourites, "
         f"{shown:.1%} of validation users get at least one of them among their "
         f"{SHOWN_PICKS} picks. Validation can't reward these picks, since nobody rated them."),
        "",
    ]
    if not run_check:
        return lines

    years = films["year"].fill_null(0).to_numpy()
    pretend = rated & (films["source"] == "movielens").to_numpy() & (years >= NEW_FILM_CHECK_FROM)
    train = build_users(
        ratings.filter(pl.col("split") == "train", ~pl.col("movie_id").is_in(movie_ids[pretend])),
        movie_ids, 10**9, seed=SEED,
    )
    print(f"new-film check: retraining without ratings for {pretend.sum():,} films from "
          f"{NEW_FILM_CHECK_FROM} on", flush=True)
    as_new, _ = fit(
        features, train[STOPPING_USERS:], train[:STOPPING_USERS], settings, device,
        log=lambda line: print(f"new-film check {line}", flush=True),
        rated=rated & ~pretend, unrated_bias_inputs=unrated_inputs,
    )
    warm_shares, warm_hits = hidden_ranks(ranker_scorer(exported), val_users, size, pretend)
    new_shares, new_hits = hidden_ranks(ranker_scorer(as_new), val_users, size, pretend)
    warm_ndcg = evaluate(ranker_scorer(exported), val_users, size).ndcg.mean()
    new_ndcg = evaluate(ranker_scorer(as_new), val_users, size).ndcg.mean()
    candidates = len(movie_ids)
    return lines + [
        (f"To estimate how well films without ratings are ranked, the model was trained again "
         f"as if the {pretend.sum():,} MovieLens films from {NEW_FILM_CHECK_FROM} on had no "
         f"ratings, then compared with the full model on the {warm_hits.size:,} times a "
         f"validation user liked one of them after their {size} favourites:"),
        "",
        f"| Model | Liked film in the top {CHECK_TOP} | Median position | NDCG@10, all films |",
        "|---|---:|---:|---:|",
        (f"| Trained with their ratings | {warm_hits.mean():.1%} | "
         f"top {np.median(warm_shares):.1%} | {warm_ndcg:.4f} |"),
        (f"| As if they had no ratings | {new_hits.mean():.1%} | "
         f"top {np.median(new_shares):.1%} | {new_ndcg:.4f} |"),
        f"| Random order | {CHECK_TOP / candidates:.1%} | top 50% | |",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-epochs", type=int, default=TrainingSettings.max_epochs)
    parser.add_argument("--train-users", type=int, help="train on at most this many users")
    parser.add_argument("--skip-new-film-check", action="store_true",
                        help="don't retrain to test how well films without ratings are ranked")
    arguments = parser.parse_args()
    settings = TrainingSettings(max_epochs=arguments.max_epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    movie_ids = films["movie_id"].to_numpy()
    raw_genome = np.load(PROCESSED_DIR / "genome.npy")
    text = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = film_popularity(films)
    expanded = "source" in films.columns
    rated = (films["source"] != "wikidata").to_numpy() if expanded else np.ones(len(films), bool)
    monthly_views = (
        films["monthly_views"].to_numpy() if expanded else np.zeros(len(films), np.float32)
    )
    features = film_features(raw_genome, text)
    unrated_inputs = bias_inputs(features, monthly_views)

    ratings = pl.read_parquet(PROCESSED_DIR / "ratings.parquet")
    started = time.perf_counter()
    train_users = build_users(ratings.filter(pl.col("split") == "train"), movie_ids, 10**9, seed=SEED)
    stopping_users, train_users = train_users[:STOPPING_USERS], train_users[STOPPING_USERS:]
    train_users = train_users[: arguments.train_users]
    val_users = build_users(ratings.filter(pl.col("split") == "val"), movie_ids, EVAL_USERS)
    print(
        f"users: {len(train_users):,} training, {len(stopping_users):,} for early stopping, "
        f"{len(val_users):,} validation | built in {time.perf_counter() - started:.0f}s | "
        f"device: {device}",
        flush=True,
    )

    exported, history = fit(
        features, train_users, stopping_users, settings, device,
        log=lambda line: print(line, flush=True), rated=rated, unrated_bias_inputs=unrated_inputs,
    )
    np.savez(RANKER_PATH, movie_ids=movie_ids, **exported)
    unrated_section = (
        unrated_report(films, ratings, movie_ids, features, rated, unrated_inputs, exported,
                       val_users, settings, device, run_check=not arguments.skip_new_film_check)
        if expanded else []
    )

    baseline_name = "Tag Genome + popularity (Stage 1)"
    summaries = {
        "Popularity": summarize(popularity_scorer(popularity), val_users),
        baseline_name: summarize(
            similarity_scorer(normalize(raw_genome), popularity, POPULARITY_WEIGHT), val_users
        ),
        "Ranker": summarize(ranker_scorer(exported), val_users),
    }
    differences = {
        size: paired_difference(
            summaries["Ranker"][size]["per_user_ndcg"], summaries[baseline_name][size]["per_user_ndcg"]
        )
        for size in PROFILE_SIZES
    }
    gain, half_width = differences[VARIETY_PROFILE_SIZE]
    verdict = (
        "beats" if gain - half_width > 0 else "does not beat" if gain + half_width < 0 else "ties"
    )
    best_epoch = max(history, key=lambda entry: entry["stopping_ndcg"])["epoch"]

    report = "\n".join(
        [
            "# Stage 3 ranker results",
            "",
            (f"At {VARIETY_PROFILE_SIZE} favourites the ranker {verdict} the best Stage 1 method: "
             f"NDCG@10 {summaries['Ranker'][VARIETY_PROFILE_SIZE]['ndcg']:.4f} against "
             f"{summaries[baseline_name][VARIETY_PROFILE_SIZE]['ndcg']:.4f}, a difference of "
             f"{gain:+.4f} ± {half_width:.4f} (95% confidence, paired over the same users)."),
            "",
            (f"Validation users: {len(val_users):,}, the same ones as Stage 1, scored once after "
             f"training. Training users: {len(train_users):,}. Early stopping used "
             f"{len(stopping_users):,} other training users and kept epoch {best_epoch}."),
            "",
            "## NDCG@10",
            "",
            table(summaries, "ndcg", ".4f"),
            "",
            "## Ranker minus Stage 1, per user",
            "",
            "| Favourites | Difference in NDCG@10 | 95% confidence |",
            "|---:|---:|---:|",
            *[f"| {size} | {differences[size][0]:+.4f} | ± {differences[size][1]:.4f} |"
              for size in PROFILE_SIZES],
            "",
            "## Recall@10",
            "",
            table(summaries, "recall", ".4f"),
            "",
            f"## Distinct first picks across {len(val_users):,} users",
            "",
            table(summaries, "distinct_first_picks", "d"),
            "",
            *unrated_section,
            "## Training",
            "",
            (f"Settings: {settings}. Device: {device}."),
            "",
            "| Epoch | Training loss | Held-out NDCG@10 | Seconds |",
            "|---:|---:|---:|---:|",
            *[f"| {entry['epoch']} | {entry['loss']:.4f} | {entry['stopping_ndcg']:.4f} | "
              f"{entry['seconds']:.0f} |" for entry in history],
            "",
            ("Unrated films count as misses, so these scores understate real-world quality. "
             "They are for comparing methods."),
            "",
        ]
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "stage3_ranker.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "stage3_ranker.json").write_text(
        json.dumps(
            {
                "settings": settings.__dict__,
                "device": str(device),
                "users": {"train": len(train_users), "stopping": len(stopping_users),
                          "validation": len(val_users)},
                "history": history,
                "results": {
                    name: {
                        size: {key: value for key, value in entry.items() if key != "per_user_ndcg"}
                        for size, entry in summary.items()
                    }
                    for name, summary in summaries.items()
                },
                "ranker_minus_stage1": {
                    size: {"ndcg": diff, "ci95": ci} for size, (diff, ci) in differences.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
