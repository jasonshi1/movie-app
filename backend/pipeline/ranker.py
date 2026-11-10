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

from app.ranker import TRAINED_MAX_FAVOURITES, taste_vectors
from app.recommend import POPULARITY_WEIGHT
from pipeline.catalog import PROCESSED_DIR, popularity_scores
from pipeline.embed import normalize
from pipeline.evaluate import (
    EVAL_USERS,
    PROFILE_SIZES,
    RESULTS_DIR,
    VARIETY_PROFILE_SIZE,
    EvalUser,
    Scorer,
    build_users,
    evaluate,
    popularity_scorer,
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
    patience: int = 3


class Ranker(nn.Module):
    def __init__(self, features: torch.Tensor, settings: TrainingSettings):
        super().__init__()
        films, feature_count = features.shape
        self.register_buffer("features", features)
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

    log_probs = torch.log_softmax(model(favourites, mask).masked_fill(seen, SEEN_PENALTY), dim=1)
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


def ranker_scorer(exported: dict[str, np.ndarray]) -> Scorer:
    """Score every film for each profile with the numpy code the app runs."""

    def score(profiles: list[np.ndarray]) -> np.ndarray:
        tastes = taste_vectors([exported["input_vectors"][profile] for profile in profiles], exported)
        return tastes @ exported["output_vectors"].T + exported["bias"]

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
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Train until the held-out users stop improving, and return the best model's export."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Ranker(torch.from_numpy(features), settings).to(device)
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
        score = mean_ndcg(ranker_scorer(exported), stopping_users)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-epochs", type=int, default=TrainingSettings.max_epochs)
    parser.add_argument("--train-users", type=int, help="train on at most this many users")
    arguments = parser.parse_args()
    settings = TrainingSettings(max_epochs=arguments.max_epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    movie_ids = films["movie_id"].to_numpy()
    raw_genome = np.load(PROCESSED_DIR / "genome.npy")
    text = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = popularity_scores(films["liked_count"].to_numpy())

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
        film_features(raw_genome, text), train_users, stopping_users, settings, device,
        log=lambda line: print(line, flush=True),
    )
    np.savez(RANKER_PATH, movie_ids=movie_ids, **exported)

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
