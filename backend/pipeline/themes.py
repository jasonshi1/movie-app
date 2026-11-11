"""Predict Tag Genome theme scores for films that have none, from their descriptions.

Picks explain themselves with Tag Genome themes, Claude's search ranks by them, and the ranker
reads them, so every film needs all 1,084 scores. Only the Stage 1 films have real ones. This
fits a ridge regression from a film's text embedding (title, year, genres, directors and
Wikipedia introduction, without themes), genres and year to its theme scores, on the films that
have them.

Regression pulls every prediction toward the average, so on its own it gives every film the
same broadly common themes, such as "original", at the top. Each theme's predictions are then
rescaled to its real spread: a film predicted higher than 90% of films on a theme gets the
score that 90% of real films fall below. That brings out what's distinctive about each film.
The report in results/stage6_themes.md compares both against simple guesses on films held out
of fitting.
"""

import json

import numpy as np

from pipeline.evaluate import RESULTS_DIR

GENRE_ORDER = (
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
)
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
HELD_OUT_SHARE = 0.1
TOP_THEMES = 10
SEED = 20261110


def features(plain: np.ndarray, genres: list[list[str]], years: list[int | None]) -> np.ndarray:
    """Text embedding, genre indicators and scaled year, one row per film."""
    genre_columns = np.array(
        [[genre in film_genres for genre in GENRE_ORDER] for film_genres in genres], dtype=np.float32
    )
    year_column = np.array(
        [[(year - 1990) / 30 if year else 0.0] for year in years], dtype=np.float32
    )
    return np.concatenate([plain, genre_columns, year_column], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights and intercept minimising squared error plus alpha times the squared weights."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centred = x - x_mean
    weights = np.linalg.solve(
        centred.T @ centred + alpha * np.eye(x.shape[1], dtype=np.float64), centred.T @ (y - y_mean)
    )
    return weights.astype(np.float32), (y_mean - x_mean @ weights).astype(np.float32)


def predict(model: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    weights, intercept = model
    return np.clip(x @ weights + intercept, 0.0, 1.0)


def agreement(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """How close predictions are: cosine similarity per film, the share of each film's top
    themes found, and the correlation per theme across films."""
    cosine = (predicted * actual).sum(axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1)
    )
    top_predicted = np.argsort(-predicted, axis=1)[:, :TOP_THEMES]
    top_actual = np.argsort(-actual, axis=1)[:, :TOP_THEMES]
    overlap = [len(set(p) & set(a)) / TOP_THEMES for p, a in zip(top_predicted, top_actual)]
    centred_p = predicted - predicted.mean(axis=0)
    centred_a = actual - actual.mean(axis=0)
    # A theme predicted the same for every film has no correlation; it's left out of the mean.
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = (centred_p * centred_a).sum(axis=0) / np.sqrt(
            (centred_p**2).sum(axis=0) * (centred_a**2).sum(axis=0)
        )
    return {
        "cosine": float(cosine.mean()),
        "top_overlap": float(np.mean(overlap)),
        "theme_correlation": float(np.nanmean(correlation)),
    }


def rescale(fitted_predictions: np.ndarray, real: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Map each theme's predictions onto the real scores' spread, by their rank among the
    predictions for the fitted films."""
    fitted_sorted = np.sort(fitted_predictions, axis=0)
    real_sorted = np.sort(real, axis=0)
    last = len(real_sorted) - 1
    out = np.empty_like(predictions)
    for theme in range(real.shape[1]):
        share = np.searchsorted(fitted_sorted[:, theme], predictions[:, theme]) / len(fitted_sorted)
        out[:, theme] = real_sorted[np.clip((share * last).astype(int), 0, last), theme]
    return out


def most_common_top_theme(scores: np.ndarray, top: int = 5) -> float:
    """The share of films whose top themes include the single most frequent one. Real Tag
    Genome scores give about a third; a prediction that's the same for every film gives 100%."""
    counts = np.bincount(np.argsort(-scores, axis=1)[:, :top].ravel(), minlength=scores.shape[1])
    return float(counts.max() / len(scores))


def genre_means(x_genres: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Guess each film's themes as the average of fitted films sharing its exact genres, or the
    overall average when none do."""
    keys = [tuple(row) for row in x_genres.astype(int)]
    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = {}
    for row in rows:
        sums[keys[row]] = sums.get(keys[row], 0) + y[row]
        counts[keys[row]] = counts.get(keys[row], 0) + 1
    overall = y[rows].mean(axis=0)
    return np.stack([sums[key] / counts[key] if key in sums else overall for key in keys])


def predict_for_catalog(
    plain: np.ndarray,
    genres: list[list[str]],
    years: list[int | None],
    core_genome: np.ndarray,
    is_core: np.ndarray,
    tags: list[str],
) -> np.ndarray:
    """Predicted theme scores for every film. The core films come first in the catalog, in the
    same order as core_genome. Writes the held-out report."""
    x = features(plain, genres, years)
    core_rows = np.flatnonzero(is_core)
    if core_rows.size != core_genome.shape[0] or not (core_rows == np.arange(core_rows.size)).all():
        raise ValueError("the core films must come first, in Tag Genome order")
    y = core_genome

    order = np.random.default_rng(SEED).permutation(core_rows.size)
    held = order[: int(core_rows.size * HELD_OUT_SHARE)]
    tune = order[held.size : 2 * held.size]
    fit_rows = order[2 * held.size :]

    tuning = {alpha: agreement(predict(fit_ridge(x[fit_rows], y[fit_rows], alpha), x[tune]), y[tune])
              for alpha in ALPHAS}
    alpha = max(tuning, key=lambda a: tuning[a]["cosine"])
    train_rows = np.concatenate([fit_rows, tune])
    model = fit_ridge(x[train_rows], y[train_rows], alpha)
    ridge_held = predict(model, x[held])
    rescaled_held = rescale(predict(model, x[train_rows]), y[train_rows], ridge_held)
    guesses = {
        "Ridge regression, rescaled to each theme's real spread (used)": rescaled_held,
        "Ridge regression alone": ridge_held,
        "Average of films with the same genres": genre_means(x[:, plain.shape[1]:-1], y,
                                                             train_rows)[held],
        "Average of all films": np.tile(y[train_rows].mean(axis=0), (held.size, 1)),
        "Real scores": y[held],
    }
    held_out = {
        name: agreement(guess, y[held]) | {"most_common_top_theme": most_common_top_theme(guess)}
        for name, guess in guesses.items()
    }

    model = fit_ridge(x[core_rows], y, alpha)
    predicted = rescale(predict(model, x[core_rows]), y, predict(model, x))
    write_report(held_out, tuning, alpha, held.size, core_rows.size)
    names = np.array(tags)
    for row in np.flatnonzero(~is_core)[:: max(1, (~is_core).sum() // 5)][:5]:
        print(f"themes for row {row}: {', '.join(names[np.argsort(-predicted[row])[:8]])}")
    return predicted


def write_report(held_out: dict, tuning: dict, alpha: float, held: int, fitted: int) -> None:
    rows = [f"| {name} | {s['top_overlap']:.1%} | {s['theme_correlation']:.3f} | "
            f"{s['cosine']:.3f} | {s['most_common_top_theme']:.0%} |"
            for name, s in held_out.items()]
    report = "\n".join([
        "# Stage 6 predicted themes",
        "",
        (f"Films added after the Tag Genome have no theme scores, so they're predicted from each "
         f"film's description, genres and year. These are the results on {held:,} films with "
         f"real scores that were held out of fitting. The final model is then fitted on all "
         f"{fitted:,} of them."),
        "",
        ("- **Top 10 themes found**: the share of a film's real top 10 themes among its predicted "
         "top 10. This is what picks show as shared themes."),
        ("- **Correlation per theme**: how well predictions order films on each theme, averaged "
         "over themes. The ranker reads themes this way."),
        "- **Cosine similarity**: how close the whole set of scores is. Claude's search uses this.",
        ("- **Most common top theme**: the share of films whose top 5 includes the most frequent "
         "theme. High means every film gets the same themes."),
        "",
        "| Method | Top 10 themes found | Correlation per theme | Cosine similarity | Most common top theme |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        ("Rescaling lowers the cosine similarity a little, because it spreads scores out, but "
         "finds more of each film's real themes and stops every film sharing the same ones."),
        "",
        f"Ridge strength {alpha}, chosen on a separate tuning set of the same size:",
        "",
        "| Strength | Cosine similarity | Top 10 themes found |",
        "|---:|---:|---:|",
        *[f"| {a} | {s['cosine']:.3f} | {s['top_overlap']:.1%} |" for a, s in tuning.items()],
        "",
    ])
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "stage6_themes.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "stage6_themes.json").write_text(
        json.dumps({"alpha": alpha, "held_out": held_out, "tuning": tuning}, indent=2),
        encoding="utf-8",
    )
    print(report)
