"""Build the film catalog, tag matrix and evaluation ratings from the raw archives."""

import io
import json
import re
import zipfile

import numpy as np
import polars as pl

from pipeline.download import DATA_DIR, RAW_DIR

PROCESSED_DIR = DATA_DIR / "processed"
GENOME_ROOT = "movie_dataset_public_final/"
LIKED_RATING = 4.0
MAX_MISSING_SCORES = 10
MAX_UNMATCHED_SHARE = 0.05

_YEAR = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")
_ALTERNATE_TITLE = re.compile(r"^(.+?)\s+\([^()]*\)$")
_TRAILING_ARTICLE = re.compile(
    r"^(?P<base>.+), (?P<article>The|A|An|Le|La|Les|L'|El|Los|Las|Il|Der|Die|Das)$"
)


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split MovieLens' "Matrix, The (1999)" into ("The Matrix", 1999)."""
    match = _YEAR.match(raw)
    title, year = (match[1], int(match[2])) if match else (raw.strip(), None)

    # Foreign releases carry an alternate title: "Amelie (Fabuleux destin d'Amélie Poulain, Le)".
    # Subtitles written as "Birdman: Or (The Unexpected...)" are part of the real title.
    alternate = _ALTERNATE_TITLE.match(title)
    if alternate and not alternate[1].endswith((" Or", ":")):
        title = alternate[1]

    article = _TRAILING_ARTICLE.match(title)
    if article:
        separator = "" if article["article"].endswith("'") else " "
        title = f"{article['article']}{separator}{article['base']}"
    return title, year


def popularity_scores(liked_counts: np.ndarray) -> np.ndarray:
    """Scale each film's liked count to 0-1 on a log scale."""
    log_liked = np.log1p(liked_counts.astype(np.float32))
    return log_liked / log_liked.max()


def film_popularity(films: pl.DataFrame) -> np.ndarray:
    """Each film's 0-1 popularity. After Stage 6 the catalog stores it, because films without
    ratings get theirs from Wikipedia page views; before, it comes from the liked counts."""
    if "popularity" in films.columns:
        return films["popularity"].to_numpy().astype(np.float32)
    return popularity_scores(films["liked_count"].to_numpy())


def split_names(value: str | None) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()] if value else []


def user_split(user_id_column: pl.Expr) -> pl.Expr:
    """Assign each user to one split permanently, so no user is in both training and testing."""
    remainder = user_id_column % 10
    return (
        pl.when(remainder == 0).then(pl.lit("test"))
        .when(remainder == 1).then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )


def load_genome(archive: zipfile.ZipFile) -> tuple[np.ndarray, list[int], list[str]]:
    scores = pl.read_csv(io.BytesIO(archive.read(GENOME_ROOT + "scores/tagdl.csv")))
    movie_ids = scores["item_id"].unique().sort().to_list()
    tags = scores["tag"].unique().sort().to_list()

    row = pl.DataFrame({"item_id": movie_ids, "row": range(len(movie_ids))})
    column = pl.DataFrame({"tag": tags, "column": range(len(tags))})
    positions = scores.join(row, on="item_id").join(column, on="tag")

    matrix = np.full((len(movie_ids), len(tags)), np.nan, dtype=np.float32)
    matrix[positions["row"].to_numpy(), positions["column"].to_numpy()] = positions["score"].to_numpy()
    missing = np.isnan(matrix)
    if missing.sum() > MAX_MISSING_SCORES:
        raise ValueError(f"Tag Genome is missing {missing.sum():,} film-tag scores")
    if missing.any():
        # The published file omits a single (airplane, Toy Story) row.
        matrix[missing] = np.nanmedian(matrix, axis=0)[np.nonzero(missing)[1]]
        print(f"genome: filled {missing.sum()} missing score(s) with the tag median")
    # TagDL predictions fall slightly outside the documented 0-1 range.
    return np.clip(matrix, 0.0, 1.0), movie_ids, tags


def load_metadata(archive: zipfile.ZipFile) -> pl.DataFrame:
    rows = [
        json.loads(line)
        for line in archive.read(GENOME_ROOT + "raw/metadata.json").decode("utf-8").splitlines()
    ]
    return pl.DataFrame(
        {
            "movie_id": [row["item_id"] for row in rows],
            "directors": [split_names(row["directedBy"]) for row in rows],
            "cast": [split_names(row["starring"]) for row in rows],
        },
        schema={
            "movie_id": pl.Int64,
            "directors": pl.List(pl.String),
            "cast": pl.List(pl.String),
        },
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW_DIR / "genome_2021.zip") as genome_archive:
        matrix, movie_ids, tags = load_genome(genome_archive)
        metadata = load_metadata(genome_archive)

    with zipfile.ZipFile(RAW_DIR / "ml-32m.zip") as archive:
        movies = pl.read_csv(io.BytesIO(archive.read("ml-32m/movies.csv")))
        links = pl.read_csv(
            io.BytesIO(archive.read("ml-32m/links.csv")), schema_overrides={"imdbId": pl.String}
        )
        ratings = pl.read_csv(io.BytesIO(archive.read("ml-32m/ratings.csv")))

    # Tag Genome 2021 and MovieLens 32M are separate snapshots, so a few films exist in only one.
    known = set(movies["movieId"].to_list()) & set(links["movieId"].to_list())
    keep = np.array([movie_id in known for movie_id in movie_ids])
    if (~keep).mean() > MAX_UNMATCHED_SHARE:
        raise ValueError(f"{(~keep).sum():,} genome films have no MovieLens record")
    print(f"catalog: dropped {(~keep).sum()} genome films that aren't in MovieLens 32M")
    matrix = matrix[keep]
    movie_ids = [movie_id for movie_id, kept in zip(movie_ids, keep) if kept]

    catalog = pl.DataFrame({"movie_id": movie_ids}, schema={"movie_id": pl.Int64})
    ratings = (
        ratings.rename({"userId": "user_id", "movieId": "movie_id"})
        .join(catalog, on="movie_id")
        .with_columns(user_split(pl.col("user_id")).alias("split"))
    )

    # Popularity comes only from training users so it can't leak hidden test ratings.
    popularity = (
        ratings.filter(pl.col("split") == "train", pl.col("rating") >= LIKED_RATING)
        .group_by("movie_id")
        .len("liked_count")
    )

    parsed = [parse_title(raw) for raw in movies["title"].to_list()]
    films = (
        catalog.join(
            movies.rename({"movieId": "movie_id"}).with_columns(
                pl.Series("clean_title", [title for title, _ in parsed]),
                pl.Series("year", [year for _, year in parsed], dtype=pl.Int32),
                pl.col("genres").replace("(no genres listed)", "").str.split("|"),
            ),
            on="movie_id",
        )
        .join(links.rename({"movieId": "movie_id", "imdbId": "imdb_id"}), on="movie_id")
        .join(metadata, on="movie_id", how="left")
        .join(popularity, on="movie_id", how="left")
        .with_columns(
            pl.col("liked_count").fill_null(0),
            pl.col("genres").list.eval(pl.element().filter(pl.element() != "")),
            pl.col("directors").fill_null([]),
            pl.col("cast").fill_null([]),
        )
        .select(
            "movie_id",
            "imdb_id",
            pl.col("clean_title").alias("title"),
            "year",
            "genres",
            "directors",
            "cast",
            "liked_count",
        )
        .sort("movie_id")
    )
    if films.height != len(movie_ids):
        raise ValueError(f"{len(movie_ids) - films.height} genome films have no MovieLens record")

    films.write_parquet(PROCESSED_DIR / "films.parquet")
    np.save(PROCESSED_DIR / "genome.npy", matrix)
    (PROCESSED_DIR / "tags.json").write_text(json.dumps(tags), encoding="utf-8")
    ratings.select("user_id", "movie_id", "rating", "timestamp", "split").write_parquet(
        PROCESSED_DIR / "ratings.parquet"
    )

    print(f"films: {films.height:,} | tags: {len(tags):,} | ratings: {ratings.height:,}")
    per_split = ratings.group_by("split").agg(
        pl.len().alias("ratings"), pl.col("user_id").n_unique().alias("users")
    )
    for split, count, users in per_split.sort("split").iter_rows():
        print(f"{split}: {users:,} users, {count:,} ratings")


if __name__ == "__main__":
    main()
