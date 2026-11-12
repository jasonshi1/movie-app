"""Find each film's IMDb rating and Rotten Tomatoes score, shown next to its title in a pick.

IMDb ratings come from IMDb's own file of every title's rating, updated daily and offered for
personal and non-commercial use, with the credit "Information courtesy of IMDb
(https://www.imdb.com). Used with permission."

Rotten Tomatoes doesn't publish its scores, so they come from Wikidata, where editors record a
film's Tomatometer score, the share of critics' reviews that are positive, with the date it was
read. Wikidata keeps the film's Rotten Tomatoes address too. A score can be a few years old.

Both change over time, so each run fetches them again. The IMDb file is about 9 MB and Wikidata
takes a few minutes. Results go to data/processed/review_scores.parquet, which pipeline.load_db
reads. Run it after pipeline.expand, which adds the newest films.
"""

import gzip
import re
import time
import urllib.parse
import urllib.request

import polars as pl

from pipeline.catalog import PROCESSED_DIR
from pipeline.download import RAW_DIR, fetch
from pipeline.enrich import (
    PAUSE_SECONDS,
    SPARQL_URL,
    USER_AGENT,
    WIKIDATA_BATCH,
    _request_json,
)

IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_RATINGS_PATH = RAW_DIR / "title.ratings.tsv.gz"
SCORES_PATH = PROCESSED_DIR / "review_scores.parquet"
# Wikidata's items for Rotten Tomatoes and for its Tomatometer. A film's review scores
# (property P444) say who gave them (P447) and how (P459): Rotten Tomatoes also records an
# average rating out of 10, which isn't used.
ROTTEN_TOMATOES = "Q105584"
TOMATOMETER = "Q108403393"

_PERCENT = re.compile(r"^\s*(\d{1,3})\s*%\s*$")
# Film pages are under m/, such as m/matrix. Anything else isn't linked.
_ROTTEN_TOMATOES_ID = re.compile(r"^m/[A-Za-z0-9_-]+$")


def download_imdb_ratings() -> None:
    """Fetch today's file. It's replaced each run, since a new day's file can be the same size."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMDB_RATINGS_PATH.unlink(missing_ok=True)
    fetch(IMDB_RATINGS_URL, IMDB_RATINGS_PATH)


def read_imdb_ratings(path) -> pl.DataFrame:
    """Each title's IMDb ID, such as tt0133093, and average rating out of 10."""
    with gzip.open(path, "rb") as file:
        table = pl.read_csv(
            file.read(),
            separator="\t",
            quote_char=None,
            null_values="\\N",
            schema={"tconst": pl.String, "averageRating": pl.Float64, "numVotes": pl.Int64},
        )
    return table.select(
        pl.col("tconst").alias("imdb"), pl.col("averageRating").alias("imdb_rating")
    )


def rotten_tomatoes_query(imdb_ids: list[str]) -> str:
    values = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    return f"""
SELECT ?imdb ?score ?date ?id WHERE {{
  VALUES ?imdb {{ {values} }}
  ?film wdt:P345 ?imdb .
  OPTIONAL {{ ?film wdt:P1258 ?id }}
  OPTIONAL {{
    ?film p:P444 ?statement .
    ?statement ps:P444 ?score ; pq:P447 wd:{ROTTEN_TOMATOES} ; pq:P459 wd:{TOMATOMETER} .
    OPTIONAL {{ ?statement pq:P585 ?date }}
  }}
}}"""


def percent(text: str) -> int | None:
    """83 from "83%", or None if the text isn't a percentage."""
    match = _PERCENT.match(text)
    if not match or int(match.group(1)) > 100:
        return None
    return int(match.group(1))


def merge_scores(bindings: list[dict], imdb_ids: list[str]) -> dict[str, dict]:
    """One record per requested film: its latest Tomatometer score and its page's address.

    A film can have several scores, read on different dates, and the latest is kept. One with
    no date counts as older than any dated one.
    """
    records: dict[str, dict] = {
        imdb_id: {"score": None, "date": None, "id": None} for imdb_id in imdb_ids
    }
    for binding in bindings:
        record = records[binding["imdb"]["value"]]
        page = binding.get("id", {}).get("value", "")
        if record["id"] is None and _ROTTEN_TOMATOES_ID.match(page):
            record["id"] = page
        score = percent(binding["score"]["value"]) if "score" in binding else None
        if score is None:
            continue
        date = binding.get("date", {}).get("value", "")
        if record["score"] is None or date > record["date"]:
            record["score"], record["date"] = score, date
    return records


def fetch_rotten_tomatoes(imdb_ids: list[str]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for start in range(0, len(imdb_ids), WIKIDATA_BATCH):
        batch = imdb_ids[start : start + WIKIDATA_BATCH]
        request = urllib.request.Request(
            SPARQL_URL,
            data=urllib.parse.urlencode({"query": rotten_tomatoes_query(batch)}).encode(),
            headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        )
        records.update(merge_scores(_request_json(request)["results"]["bindings"], batch))
        if (start // WIKIDATA_BATCH) % 10 == 0:
            print(f"rotten tomatoes: {start + len(batch):,}/{len(imdb_ids):,}", flush=True)
        time.sleep(PAUSE_SECONDS)
    return records


def review_scores(
    films: pl.DataFrame, imdb: pl.DataFrame, rotten_tomatoes: dict[str, dict]
) -> pl.DataFrame:
    """movie_id, imdb_rating, rotten_tomatoes and rotten_tomatoes_id for every film."""
    keyed = films.select("movie_id", ("tt" + pl.col("imdb_id")).alias("imdb"))
    keys = keyed["imdb"].to_list()
    return (
        keyed.join(imdb, on="imdb", how="left")
        .with_columns(
            pl.Series(
                "rotten_tomatoes",
                [rotten_tomatoes.get(key, {}).get("score") for key in keys],
                dtype=pl.Int32,
            ),
            pl.Series(
                "rotten_tomatoes_id",
                [rotten_tomatoes.get(key, {}).get("id") for key in keys],
                dtype=pl.String,
            ),
        )
        .select("movie_id", "imdb_rating", "rotten_tomatoes", "rotten_tomatoes_id")
    )


def load_review_scores() -> pl.DataFrame | None:
    """The saved scores, or None if pipeline.review_scores hasn't run."""
    return pl.read_parquet(SCORES_PATH) if SCORES_PATH.exists() else None


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet").select("movie_id", "imdb_id")
    download_imdb_ratings()
    imdb = read_imdb_ratings(IMDB_RATINGS_PATH)
    rotten_tomatoes = fetch_rotten_tomatoes([f"tt{imdb_id}" for imdb_id in films["imdb_id"]])
    scores = review_scores(films, imdb, rotten_tomatoes)
    scores.write_parquet(SCORES_PATH)

    total = scores.height
    for column in ("imdb_rating", "rotten_tomatoes", "rotten_tomatoes_id"):
        found = total - scores[column].null_count()
        print(f"{column}: {found:,} of {total:,} films ({found / total:.0%})")


if __name__ == "__main__":
    main()
