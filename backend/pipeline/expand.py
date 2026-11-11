"""Stage 6: bring the catalog up to date, from films up to 2014 to films released this year.

The Stage 1 catalog is the 9,585 films with Tag Genome scores, which stop around 2014. This
step adds two kinds of film:

- MovieLens films without Tag Genome scores that training users liked at least MIN_TRAIN_LIKES
  times, mostly from 2015 to 2023. They have ratings, so the ranker learns them like any other.
- Films first released from RECENT_FROM up to today, from Wikidata, that have an English
  Wikipedia article in at least MIN_SITELINKS languages. MovieLens ends in October 2023, so these
  have no ratings: they're recommended from their themes and description, and their popularity
  comes from Wikipedia page views.

Neither kind has Tag Genome scores, so pipeline.themes predicts them from the film's
description. Every film's source is kept in the "source" column: "genome", "movielens" or
"wikidata". Results from Wikidata and Wikipedia are cached in data/processed, so a repeat run
only asks for what's new. Run it after pipeline.embed, then retrain with pipeline.ranker.
"""

import argparse
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import polars as pl
from fastembed import TextEmbedding

from app.embedding import MODEL_NAME
from pipeline import themes
from pipeline.catalog import LIKED_RATING, PROCESSED_DIR, parse_title, popularity_scores, user_split
from pipeline.download import RAW_DIR
from pipeline.embed import BATCH_SIZE, MODEL_DIR, TOP_TAGS, film_text, normalize
from pipeline.enrich import (
    SPARQL_URL,
    USER_AGENT,
    _load,
    _request_json,
    _save,
    fetch_wikidata,
    fetch_wikipedia,
)

MIN_TRAIN_LIKES = 40
RECENT_FROM = date(2023, 1, 1)
MIN_SITELINKS = 10
# Films, feature films, animated and anime films, and documentaries: the classes well-known
# recent films carry on Wikidata. TV films and shorts are left out.
FILM_CLASSES = ("Q11424", "Q24869", "Q29168811", "Q202866", "Q20650540", "Q93204")
ANIMATED_CLASSES = {"Q29168811", "Q202866", "Q20650540"}
DOCUMENTARY_CLASS = "Q93204"
# Names come in English, or once for every language under "mul" when they're the same in all.
LABEL_LANGUAGES = ("en", "mul")
# Bump when the cached records change shape, so they're fetched again.
RECENT_CACHE_VERSION = 2
DAY_PRECISION, MONTH_PRECISION = 11, 10
# Wikidata-only films get IDs above every MovieLens ID, built from their Wikidata number, so a
# film keeps its ID across refreshes and favourites saved in browsers stay valid.
WIKIDATA_ID_OFFSET = 1_000_000_000
MAX_CAST = 10
PAGEVIEW_MONTHS = 12
PAGEVIEW_WORKERS = 8
DETAIL_BATCH = 150

PLAIN_CACHE = PROCESSED_DIR / "plain_embeddings.npz"
RECENT_CACHE = PROCESSED_DIR / "recent_films.json"
PEOPLE_CACHE = PROCESSED_DIR / "film_people.json"
PAGEVIEWS_CACHE = PROCESSED_DIR / "pageviews.json"
PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"

# Wikidata's genres are finer than MovieLens'. A genre maps to every MovieLens genre whose
# keyword appears in its English label, so "science fiction action film" is Sci-Fi and Action.
GENRE_KEYWORDS = {
    "action": "Action",
    "adventure": "Adventure",
    "animated": "Animation",
    "animation": "Animation",
    "children's": "Children",
    "family": "Children",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "noir": "Film-Noir",
    "horror": "Horror",
    "musical": "Musical",
    "mystery": "Mystery",
    "romance": "Romance",
    "romantic": "Romance",
    "science fiction": "Sci-Fi",
    "thriller": "Thriller",
    "war film": "War",
    "western": "Western",
}


def movielens_genres(labels: list[str], classes: list[str] = ()) -> list[str]:
    """MovieLens genres for a film's Wikidata genre labels and classes."""
    found = {genre for label in labels for keyword, genre in GENRE_KEYWORDS.items()
             if keyword in label.lower()}
    if ANIMATED_CLASSES & set(classes):
        found.add("Animation")
    if DOCUMENTARY_CLASS in classes:
        found.add("Documentary")
    return [genre for genre in themes.GENRE_ORDER if genre in found]


def wikidata_movie_id(qid: str) -> int:
    return WIKIDATA_ID_OFFSET + int(qid.removeprefix("Q"))


def _sparql(query: str) -> list[dict]:
    request = urllib.request.Request(
        SPARQL_URL,
        data=urllib.parse.urlencode({"query": query}).encode(),
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    rows = _request_json(request)["results"]["bindings"]
    time.sleep(1.0)
    return [{key: value["value"] for key, value in row.items()} for row in rows]


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def _article_title(uri: str) -> str:
    return urllib.parse.unquote(uri.rsplit("/wiki/", 1)[1]).replace("_", " ")


_LANGUAGES = ", ".join(f'"{language}"' for language in LABEL_LANGUAGES)


def _name_pattern(subject: str) -> str:
    """Bind ?value to the subject's English or language-neutral name, and ?lang to which."""
    return (f"{subject} rdfs:label ?value . BIND(LANG(?value) AS ?lang) "
            f"FILTER(?lang IN ({_LANGUAGES}))")


def preferred_names(rows: list[dict], key: str, item: str) -> dict[str, list[str]]:
    """Each key's names, one per item, taking the English name over the language-neutral one."""
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for row in rows:
        rank = LABEL_LANGUAGES.index(row["lang"])
        slot = (row[key], row[item])
        if slot not in best or rank < best[slot][0]:
            best[slot] = (rank, row["value"])
    names: dict[str, list[str]] = {}
    for (owner, _), (_, name) in sorted(best.items()):
        names.setdefault(owner, []).append(name)
    return names


def title_from_article(article: str) -> str:
    """A film's title from its Wikipedia article: "Wicked (2024 film)" is "Wicked"."""
    return re.sub(r"\s*\((?:[^()]* )?film\)$", "", article)


def latest_release(day: str, precision: int) -> date:
    """The latest a release date could be: a date known only to the month or year counts as
    the end of it, so a film dated just "2026" isn't taken as out before it surely is."""
    known = date.fromisoformat(day)
    if precision >= DAY_PRECISION:
        return known
    if precision == MONTH_PRECISION:
        next_month = date(known.year + known.month // 12, known.month % 12 + 1, 1)
        return next_month - timedelta(days=1)
    return date(known.year, 12, 31)


# --- Recent films from Wikidata ---------------------------------------------------------------


def recent_film_ids(year: int) -> list[str]:
    """Wikidata IDs of well-covered films with a release date in the given year."""
    classes = " ".join(f"wd:{qid}" for qid in FILM_CLASSES)
    rows = _sparql(f"""
SELECT DISTINCT ?film WHERE {{
  VALUES ?class {{ {classes} }}
  ?film wdt:P31 ?class ; wdt:P577 ?date ; wikibase:sitelinks ?links .
  FILTER(?links >= {MIN_SITELINKS})
  FILTER(?date >= "{year}-01-01"^^xsd:dateTime && ?date < "{year + 1}-01-01"^^xsd:dateTime)
  ?article schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> .
}}""")
    return sorted({_qid(row["film"]) for row in rows})


def _values(qids: list[str]) -> str:
    return " ".join(f"wd:{qid}" for qid in qids)


def recent_film_details(qids: list[str]) -> dict[str, dict]:
    """Title, dates, IDs, genres, people and runtime for each film, one property at a time so
    multi-valued properties don't multiply into huge results."""
    values = _values(qids)
    records = {
        qid: {"title": None, "article": None, "imdb": None, "dates": [], "classes": [],
              "genres": [], "directors": [], "cast": [], "runtime": None, "certification": None}
        for qid in qids
    }

    def each(pattern: str, extra: str = "") -> list[dict]:
        return _sparql(
            f"SELECT ?film ?value {extra} WHERE {{ VALUES ?film {{ {values} }} {pattern} }}"
        )

    def names(prop: str) -> dict[str, list[str]]:
        rows = each(f"?film wdt:{prop} ?item . {_name_pattern('?item')}", "?item ?lang")
        return preferred_names(rows, "film", "item")

    for film, title in preferred_names(
        [row | {"self": row["film"]} for row in each(_name_pattern("?film"), "?lang")],
        "film", "self",
    ).items():
        records[_qid(film)]["title"] = title[0]
    for row in each("?value schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> ."):
        records[_qid(row["film"])]["article"] = _article_title(row["value"])
    for row in each("?film wdt:P345 ?value ."):
        records[_qid(row["film"])]["imdb"] = row["value"]
    for row in each(
        "?film p:P577 ?statement . ?statement wikibase:rank ?rank ; psv:P577 ?node . "
        "?node wikibase:timeValue ?value ; wikibase:timePrecision ?precision . "
        "FILTER(?rank != wikibase:DeprecatedRank)",
        "?precision",
    ):
        records[_qid(row["film"])]["dates"].append([row["value"][:10], int(row["precision"])])
    for row in each("?film wdt:P31 ?value ."):
        records[_qid(row["film"])]["classes"].append(_qid(row["value"]))
    for film, genres in names("P136").items():
        records[_qid(film)]["genres"] = genres
    for film, directors in names("P57").items():
        records[_qid(film)]["directors"] = directors
    for film, certifications in names("P1657").items():
        records[_qid(film)]["certification"] = certifications[0]
    for row in each("?film wdt:P2047 ?value ."):
        minutes = float(row["value"])
        if 20 <= minutes <= 600:
            records[_qid(row["film"])]["runtime"] = round(minutes)
    for qid, cast in cast_members(f"VALUES ?film {{ {values} }}", "?film").items():
        records[_qid(qid)]["cast"] = cast
    return records


def cast_members(values_clause: str, key: str) -> dict[str, list[str]]:
    """Up to MAX_CAST cast members per film, in billing order where Wikidata records one."""
    rows = _sparql(f"""
SELECT {key} ?person ?value ?lang ?order WHERE {{
  {values_clause}
  ?film p:P161 ?statement . ?statement ps:P161 ?person .
  {_name_pattern("?person")}
  OPTIONAL {{ ?statement pq:P1545 ?order }}
}}""")
    owner = key.lstrip("?")
    best: dict[tuple[str, str], tuple[int, float, str]] = {}
    for row in rows:
        rank = LABEL_LANGUAGES.index(row["lang"])
        order = float(row["order"]) if row.get("order", "").isdigit() else 1e9
        slot = (row[owner], row["person"])
        if slot not in best or rank < best[slot][0]:
            best[slot] = (rank, order, row["value"])
    cast: dict[str, list[tuple[float, str]]] = {}
    for (film, _), (_, order, name) in best.items():
        cast.setdefault(film, []).append((order, name))
    return {film: [name for _, name in sorted(people)[:MAX_CAST]] for film, people in cast.items()}


def fetch_recent(today: date) -> dict[str, dict]:
    cache = _load(RECENT_CACHE)
    if cache.get("version") != RECENT_CACHE_VERSION:
        cache = {"version": RECENT_CACHE_VERSION}
    for year in range(RECENT_FROM.year, today.year + 1):
        # The current year is fetched again on every run, since films keep coming out.
        key = f"ids:{year}"
        if key not in cache or year == today.year:
            cache[key] = recent_film_ids(year)
            _save(RECENT_CACHE, cache)
    qids = sorted({qid for key, ids in cache.items() if key.startswith("ids:") for qid in ids})
    missing = [qid for qid in qids if qid not in cache]
    for start in range(0, len(missing), DETAIL_BATCH):
        cache.update(recent_film_details(missing[start : start + DETAIL_BATCH]))
        _save(RECENT_CACHE, cache)
        print(f"recent films: {start + DETAIL_BATCH:,}/{len(missing):,} described", flush=True)
    return {qid: cache[qid] for qid in qids}


def recent_rows(records: dict[str, dict], known_imdb: set[str], today: date) -> list[dict]:
    """Films first released from RECENT_FROM to today that aren't already in the catalog."""
    rows = []
    for qid, record in records.items():
        if not record["dates"] or not record["article"]:
            continue
        released = min(latest_release(day, precision) for day, precision in record["dates"])
        if not RECENT_FROM <= released <= today:
            continue
        if record["imdb"] and record["imdb"] in known_imdb:
            continue
        rows.append(
            {
                "movie_id": wikidata_movie_id(qid),
                "imdb_id": (record["imdb"] or "").removeprefix("tt"),
                "title": record["title"] or title_from_article(record["article"]),
                "year": released.year,
                "released": released.isoformat(),
                "genres": movielens_genres(record["genres"], record["classes"]),
                "directors": sorted(set(record["directors"])),
                "cast": record["cast"],
                "runtime": record["runtime"],
                "certification": record["certification"],
                "wikipedia_title": record["article"],
            }
        )
    return rows


# --- MovieLens films without Tag Genome scores ------------------------------------------------


def fetch_people(imdb_ids: list[str]) -> dict[str, dict]:
    """Directors and cast from Wikidata for films found by IMDb ID. Films cached with no
    directors are asked about again, since earlier runs only read English names."""
    cache = _load(PEOPLE_CACHE)
    missing = [imdb for imdb in imdb_ids if imdb not in cache or not cache[imdb]["directors"]]
    for start in range(0, len(missing), DETAIL_BATCH):
        batch = missing[start : start + DETAIL_BATCH]
        values = "VALUES ?imdb { " + " ".join(f'"{imdb}"' for imdb in batch) + " } ?film wdt:P345 ?imdb ."
        directors = preferred_names(_sparql(f"""
SELECT ?imdb ?person ?value ?lang WHERE {{
  {values} ?film wdt:P57 ?person . {_name_pattern("?person")}
}}"""), "imdb", "person")
        cast = cast_members(values, "?imdb")
        for imdb in batch:
            cache[imdb] = {"directors": sorted(directors.get(imdb, ())), "cast": cast.get(imdb, [])}
        _save(PEOPLE_CACHE, cache)
        print(f"people: {start + len(batch):,}/{len(missing):,}", flush=True)
    return cache


# --- Popularity from Wikipedia page views -----------------------------------------------------


def pageview_window(today: date) -> tuple[date, date]:
    """The first and last day of the last PAGEVIEW_MONTHS full months."""
    this_month = date(today.year, today.month, 1)
    month_index = this_month.year * 12 + this_month.month - 1 - PAGEVIEW_MONTHS
    return date(month_index // 12, month_index % 12 + 1, 1), this_month - timedelta(days=1)


def monthly_views(items: list[dict], released: str | None, window_start: date) -> float:
    """Average monthly views, counting only months from the film's release onwards, so a film
    out for two months isn't compared with a year of an older film's views."""
    first = window_start
    if released:
        release_month = date.fromisoformat(released).replace(day=1)
        first = max(first, release_month)
    months = [item["views"] for item in items if item["timestamp"][:8] >= first.strftime("%Y%m%d")]
    return float(np.mean(months)) if months else 0.0


def _fetch_views(title: str, start: date, end: date) -> list[dict]:
    article = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = (f"{PAGEVIEWS_API}/{article}/monthly/{start.strftime('%Y%m%d')}00/"
           f"{end.strftime('%Y%m%d')}00")
    try:
        return _request_json(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}))["items"]
    except urllib.error.HTTPError as error:
        if error.code == 404:  # no views recorded in the window
            return []
        raise


def fetch_pageviews(titles: list[str], today: date) -> dict[str, list[dict]]:
    start, end = pageview_window(today)
    key = f"{start.isoformat()}/{end.isoformat()}"
    cache = _load(PAGEVIEWS_CACHE)
    if cache.get("window") != key:
        cache = {"window": key, "views": {}}
    missing = [title for title in titles if title not in cache["views"]]
    with ThreadPoolExecutor(PAGEVIEW_WORKERS) as pool:
        for start_at in range(0, len(missing), 400):
            batch = missing[start_at : start_at + 400]
            for title, items in zip(batch, pool.map(lambda t: _fetch_views(t, start, end), batch)):
                cache["views"][title] = [
                    {"timestamp": item["timestamp"], "views": item["views"]} for item in items
                ]
            _save(PAGEVIEWS_CACHE, cache)
            print(f"page views: {start_at + len(batch):,}/{len(missing):,}", flush=True)
    return cache["views"]


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares intercept and slope, and R²."""
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    return float(intercept), float(slope), float(1 - residual.var() / y.var())


# --- Putting it together ----------------------------------------------------------------------


def plain_embeddings(model: TextEmbedding, films: pl.DataFrame) -> np.ndarray:
    """Each film's text embedding without themes, which themes are predicted from. Cached by
    movie ID, since embedding every film takes over ten minutes on a CPU."""
    cached: dict[int, np.ndarray] = {}
    if PLAIN_CACHE.exists():
        with np.load(PLAIN_CACHE) as data:
            cached = dict(zip(data["movie_ids"].tolist(), data["vectors"]))
    ids = films["movie_id"].to_list()
    missing = [row for row, movie_id in enumerate(ids) if movie_id not in cached]
    if missing:
        rows = films[missing]
        texts = [film_text(r["title"], r["year"], r["genres"], r["directors"], [], r["description"])
                 for r in rows.iter_rows(named=True)]
        new = normalize(np.array(list(model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32))
        cached.update(zip(rows["movie_id"].to_list(), new))
    vectors = np.stack([cached[movie_id] for movie_id in ids]).astype(np.float32)
    np.savez(PLAIN_CACHE, movie_ids=np.array(ids), vectors=vectors)
    print(f"plain embeddings: {len(missing):,} new, {len(ids) - len(missing):,} cached")
    return vectors


def core_catalog() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """The Stage 1 films and their aligned arrays, without anything an earlier run added."""
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    enrichment = pl.read_parquet(PROCESSED_DIR / "enrichment.parquet")
    genome = np.load(PROCESSED_DIR / "genome.npy")
    text = np.load(PROCESSED_DIR / "text_embeddings.npy")
    if "source" in films.columns:
        keep = (films["source"] == "genome").to_numpy()
        films, genome, text = films.filter(pl.Series(keep)), genome[keep], text[keep]
        enrichment = films.select("movie_id").join(enrichment, on="movie_id", how="left")
    return films.select(
        "movie_id", "imdb_id", "title", "year", "genres", "directors", "cast", "liked_count"
    ), enrichment, genome, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(),
                        help="treat this as today's date (YYYY-MM-DD)")
    today = parser.parse_args().today

    core, core_enrichment, core_genome, core_text = core_catalog()
    print(f"core: {core.height:,} films with Tag Genome scores")

    with zipfile.ZipFile(RAW_DIR / "ml-32m.zip") as archive:
        movies = pl.read_csv(io.BytesIO(archive.read("ml-32m/movies.csv")))
        links = pl.read_csv(
            io.BytesIO(archive.read("ml-32m/links.csv")), schema_overrides={"imdbId": pl.String}
        )
        ratings = pl.read_csv(io.BytesIO(archive.read("ml-32m/ratings.csv"))).rename(
            {"userId": "user_id", "movieId": "movie_id"}
        ).with_columns(user_split(pl.col("user_id")).alias("split"))

    # MovieLens films without Tag Genome scores, liked often enough by training users.
    train_likes = (
        ratings.filter(pl.col("split") == "train", pl.col("rating") >= LIKED_RATING)
        .group_by("movie_id").len("liked_count")
    )
    parsed = {row[0]: parse_title(row[1]) for row in movies.select("movieId", "title").iter_rows()}
    extras = (
        train_likes.filter(
            pl.col("liked_count") >= MIN_TRAIN_LIKES,
            ~pl.col("movie_id").is_in(core["movie_id"].to_list()),
        )
        .join(movies.rename({"movieId": "movie_id"}), on="movie_id")
        .join(links.rename({"movieId": "movie_id", "imdbId": "imdb_id"}), on="movie_id")
        .filter(pl.col("imdb_id").is_not_null())
        .with_columns(
            pl.col("movie_id").map_elements(lambda i: parsed[i][0], return_dtype=pl.String).alias("title"),
            pl.col("movie_id").map_elements(lambda i: parsed[i][1], return_dtype=pl.Int32).alias("year"),
            pl.col("genres").replace("(no genres listed)", "").str.split("|")
            .list.eval(pl.element().filter(pl.element() != "")),
        )
        .sort("movie_id")
    )
    extra_imdb = [f"tt{imdb}" for imdb in extras["imdb_id"].to_list()]
    wikidata = fetch_wikidata(extra_imdb)
    people = fetch_people(extra_imdb)
    extras = extras.with_columns(
        pl.Series("directors", [people[imdb]["directors"] for imdb in extra_imdb],
                  dtype=pl.List(pl.String)),
        pl.Series("cast", [people[imdb]["cast"] for imdb in extra_imdb], dtype=pl.List(pl.String)),
        pl.Series("runtime", [wikidata[imdb]["runtime"] for imdb in extra_imdb], dtype=pl.Int32),
        pl.Series("certification", [wikidata[imdb]["certification"] for imdb in extra_imdb],
                  dtype=pl.String),
        pl.Series("wikipedia_title", [wikidata[imdb]["wikipedia"] for imdb in extra_imdb],
                  dtype=pl.String),
        pl.lit(None, dtype=pl.String).alias("released"),
        pl.lit("movielens").alias("source"),
    )
    print(f"movielens: {extras.height:,} rated films without Tag Genome scores, "
          f"{extras.filter(pl.col('year') >= 2015).height:,} of them from 2015 on")

    known_imdb = {f"tt{imdb}" for imdb in core["imdb_id"].to_list()} | set(extra_imdb)
    recent = pl.DataFrame(
        recent_rows(fetch_recent(today), known_imdb, today),
        schema={"movie_id": pl.Int64, "imdb_id": pl.String, "title": pl.String, "year": pl.Int32,
                "released": pl.String, "genres": pl.List(pl.String),
                "directors": pl.List(pl.String), "cast": pl.List(pl.String),
                "runtime": pl.Int32, "certification": pl.String, "wikipedia_title": pl.String},
    ).with_columns(pl.lit(0, dtype=pl.UInt32).alias("liked_count"), pl.lit("wikidata").alias("source"))
    print(f"wikidata: {recent.height:,} films released {RECENT_FROM} to {today}")

    columns = ["movie_id", "imdb_id", "title", "year", "released", "genres", "directors", "cast",
               "liked_count", "runtime", "certification", "wikipedia_title", "source"]
    added = pl.concat([
        extras.with_columns(pl.col("liked_count").cast(pl.UInt32)).select(columns),
        recent.select(columns),
    ])
    titles = sorted({title for title in added["wikipedia_title"].to_list() if title})
    extracts = fetch_wikipedia(titles)
    added = added.with_columns(
        pl.Series("description", [extracts.get(title, "") if title else ""
                                  for title in added["wikipedia_title"].to_list()], dtype=pl.String)
    )

    core = core.join(core_enrichment, on="movie_id").with_columns(
        pl.lit(None, dtype=pl.String).alias("released"), pl.lit("genome").alias("source"),
        pl.col("liked_count").cast(pl.UInt32),
    ).select(*columns, "description")
    films = pl.concat([core, added.select(*columns, "description")])

    # Popularity: rated films use their likes, as before. Wikidata films get it from page views,
    # through a line fitted on the rated films that have both.
    all_titles = sorted({title for title in films["wikipedia_title"].to_list() if title})
    window_start, _ = pageview_window(today)
    views = fetch_pageviews(all_titles, today)
    monthly = np.array([
        monthly_views(views.get(title, []), released, window_start) if title else 0.0
        for title, released in zip(films["wikipedia_title"].to_list(), films["released"].to_list())
    ])
    rated = (films["source"] != "wikidata").to_numpy()
    liked_popularity = popularity_scores(films["liked_count"].to_numpy())
    has_views = monthly > 0
    intercept, slope, r2 = fit_line(
        np.log1p(monthly[rated & has_views]), liked_popularity[rated & has_views]
    )
    from_views = np.clip(intercept + slope * np.log1p(monthly), 0.0, 1.0)
    popularity = np.where(rated, liked_popularity, from_views)
    print(f"popularity from page views: R² {r2:.2f} on rated films; "
          f"Wikidata films median {np.median(popularity[~rated]):.2f}, "
          f"rated films median {np.median(popularity[rated]):.2f}")

    # Tag Genome themes for the added films, predicted from their descriptions.
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_DIR))
    plain = plain_embeddings(model, films)
    is_core = (films["source"] == "genome").to_numpy()
    tags = json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8"))
    predicted = themes.predict_for_catalog(
        plain, films["genres"].to_list(), films["year"].to_list(), core_genome, is_core, tags
    )
    genome = np.concatenate([core_genome, predicted[~is_core]])

    # Text embeddings for the added films, built the same way as the core films'.
    top_rows = np.argsort(-genome[~is_core], axis=1)[:, :TOP_TAGS]
    tag_names = np.array(tags)
    added_text = normalize(np.array(list(model.embed(
        [film_text(row["title"], row["year"], row["genres"], row["directors"],
                   tag_names[top_rows[index]].tolist(), row["description"])
         for index, row in enumerate(films.filter(pl.Series(~is_core)).iter_rows(named=True))],
        batch_size=BATCH_SIZE,
    )), dtype=np.float32))
    text = np.concatenate([core_text, added_text])

    # Ratings for every rated film, so the ranker can learn the MovieLens additions.
    rated_ids = films.filter(pl.col("source") != "wikidata").select("movie_id")
    catalog_ratings = ratings.join(rated_ids, on="movie_id")

    films = films.with_columns(pl.Series("popularity", popularity, dtype=pl.Float32),
                               pl.Series("monthly_views", monthly, dtype=pl.Float32))
    films.drop("runtime", "certification", "wikipedia_title", "description").write_parquet(
        PROCESSED_DIR / "films.parquet"
    )
    films.select("movie_id", "runtime", "certification", "wikipedia_title", "description").write_parquet(
        PROCESSED_DIR / "enrichment.parquet"
    )
    np.save(PROCESSED_DIR / "genome.npy", genome.astype(np.float32))
    np.save(PROCESSED_DIR / "text_embeddings.npy", text.astype(np.float32))
    catalog_ratings.select("user_id", "movie_id", "rating", "timestamp", "split").write_parquet(
        PROCESSED_DIR / "ratings.parquet"
    )
    by_source = films.group_by("source").agg(pl.len(), pl.col("year").min().alias("from"),
                                             pl.col("year").max().alias("to"))
    for source, count, first, last in by_source.sort("source").iter_rows():
        print(f"{source}: {count:,} films, {first} to {last}")
    print(f"ratings: {catalog_ratings.height:,}")


if __name__ == "__main__":
    main()
