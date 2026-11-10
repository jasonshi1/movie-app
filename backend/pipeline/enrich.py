"""Add runtime, certification and short descriptions from Wikidata and Wikipedia.

Results are cached in data/processed, so an interrupted or repeated run only requests what's missing.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from pipeline.catalog import PROCESSED_DIR

USER_AGENT = "movie-app-pipeline/0.1 (https://github.com/jasonshi1/movie-app)"
SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_CACHE = PROCESSED_DIR / "wikidata.json"
WIKIPEDIA_CACHE = PROCESSED_DIR / "wikipedia.json"
WIKIDATA_BATCH = 200
WIKIPEDIA_BATCH = 20
PAUSE_SECONDS = 1.0
MAX_ATTEMPTS = 5


def _request_json(request: urllib.request.Request) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 5 * attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _save(path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def wikidata_query(imdb_ids: list[str]) -> str:
    values = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    return f"""
SELECT ?imdb ?runtime ?certificationLabel ?article WHERE {{
  VALUES ?imdb {{ {values} }}
  ?film wdt:P345 ?imdb .
  OPTIONAL {{ ?film wdt:P2047 ?runtime }}
  OPTIONAL {{
    ?film wdt:P1657 ?certification .
    ?certification rdfs:label ?certificationLabel .
    FILTER(LANG(?certificationLabel) = "en")
  }}
  OPTIONAL {{ ?article schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> }}
}}"""


def merge_bindings(bindings: list[dict], imdb_ids: list[str]) -> dict[str, dict]:
    """Collapse SPARQL rows to one record per film; every requested ID gets a cache entry."""
    records: dict[str, dict] = {
        imdb_id: {"runtime": None, "certification": None, "wikipedia": None} for imdb_id in imdb_ids
    }
    for binding in bindings:
        record = records[binding["imdb"]["value"]]
        if "runtime" in binding and record["runtime"] is None:
            minutes = float(binding["runtime"]["value"])
            # A few entries hold placeholder or series-length values.
            if 20 <= minutes <= 600:
                record["runtime"] = round(minutes)
        if "certificationLabel" in binding and record["certification"] is None:
            record["certification"] = binding["certificationLabel"]["value"]
        if "article" in binding and record["wikipedia"] is None:
            record["wikipedia"] = urllib.parse.unquote(
                binding["article"]["value"].rsplit("/wiki/", 1)[1]
            ).replace("_", " ")
    return records


def fetch_wikidata(imdb_ids: list[str]) -> dict:
    cache = _load(WIKIDATA_CACHE)
    missing = [imdb_id for imdb_id in imdb_ids if imdb_id not in cache]
    for start in range(0, len(missing), WIKIDATA_BATCH):
        batch = missing[start : start + WIKIDATA_BATCH]
        request = urllib.request.Request(
            SPARQL_URL,
            data=urllib.parse.urlencode({"query": wikidata_query(batch)}).encode(),
            headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        )
        cache.update(merge_bindings(_request_json(request)["results"]["bindings"], batch))
        _save(WIKIDATA_CACHE, cache)
        print(f"wikidata: {start + len(batch):,}/{len(missing):,}", flush=True)
        time.sleep(PAUSE_SECONDS)
    return cache


def resolve_extracts(query: dict, titles: list[str]) -> dict[str, str]:
    """Map each requested title to its intro text, following normalization and redirects."""
    forward: dict[str, str] = {}
    for step in query.get("normalized", []) + query.get("redirects", []):
        forward[step["from"]] = step["to"]
    extracts = {page["title"]: page.get("extract", "") for page in query["pages"].values()}

    resolved = {}
    for title in titles:
        current = title
        seen = {current}
        while current in forward and forward[current] not in seen:
            current = forward[current]
            seen.add(current)
        resolved[title] = extracts.get(current, "")
    return resolved


def fetch_wikipedia(titles: list[str]) -> dict:
    cache = _load(WIKIPEDIA_CACHE)
    missing = [title for title in titles if title not in cache]
    for start in range(0, len(missing), WIKIPEDIA_BATCH):
        batch = missing[start : start + WIKIPEDIA_BATCH]
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": WIKIPEDIA_BATCH,
                "redirects": 1,
                "titles": "|".join(batch),
            }
        )
        request = urllib.request.Request(
            f"{WIKIPEDIA_API}?{params}", headers={"User-Agent": USER_AGENT}
        )
        cache.update(resolve_extracts(_request_json(request)["query"], batch))
        _save(WIKIPEDIA_CACHE, cache)
        if (start // WIKIPEDIA_BATCH) % 25 == 0:
            print(f"wikipedia: {start + len(batch):,}/{len(missing):,}", flush=True)
        time.sleep(PAUSE_SECONDS)
    return cache


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    imdb_ids = [f"tt{imdb_id}" for imdb_id in films["imdb_id"].to_list()]

    wikidata = fetch_wikidata(imdb_ids)
    titles = sorted({record["wikipedia"] for record in wikidata.values() if record["wikipedia"]})
    wikipedia = fetch_wikipedia(titles)

    records = [wikidata[imdb_id] for imdb_id in imdb_ids]
    enriched = films.select("movie_id").with_columns(
        pl.Series("runtime", [r["runtime"] for r in records], dtype=pl.Int32),
        pl.Series("certification", [r["certification"] for r in records], dtype=pl.String),
        pl.Series("wikipedia_title", [r["wikipedia"] for r in records], dtype=pl.String),
        pl.Series(
            "description",
            [wikipedia.get(r["wikipedia"], "") if r["wikipedia"] else "" for r in records],
            dtype=pl.String,
        ),
    )
    enriched.write_parquet(PROCESSED_DIR / "enrichment.parquet")

    total = enriched.height
    for column in ("runtime", "certification", "wikipedia_title"):
        print(f"{column}: {enriched[column].drop_nulls().len() / total:.1%}")
    print(f"description: {enriched.filter(pl.col('description') != '').height / total:.1%}")


if __name__ == "__main__":
    main()
