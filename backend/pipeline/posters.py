"""Find a poster for each film: the lead image of its English Wikipedia article.

For almost every film that's the poster. Most posters are copyrighted, and Wikipedia uses them
under fair use to identify the film; the app does the same, showing them small next to the
film's title and crediting Wikipedia. Old films' posters are often in the public domain.

Results are cached in data/processed/posters.json by article title, with an empty string for
an article without an image, so a refresh only asks about new articles. pipeline.load_db reads
the cache. Run it after pipeline.expand, which gives every film its article title.
"""

import json
import time
import urllib.parse
import urllib.request

import polars as pl

from pipeline.catalog import PROCESSED_DIR
from pipeline.enrich import (
    PAUSE_SECONDS,
    USER_AGENT,
    WIKIPEDIA_API,
    _load,
    _request_json,
    _save,
    final_titles,
)

POSTERS_CACHE = PROCESSED_DIR / "posters.json"
# One of the widths Wikimedia keeps thumbnails at. The app shows posters at most 6rem wide,
# so this is sharp on high-density screens.
POSTER_WIDTH = 250
BATCH = 50
# The app's Content-Security-Policy allows images from these hosts only.
POSTER_HOSTS = ("upload.wikimedia.org", "thumb.wikimedia.org")


def clean_url(source: str) -> str:
    """The image address without its tracking parameters, or "" if it isn't on a host the
    app allows."""
    parts = urllib.parse.urlsplit(source)
    if parts.scheme != "https" or parts.hostname not in POSTER_HOSTS:
        return ""
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def resolve_posters(query: dict, titles: list[str]) -> dict[str, str]:
    """Map each requested title to its poster address, or "" if its article has no image."""
    sources = {
        page["title"]: clean_url(page["thumbnail"]["source"]) if "thumbnail" in page else ""
        for page in query["pages"].values()
    }
    return {title: sources.get(final, "") for title, final in final_titles(query, titles).items()}


def fetch_posters(titles: list[str]) -> dict[str, str]:
    cache = _load(POSTERS_CACHE)
    missing = [title for title in titles if title not in cache]
    for start in range(0, len(missing), BATCH):
        batch = missing[start : start + BATCH]
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": POSTER_WIDTH,
                # Without this, only freely licensed images come back, which leaves out
                # nearly every poster.
                "pilicense": "any",
                "pilimit": BATCH,
                "redirects": 1,
                "titles": "|".join(batch),
            }
        )
        request = urllib.request.Request(
            f"{WIKIPEDIA_API}?{params}", headers={"User-Agent": USER_AGENT}
        )
        cache.update(resolve_posters(_request_json(request)["query"], batch))
        _save(POSTERS_CACHE, cache)
        if (start // BATCH) % 25 == 0:
            print(f"posters: {start + len(batch):,}/{len(missing):,}", flush=True)
        time.sleep(PAUSE_SECONDS)
    return cache


def load_posters() -> dict[str, str]:
    """The cached poster address for each article title, empty if there's no cache yet."""
    if not POSTERS_CACHE.exists():
        return {}
    return json.loads(POSTERS_CACHE.read_text(encoding="utf-8"))


def main() -> None:
    enrichment = pl.read_parquet(PROCESSED_DIR / "enrichment.parquet")
    titles = sorted({title for title in enrichment["wikipedia_title"].to_list() if title})
    posters = fetch_posters(titles)
    found = sum(1 for title in titles if posters.get(title))
    print(f"posters for {found:,} of {len(titles):,} articles ({found / len(titles):.0%})")


if __name__ == "__main__":
    main()
