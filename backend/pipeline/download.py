"""Download the MovieLens and Tag Genome archives into data/raw."""

import hashlib
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
GENOME_URL = "https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip"
USER_AGENT = "movie-app-pipeline"
CHUNK_BYTES = 1024 * 1024


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}))


def fetch(url: str, dest: Path) -> None:
    """Download url to dest, skipping the transfer if a complete copy already exists."""
    with _open(url) as response:
        expected = int(response.headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == expected:
            print(f"{dest.name}: already downloaded")
            return

        partial = dest.with_suffix(dest.suffix + ".part")
        with partial.open("wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)

    if partial.stat().st_size != expected:
        raise OSError(f"{dest.name}: incomplete download")
    partial.replace(dest)
    print(f"{dest.name}: downloaded {expected / CHUNK_BYTES:.0f} MB")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_movielens(archive: Path) -> None:
    with _open(MOVIELENS_URL + ".md5") as response:
        expected = response.read().decode().split()[0]
    if md5_of(archive) != expected:
        raise OSError(f"{archive.name}: checksum mismatch")
    print(f"{archive.name}: checksum verified")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    movielens = RAW_DIR / "ml-32m.zip"
    fetch(MOVIELENS_URL, movielens)
    verify_movielens(movielens)
    fetch(GENOME_URL, RAW_DIR / "genome_2021.zip")


if __name__ == "__main__":
    main()
