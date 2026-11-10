"""Load the processed catalog into PostgreSQL, replacing any earlier copy."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

from app.config import settings
from pipeline.catalog import PROCESSED_DIR, popularity_scores
from pipeline.embed import normalize
from pipeline.tag_policy import is_displayable

INIT_DIR = Path(__file__).resolve().parents[2] / "db" / "init"


def apply_schema(conn: psycopg.Connection) -> None:
    """Run the init scripts, which Docker only applies when a database is first created."""
    scripts = sorted(INIT_DIR.glob("*.sql"))
    if not scripts:
        raise FileNotFoundError(f"No schema scripts in {INIT_DIR}")
    for script in scripts:
        conn.execute(script.read_text(encoding="utf-8"))


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector.tolist()) + "]"


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet").join(
        pl.read_parquet(PROCESSED_DIR / "enrichment.parquet"), on="movie_id"
    )
    genome = normalize(np.load(PROCESSED_DIR / "genome.npy"))
    embeddings = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = popularity_scores(films["liked_count"].to_numpy())
    tags = json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8"))

    with psycopg.connect(settings.database_url) as conn:
        apply_schema(conn)
        conn.execute("TRUNCATE films, genome_tags")
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO genome_tags (position, tag, displayable) VALUES (%s, %s, %s)",
                [(position, tag, is_displayable(tag)) for position, tag in enumerate(tags)],
            )
            with cursor.copy(
                "COPY films (movie_id, imdb_id, title, year, genres, directors, cast_members,"
                " runtime, certification, wikipedia_title, popularity, genome, embedding)"
                " FROM STDIN"
            ) as copy:
                for index, row in enumerate(films.iter_rows(named=True)):
                    copy.write_row(
                        (
                            row["movie_id"],
                            row["imdb_id"],
                            row["title"],
                            row["year"],
                            row["genres"],
                            row["directors"],
                            row["cast"],
                            row["runtime"],
                            row["certification"],
                            row["wikipedia_title"],
                            float(popularity[index]),
                            vector_literal(genome[index]),
                            vector_literal(embeddings[index]),
                        )
                    )
        count = conn.execute("SELECT count(*) FROM films").fetchone()[0]
    print(f"loaded {count:,} films and {len(tags):,} tags")


if __name__ == "__main__":
    main()
