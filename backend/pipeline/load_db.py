"""Load the processed catalog into PostgreSQL, replacing any earlier copy."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

from app.config import settings
from app.ranker import encoder_to_bytes
from app.recommend import vector_literal
from pipeline.catalog import PROCESSED_DIR, popularity_scores
from pipeline.embed import normalize
from pipeline.tag_policy import is_displayable

INIT_DIR = Path(__file__).resolve().parents[2] / "db" / "init"
# Written by pipeline.ranker. Without it the catalog loads with no ranker and picks use Stage 1.
RANKER_PATH = PROCESSED_DIR / "ranker.npz"


def apply_schema(conn: psycopg.Connection) -> None:
    """Run the init scripts, which Docker only applies when a database is first created."""
    scripts = sorted(INIT_DIR.glob("*.sql"))
    if not scripts:
        raise FileNotFoundError(f"No schema scripts in {INIT_DIR}")
    for script in scripts:
        conn.execute(script.read_text(encoding="utf-8"))


def ranker_columns(ranker: dict[str, np.ndarray], row: int) -> tuple[str, str]:
    """A film's ranker input vector, and its output vector with the bias appended."""
    output = np.append(ranker["output_vectors"][row], ranker["bias"][row])
    return vector_literal(ranker["input_vectors"][row]), vector_literal(output)


def load_ranker(movie_ids: np.ndarray) -> dict[str, np.ndarray] | None:
    if not RANKER_PATH.exists():
        print(f"no ranker at {RANKER_PATH}, so picks will use the Stage 1 method")
        return None
    with np.load(RANKER_PATH, allow_pickle=False) as arrays:
        ranker = {name: arrays[name] for name in arrays.files}
    if not np.array_equal(ranker["movie_ids"], movie_ids):
        raise ValueError(f"{RANKER_PATH} was trained on a different catalog; retrain it")
    return ranker


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet").join(
        pl.read_parquet(PROCESSED_DIR / "enrichment.parquet"), on="movie_id"
    )
    genome = normalize(np.load(PROCESSED_DIR / "genome.npy"))
    embeddings = np.load(PROCESSED_DIR / "text_embeddings.npy")
    popularity = popularity_scores(films["liked_count"].to_numpy())
    tags = json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8"))
    ranker = load_ranker(films["movie_id"].to_numpy())

    with psycopg.connect(settings.database_url) as conn:
        apply_schema(conn)
        conn.execute("TRUNCATE films, genome_tags, ranker")
        if ranker is not None:
            conn.execute("INSERT INTO ranker (encoder) VALUES (%s)", (encoder_to_bytes(ranker),))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO genome_tags (position, tag, displayable) VALUES (%s, %s, %s)",
                [(position, tag, is_displayable(tag)) for position, tag in enumerate(tags)],
            )
            with cursor.copy(
                "COPY films (movie_id, imdb_id, title, year, genres, directors, cast_members,"
                " runtime, certification, wikipedia_title, description, popularity, genome, embedding,"
                " ranker_input, ranker_output)"
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
                            row["description"],
                            float(popularity[index]),
                            vector_literal(genome[index]),
                            vector_literal(embeddings[index]),
                            *(ranker_columns(ranker, index) if ranker is not None else (None, None)),
                        )
                    )
        count = conn.execute("SELECT count(*) FROM films").fetchone()[0]
    print(f"loaded {count:,} films and {len(tags):,} tags")


if __name__ == "__main__":
    main()
