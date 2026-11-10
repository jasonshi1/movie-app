"""Embed a text description of every film for similarity search."""

import json

import numpy as np
import polars as pl
from fastembed import TextEmbedding

from app.embedding import MODEL_NAME
from pipeline.catalog import DATA_DIR, PROCESSED_DIR

MODEL_DIR = DATA_DIR / "models"
BATCH_SIZE = 64
TOP_TAGS = 15
MAX_DESCRIPTION_CHARS = 1500


def film_text(
    title: str,
    year: int | None,
    genres: list[str],
    directors: list[str],
    top_tags: list[str],
    description: str,
) -> str:
    parts = [f"{title} ({year})." if year else f"{title}."]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")
    if directors:
        parts.append(f"Directed by {', '.join(directors)}.")
    if top_tags:
        parts.append(f"Themes: {', '.join(top_tags)}.")
    if description:
        parts.append(description[:MAX_DESCRIPTION_CHARS])
    return " ".join(parts)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def main() -> None:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet").join(
        pl.read_parquet(PROCESSED_DIR / "enrichment.parquet"), on="movie_id"
    )
    genome = np.load(PROCESSED_DIR / "genome.npy")
    tags = np.array(json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8")))
    top_tag_rows = np.argsort(-genome, axis=1)[:, :TOP_TAGS]

    texts = [
        film_text(
            row["title"],
            row["year"],
            row["genres"],
            row["directors"],
            tags[top_tag_rows[index]].tolist(),
            row["description"],
        )
        for index, row in enumerate(films.iter_rows(named=True))
    ]

    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_DIR))
    vectors = normalize(np.array(list(model.embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32))
    np.save(PROCESSED_DIR / "text_embeddings.npy", vectors)
    print(f"embedded {vectors.shape[0]:,} films into {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()
