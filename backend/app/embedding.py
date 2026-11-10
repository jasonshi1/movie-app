"""Embed text with the same model the catalog's film descriptions were embedded with."""

from functools import cache

import numpy as np
from fastembed import TextEmbedding

from app.config import settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@cache
def load_model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=settings.embedding_model_dir)


def embed_query(text: str) -> list[float]:
    vector = next(iter(load_model().query_embed(text)))
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm else vector).tolist()
