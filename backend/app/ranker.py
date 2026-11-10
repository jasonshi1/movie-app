"""Turn favourites into a taste vector with the trained ranker, using numpy only.

pipeline/ranker.py trains the model with PyTorch. The backend doesn't need PyTorch: each
film's trained vectors are stored in the database, and the small network that turns
favourites into a taste vector is applied here. A film's score is its output vector dotted
with the taste vector, plus the film's bias.
"""

import io

import numpy as np

ENCODER_ARRAYS = ("w1", "b1", "w2", "b2")
# Training never showed the model more favourites than this, so larger counts are capped.
TRAINED_MAX_FAVOURITES = 20


def taste_vectors(
    favourite_vectors: list[np.ndarray], encoder: dict[str, np.ndarray]
) -> np.ndarray:
    """Return one taste vector per person, from the input vectors of their favourites."""
    means = np.stack([vectors.mean(axis=0) for vectors in favourite_vectors])
    counts = np.log(
        np.array(
            [[min(len(vectors), TRAINED_MAX_FAVOURITES)] for vectors in favourite_vectors],
            dtype=means.dtype,
        )
    )
    features = np.concatenate([means, counts], axis=1)
    hidden = np.maximum(features @ encoder["w1"].T + encoder["b1"], 0)
    return means + hidden @ encoder["w2"].T + encoder["b2"]


def encoder_to_bytes(encoder: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: encoder[name] for name in ENCODER_ARRAYS})
    return buffer.getvalue()


def encoder_from_bytes(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in ENCODER_ARRAYS}
