import numpy as np
import psycopg
import pytest
import torch
from fastapi.testclient import TestClient
from support import GENOME_SIZE, insert_film, tagged

from app import main
from app.ranker import encoder_from_bytes, encoder_to_bytes
from pipeline.embed import normalize
from pipeline.evaluate import EvalUser
from pipeline.ranker import (
    MAX_FAVOURITES,
    Ranker,
    TrainingSettings,
    export,
    film_features,
    fit,
    ranker_scorer,
)

TINY = TrainingSettings(dim=8, hidden=16, dropout=0.0, batch_size=32, max_epochs=40, patience=40)


def random_ranker(films: int = 30, features: int = 6) -> Ranker:
    torch.manual_seed(0)
    model = Ranker(torch.randn(films, features), TINY)
    with torch.no_grad():
        model.input_ids.weight.normal_()
        model.output_ids.weight.normal_()
        model.bias.normal_()
    return model.eval()


def test_app_scores_match_the_trained_model():
    model = random_ranker()
    profiles = [np.array([3]), np.array([1, 4, 9]), np.arange(10, 30)]
    favourites = torch.zeros(len(profiles), MAX_FAVOURITES, dtype=torch.long)
    mask = torch.zeros(len(profiles), MAX_FAVOURITES, dtype=torch.bool)
    for row, profile in enumerate(profiles):
        favourites[row, : profile.size] = torch.from_numpy(profile)
        mask[row, : profile.size] = True

    with torch.no_grad():
        expected = model(favourites, mask).numpy()
    actual = ranker_scorer(export(model))(profiles)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_encoder_weights_survive_a_round_trip_through_bytes():
    exported = export(random_ranker())
    restored = encoder_from_bytes(encoder_to_bytes(exported))
    for name, values in restored.items():
        np.testing.assert_array_equal(values, exported[name])


def test_film_features_are_standardized_and_ignore_constant_columns():
    genome = np.array([[0.1, 0.5], [0.3, 0.5], [0.5, 0.5]], dtype=np.float32)
    text = np.array([[1.0], [0.0], [-1.0]], dtype=np.float32)
    features = film_features(genome, text)
    np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(features[:, [0, 2]].std(axis=0), 1, atol=1e-5)
    np.testing.assert_array_equal(features[:, 1], 0)


def clustered_users(count: int, rng: np.random.Generator) -> list[EvalUser]:
    """Users who like films from only one of two groups of 40 (rows 0-39 and 40-79)."""
    users = []
    for _ in range(count):
        group = rng.integers(2) * 40
        liked = group + rng.permutation(40)
        users.append(EvalUser(favourites=liked[:MAX_FAVOURITES], seen=liked[:MAX_FAVOURITES],
                              hidden=liked[MAX_FAVOURITES:30]))
    return users


def test_training_learns_which_films_go_together():
    rng = np.random.default_rng(0)
    # The features carry no information, so the model has to learn the groups from users.
    features = rng.normal(size=(80, 4)).astype(np.float32)
    exported, history = fit(
        features, clustered_users(300, rng), clustered_users(50, rng), TINY,
        torch.device("cpu"), log=lambda _: None,
    )

    scores = ranker_scorer(exported)([np.arange(5), np.arange(40, 45)])
    assert scores[0, 5:40].mean() > scores[0, 40:].mean()
    assert scores[1, 45:].mean() > scores[1, :40].mean()
    assert max(entry["stopping_ndcg"] for entry in history) > history[0]["stopping_ndcg"]


def load_films(database_url: str, genome: np.ndarray, popularity: np.ndarray, ranker: dict) -> None:
    """Insert one film per row, with movie ID row + 1, and the ranker's encoder."""
    with psycopg.connect(database_url) as conn:
        conn.execute("INSERT INTO ranker (encoder) VALUES (%s)", (encoder_to_bytes(ranker),))
        for row in range(len(genome)):
            insert_film(
                conn, row + 1, f"Film {row}", genome[row], float(popularity[row]),
                ranker_input=ranker["input_vectors"][row],
                ranker_output=np.append(ranker["output_vectors"][row], ranker["bias"][row]),
            )


client = TestClient(main.app)


@pytest.mark.parametrize("liked", [[0], [1, 4], [2, 3, 7, 9]])
def test_pick_matches_the_evaluated_ranker_score(empty_catalog, liked):
    """The app must rank exactly as the evaluation does, or the published scores don't apply."""
    exported = export(random_ranker(films=12))
    rng = np.random.default_rng(3)
    load_films(
        empty_catalog, normalize(rng.random((12, GENOME_SIZE), dtype=np.float32)),
        rng.random(12, dtype=np.float32), exported,
    )

    scores = ranker_scorer(exported)([np.array(liked)])[0]
    scores[liked] = -np.inf
    response = client.post("/api/pick", json={"liked": [row + 1 for row in liked]})

    assert response.json()["film"]["movie_id"] == int(scores.argmax()) + 1


def test_pick_uses_the_ranker_instead_of_stage_1_when_one_is_loaded(empty_catalog):
    # Stage 1 would pick film 2, the most popular. The ranker's bias prefers film 3.
    dim, hidden = 2, 3
    ranker = {
        "input_vectors": np.zeros((3, dim), dtype=np.float32),
        "output_vectors": np.zeros((3, dim), dtype=np.float32),
        "bias": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "w1": np.zeros((hidden, dim + 1), dtype=np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": np.zeros((dim, hidden), dtype=np.float32),
        "b2": np.zeros(dim, dtype=np.float32),
    }
    genome = np.stack([tagged(t0=1.0)] * 3)
    load_films(empty_catalog, genome, np.array([0.0, 0.9, 0.1]), ranker)

    body = client.post("/api/pick", json={"liked": [1]}).json()

    assert body["film"]["movie_id"] == 3
    assert body["reason"]["similar_to"] == "Film 0"
