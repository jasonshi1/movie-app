import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from app import main
from app.recommend import POPULARITY_WEIGHT, shared_themes
from pipeline.embed import normalize
from pipeline.evaluate import similarity_scorer
from pipeline.load_db import vector_literal
from pipeline.tag_policy import is_displayable

client = TestClient(main.app)
GENOME_SIZE = 1084
EMBEDDING = vector_literal(np.eye(384, dtype=np.float32)[0])


def insert_film(conn, movie_id, title, genome, popularity=0.0, **extra):
    conn.execute(
        "INSERT INTO films (movie_id, imdb_id, title, year, genres, directors, cast_members,"
        " runtime, certification, wikipedia_title, popularity, genome, embedding)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            movie_id,
            f"{movie_id:07d}",
            title,
            extra.get("year", 2000),
            extra.get("genres", ["Drama"]),
            extra.get("directors", ["A Director"]),
            [],
            extra.get("runtime", 100),
            extra.get("certification"),
            extra.get("wikipedia_title"),
            popularity,
            vector_literal(genome),
            EMBEDDING,
        ),
    )


def tagged(**weights) -> np.ndarray:
    """Build a normalized genome vector from tag positions written as t0=1.0, t5=0.5."""
    vector = np.zeros(GENOME_SIZE, dtype=np.float32)
    for name, weight in weights.items():
        vector[int(name[1:])] = weight
    return vector / np.linalg.norm(vector)


@pytest.fixture
def catalog(empty_catalog):
    with psycopg.connect(empty_catalog) as conn:
        conn.cursor().executemany(
            "INSERT INTO genome_tags (position, tag, displayable) VALUES (%s, %s, %s)",
            [
                (0, "dystopia", True),
                (1, "cyberpunk", True),
                (2, "twist ending", False),
                (3, "romance", True),
            ],
        )
        insert_film(conn, 1, "The Matrix", tagged(t0=1.0, t1=1.0, t2=1.0))
        insert_film(
            conn,
            2,
            "Blade Runner",
            tagged(t0=1.0, t1=0.9, t2=1.0),
            popularity=0.2,
            wikipedia_title="Blade Runner",
            year=1982,
            runtime=117,
            certification="R",
        )
        insert_film(conn, 3, "Notting Hill", tagged(t3=1.0), popularity=0.3)
        insert_film(conn, 4, "100% Wolf", tagged(t3=1.0))
    return empty_catalog


def test_search_finds_films_by_part_of_the_title(catalog):
    response = client.get("/api/films/search", params={"q": "matr"})

    assert response.status_code == 200
    assert [film["title"] for film in response.json()] == ["The Matrix"]


def test_search_tolerates_a_typo(catalog):
    response = client.get("/api/films/search", params={"q": "Blade Runer"})

    assert response.json()[0]["title"] == "Blade Runner"


def test_search_treats_percent_signs_literally(catalog):
    response = client.get("/api/films/search", params={"q": "100%"})

    assert [film["title"] for film in response.json()] == ["100% Wolf"]


def test_search_with_blank_query_returns_nothing(catalog):
    assert client.get("/api/films/search", params={"q": "   "}).json() == []


def test_pick_returns_the_closest_film_with_a_reason(catalog):
    response = client.post("/api/pick", json={"liked": [1]})

    assert response.status_code == 200
    body = response.json()
    assert body["film"]["title"] == "Blade Runner"
    assert body["film"]["runtime"] == 117
    assert body["film"]["wikipedia_url"] == "https://en.wikipedia.org/wiki/Blade_Runner"
    assert body["reason"]["similar_to"] == "The Matrix"


def test_pick_reason_never_shows_hidden_tags(catalog):
    themes = client.post("/api/pick", json={"liked": [1]}).json()["reason"]["themes"]

    assert "twist ending" not in themes
    assert themes[:2] == ["dystopia", "cyberpunk"]


def test_pick_skips_favourites_and_rejected_films(catalog):
    body = client.post("/api/pick", json={"liked": [1], "rejected": [2]}).json()

    assert body["film"]["title"] == "Notting Hill"


def test_pick_is_404_when_every_film_is_excluded(catalog):
    response = client.post("/api/pick", json={"liked": [1], "rejected": [2, 3, 4]})

    assert response.status_code == 404


def test_pick_is_404_for_unknown_favourites(catalog):
    assert client.post("/api/pick", json={"liked": [999]}).status_code == 404


def test_pick_requires_at_least_one_favourite(catalog):
    assert client.post("/api/pick", json={"liked": []}).status_code == 422


def test_pick_is_503_when_the_database_is_down(monkeypatch):
    def refuse(*_):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(main, "pick_film", refuse)

    no_raise = TestClient(main.app, raise_server_exceptions=False)
    assert no_raise.post("/api/pick", json={"liked": [1]}).status_code == 503


@pytest.mark.parametrize("liked", [[0], [1, 4], [2, 3, 7, 9]])
def test_pick_matches_the_evaluated_score(empty_catalog, liked):
    """The app must rank exactly as the evaluation does, or the published scores don't apply."""
    random = np.random.default_rng(7)
    genome = normalize(random.random((12, GENOME_SIZE), dtype=np.float32))
    popularity = random.random(12, dtype=np.float32)
    with psycopg.connect(empty_catalog) as conn:
        for row in range(12):
            insert_film(conn, row + 1, f"Film {row}", genome[row], float(popularity[row]))

    scorer = similarity_scorer(genome, popularity, POPULARITY_WEIGHT)
    scores = scorer([np.array(liked)])[0]
    scores[liked] = -np.inf

    response = client.post("/api/pick", json={"liked": [row + 1 for row in liked]})

    assert response.json()["film"]["movie_id"] == int(scores.argmax()) + 1


def test_shared_themes_needs_both_sides_to_score_highly():
    taste = [0.9, 0.9, 0.0]
    film = [0.8, 0.0, 0.9]
    tags = [(0, "shared"), (1, "only taste"), (2, "only film")]

    assert shared_themes(taste, film, tags)[0] == "shared"


@pytest.mark.parametrize("tag", ["twist ending", "nudity (topless)", "oscar (best picture)", "boring"])
def test_tag_policy_hides_unsuitable_tags(tag):
    assert not is_displayable(tag)


def test_tag_policy_keeps_useful_themes():
    assert is_displayable("dystopia")
