import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from support import GENOME_SIZE, insert_film, load_sample_catalog, tagged

from app import main
from app.recommend import POPULARITY_WEIGHT, shared_themes
from pipeline.embed import normalize
from pipeline.evaluate import similarity_scorer
from pipeline.tag_policy import is_displayable

client = TestClient(main.app)


@pytest.fixture
def catalog(empty_catalog):
    load_sample_catalog(empty_catalog)
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


def test_pick_returns_the_closest_film_first_with_a_reason(catalog):
    response = client.post("/api/pick", json={"liked": [1]})

    assert response.status_code == 200
    best = response.json()["picks"][0]
    assert best["film"]["title"] == "Blade Runner"
    assert best["film"]["runtime"] == 117
    assert best["film"]["wikipedia_url"] == "https://en.wikipedia.org/wiki/Blade_Runner"
    assert best["reason"]["similar_to"] == "The Matrix"


def test_pick_returns_three_films_best_first_each_with_a_reason(catalog):
    picks = client.post("/api/pick", json={"liked": [1]}).json()["picks"]

    # 100% Wolf has Notting Hill's themes, so it would be skipped, but it's the only film left.
    assert [pick["film"]["title"] for pick in picks] == ["Blade Runner", "Notting Hill", "100% Wolf"]
    assert all(pick["reason"]["similar_to"] == "The Matrix" for pick in picks)


def test_pick_can_ask_for_fewer_films(catalog):
    picks = client.post("/api/pick", json={"liked": [1], "count": 1}).json()["picks"]

    assert [pick["film"]["title"] for pick in picks] == ["Blade Runner"]


@pytest.mark.parametrize("count", [0, 6])
def test_pick_count_is_limited(catalog, count):
    assert client.post("/api/pick", json={"liked": [1], "count": count}).status_code == 422


def test_replacing_a_pick_leaves_out_the_films_still_showing(catalog):
    body = client.post(
        "/api/pick", json={"liked": [1], "rejected": [3], "count": 1, "showing": [2]}
    ).json()

    assert [pick["film"]["title"] for pick in body["picks"]] == ["100% Wolf"]


def summary(picks: list[dict]) -> list[tuple[str, bool]]:
    return [(pick["film"]["title"], pick["recent"]) for pick in picks]


def test_the_last_picks_can_be_asked_to_be_recent_releases(catalog):
    # The sample catalog ends in 2020, so films from 2018 on are recent: only 100% Wolf.
    def picks(recent: int) -> list[tuple[str, bool]]:
        body = client.post("/api/pick", json={"liked": [1], "count": 2, "recent": recent}).json()
        return summary(body["picks"])

    assert picks(0) == [("Blade Runner", False), ("Notting Hill", False)]
    assert picks(1) == [("Blade Runner", False), ("100% Wolf", True)]


def test_a_recent_pick_is_any_film_once_no_recent_one_is_left(catalog):
    body = client.post(
        "/api/pick", json={"liked": [1], "rejected": [4], "count": 2, "recent": 1}
    ).json()

    assert summary(body["picks"]) == [("Blade Runner", False), ("Notting Hill", False)]


def test_a_recent_film_picked_on_its_merits_isnt_picked_twice(empty_catalog):
    with psycopg.connect(empty_catalog) as conn:
        insert_film(conn, 1, "Alpha", tagged(t0=1.0), year=1999)
        insert_film(conn, 2, "Bravo", tagged(t0=1.0), popularity=0.5, year=2025)
        insert_film(conn, 3, "Charlie", tagged(t0=1.0, t1=0.5), popularity=0.3, year=1990)
        insert_film(conn, 4, "Delta", tagged(t5=1.0), year=2026)

    body = client.post("/api/pick", json={"liked": [1], "count": 2, "recent": 1}).json()

    assert summary(body["picks"]) == [("Bravo", True), ("Delta", True)]


def test_recent_picks_cant_outnumber_the_picks(catalog):
    response = client.post("/api/pick", json={"liked": [1], "count": 1, "recent": 2})

    assert response.status_code == 422


def test_turning_a_film_down_skips_its_sequels_too(empty_catalog):
    with psycopg.connect(empty_catalog) as conn:
        insert_film(conn, 1, "Alpha", tagged(t0=1.0))
        insert_film(conn, 2, "Rocky II", tagged(t0=1.0), popularity=0.5)
        insert_film(conn, 3, "Heat", tagged(t0=1.0, t1=0.5), popularity=0.1)
        insert_film(conn, 4, "Rocky", tagged(t5=1.0))

    def first_pick(rejected: list[int]) -> str:
        body = client.post("/api/pick", json={"liked": [1], "rejected": rejected, "count": 1})
        return body.json()["picks"][0]["film"]["title"]

    assert first_pick([]) == "Rocky II"
    assert first_pick([4]) == "Heat"


def test_pick_reason_never_shows_hidden_tags(catalog):
    themes = client.post("/api/pick", json={"liked": [1]}).json()["picks"][0]["reason"]["themes"]

    assert "twist ending" not in themes
    assert themes[:2] == ["dystopia", "cyberpunk"]


def test_pick_skips_favourites_and_rejected_films(catalog):
    picks = client.post("/api/pick", json={"liked": [1], "rejected": [2]}).json()["picks"]

    assert [pick["film"]["title"] for pick in picks] == ["Notting Hill", "100% Wolf"]


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

    monkeypatch.setattr(main, "pick_films", refuse)

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

    assert response.json()["picks"][0]["film"]["movie_id"] == int(scores.argmax()) + 1


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
