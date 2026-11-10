import numpy as np
import polars as pl
import pytest

from pipeline.catalog import parse_title
from pipeline.enrich import merge_bindings, resolve_extracts
from pipeline.evaluate import (
    EvalUser,
    build_users,
    choose_alpha,
    evaluate,
    genre_matrix,
    genre_scorer,
    ndcg_and_recall,
    similarity_scorer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("City of Lost Children, The (Cité des enfants perdus, La) (1995)", ("The City of Lost Children", 1995)),
        ("Auberge espagnole, L' (2002)", ("L'Auberge espagnole", 2002)),
        ("Birdman: Or (The Unexpected Virtue of Ignorance) (2014)", ("Birdman: Or (The Unexpected Virtue of Ignorance)", 2014)),
        ("Hellraiser", ("Hellraiser", None)),
    ],
)
def test_parse_title(raw, expected):
    assert parse_title(raw) == expected


def test_merge_bindings_keeps_every_requested_film_and_rejects_bad_runtimes():
    bindings = [
        {
            "imdb": {"value": "tt1"},
            "runtime": {"value": "95.0"},
            "certificationLabel": {"value": "PG"},
            "article": {"value": "https://en.wikipedia.org/wiki/Toy_Story"},
        },
        {"imdb": {"value": "tt1"}, "runtime": {"value": "81.0"}},
        {"imdb": {"value": "tt2"}, "runtime": {"value": "9000"}},
    ]

    records = merge_bindings(bindings, ["tt1", "tt2", "tt3"])

    assert records["tt1"] == {"runtime": 95, "certification": "PG", "wikipedia": "Toy Story"}
    assert records["tt2"]["runtime"] is None
    assert records["tt3"] == {"runtime": None, "certification": None, "wikipedia": None}


def test_resolve_extracts_follows_redirects_and_marks_missing_pages():
    query = {
        "normalized": [{"from": "alien (film)", "to": "Alien (film)"}],
        "redirects": [{"from": "Alien (film)", "to": "Alien (1979 film)"}],
        "pages": {"1": {"title": "Alien (1979 film)", "extract": "A horror film."}},
    }

    resolved = resolve_extracts(query, ["alien (film)", "Nonexistent"])

    assert resolved == {"alien (film)": "A horror film.", "Nonexistent": ""}


def test_ndcg_and_recall_scores_a_perfect_list_as_one():
    ranked = np.array([3, 1, 2])

    assert ndcg_and_recall(ranked, np.array([3, 1]), k=3) == (1.0, 1.0)


def test_ndcg_rewards_hits_nearer_the_top():
    hidden = np.array([9])

    near_top, _ = ndcg_and_recall(np.array([9, 1, 2]), hidden, k=3)
    near_bottom, _ = ndcg_and_recall(np.array([1, 2, 9]), hidden, k=3)

    assert near_top > near_bottom > 0


def _ratings(rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "movie_id", "rating", "timestamp"], orient="row")


def test_build_users_splits_at_the_last_favourite_by_time():
    # Listed out of order; movie 30 is rated first and 50 is disliked before the cutoff.
    ratings = _ratings(
        [
            (1, 20, 4.0, 200),
            (1, 30, 5.0, 100),
            (1, 50, 1.0, 150),
            (1, 40, 4.5, 300),
            (1, 10, 2.0, 350),
        ]
    )
    movie_ids = np.array([10, 20, 30, 40, 50])

    [user] = build_users(ratings, movie_ids, limit=5, profile_size=2, min_hidden=1)

    assert user.favourites.tolist() == [2, 1]
    assert sorted(user.seen.tolist()) == [1, 2, 4]
    assert user.hidden.tolist() == [3]


def test_build_users_skips_users_without_enough_hidden_likes():
    ratings = _ratings([(1, 10, 5.0, 1), (1, 20, 5.0, 2)])

    assert build_users(ratings, np.array([10, 20]), limit=5, profile_size=2, min_hidden=1) == []


def test_evaluate_never_recommends_seen_films():
    # Film 1 is the closest match to the favourite, but it has already been seen.
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.1, 0.99]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    user = EvalUser(favourites=np.array([0]), seen=np.array([0, 1]), hidden=np.array([2]))

    scores = evaluate(similarity_scorer(vectors, np.zeros(4)), [user], profile_size=1, k=2)

    # Film 3 ranks first and hidden film 2 second, which would be pushed out if film 1 stayed.
    assert scores.first_picks.tolist() == [3]
    assert scores.recall[0] == 1.0
    assert scores.ndcg[0] == pytest.approx(1 / np.log2(3))


def test_choose_alpha_prefers_the_smallest_weight_near_the_best():
    sweep = [
        {"alpha": 0.0, "ndcg": 0.10},
        {"alpha": 1.0, "ndcg": 0.295},
        {"alpha": 4.0, "ndcg": 0.30},
    ]

    assert choose_alpha(sweep) == 1.0


def test_genre_scorer_does_not_favour_films_with_many_genres():
    genres = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.float32)

    scores = genre_scorer(genres, np.zeros(3))([np.array([0])])

    assert scores[0, 1] > scores[0, 2]


def test_genre_matrix_marks_each_films_genres():
    matrix = genre_matrix([["Comedy"], ["Drama", "Comedy"], []])

    assert matrix.tolist() == [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
