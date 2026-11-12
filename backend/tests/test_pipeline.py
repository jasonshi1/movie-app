import gzip

import numpy as np
import polars as pl
import pytest

from pipeline import load_db
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
from pipeline.posters import clean_url, resolve_posters
from pipeline.review_scores import merge_scores, percent, read_imdb_ratings, review_scores


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


def test_apply_schema_fails_loudly_when_scripts_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(load_db, "INIT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_db.apply_schema(None)


def test_resolve_posters_follows_redirects_and_drops_tracking_parameters():
    query = {
        "redirects": [{"from": "Alien (film)", "to": "Alien (1979 film)"}],
        "pages": {
            "1": {
                "title": "Alien (1979 film)",
                "thumbnail": {
                    "source": "https://upload.wikimedia.org/wikipedia/en/c/c3/Alien_movie_poster.jpg"
                    "?utm_source=en.wikipedia.org&utm_campaign=api"
                },
            },
            "-1": {"title": "No Image Film"},
        },
    }

    resolved = resolve_posters(query, ["Alien (film)", "No Image Film", "Nonexistent"])

    assert resolved == {
        "Alien (film)": "https://upload.wikimedia.org/wikipedia/en/c/c3/Alien_movie_poster.jpg",
        "No Image Film": "",
        "Nonexistent": "",
    }


def test_posters_only_come_from_hosts_the_app_allows():
    assert clean_url("https://thumb.wikimedia.org/a/250px-b.png?x=1") == (
        "https://thumb.wikimedia.org/a/250px-b.png"
    )
    assert clean_url("http://upload.wikimedia.org/a.png") == ""
    assert clean_url("https://example.com/a.png") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [("83%", 83), ("100%", 100), ("0%", 0), (" 7 % ", 7), ("7.8/10", None), ("101%", None), ("", None)],
)
def test_percent_reads_only_percentages(text, expected):
    assert percent(text) == expected


def _binding(imdb, score=None, date=None, page=None):
    binding = {"imdb": {"value": imdb}}
    if score is not None:
        binding["score"] = {"value": score}
    if date is not None:
        binding["date"] = {"value": date}
    if page is not None:
        binding["id"] = {"value": page}
    return binding


def test_merge_scores_keeps_the_latest_tomatometer_score_and_a_safe_page():
    bindings = [
        _binding("tt0133093", "87%", "2021-09-11T00:00:00Z", "m/matrix"),
        _binding("tt0133093", "83%", "2026-01-16T00:00:00Z", "m/matrix"),
        _binding("tt0133093", "90%", None, "m/matrix"),
        _binding("tt8999762", "not a score", "2025-05-03T00:00:00Z", "javascript:alert(1)"),
    ]

    records = merge_scores(bindings, ["tt0133093", "tt8999762", "tt0000001"])

    assert records["tt0133093"]["score"] == 83
    assert records["tt0133093"]["id"] == "m/matrix"
    assert records["tt8999762"] == {"score": None, "date": None, "id": None}
    assert records["tt0000001"] == {"score": None, "date": None, "id": None}


def test_review_scores_matches_imdb_ratings_and_rotten_tomatoes_to_films(tmp_path):
    ratings = tmp_path / "title.ratings.tsv.gz"
    with gzip.open(ratings, "wt", encoding="utf-8") as file:
        for line in (
            ["tconst", "averageRating", "numVotes"],
            ["tt0133093", "8.7", "2200000"],
            ["tt12037194", "7.5", "300000"],
        ):
            file.write("\t".join(line) + "\n")
    films = pl.DataFrame({"movie_id": [2571, 1100361125, 5], "imdb_id": ["0133093", "12037194", "0000005"]})
    rotten_tomatoes = {"tt0133093": {"score": 83, "date": "2026-01-16", "id": "m/matrix"}}

    scores = review_scores(films, read_imdb_ratings(ratings), rotten_tomatoes)

    assert scores.to_dicts() == [
        {"movie_id": 2571, "imdb_rating": 8.7, "rotten_tomatoes": 83, "rotten_tomatoes_id": "m/matrix"},
        {"movie_id": 1100361125, "imdb_rating": 7.5, "rotten_tomatoes": None, "rotten_tomatoes_id": None},
        {"movie_id": 5, "imdb_rating": None, "rotten_tomatoes": None, "rotten_tomatoes_id": None},
    ]
