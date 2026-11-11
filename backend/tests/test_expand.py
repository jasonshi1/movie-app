from datetime import date

import numpy as np
import pytest

from pipeline import themes
from pipeline.expand import (
    WIKIDATA_ID_OFFSET,
    latest_release,
    monthly_views,
    movielens_genres,
    pageview_window,
    preferred_names,
    recent_rows,
    title_from_article,
    wikidata_movie_id,
)

TODAY = date(2026, 11, 10)


@pytest.mark.parametrize(
    ("labels", "classes", "expected"),
    [
        (["science fiction action film"], [], ["Action", "Sci-Fi"]),
        (["romantic comedy"], [], ["Comedy", "Romance"]),
        (["drama film", "war film"], [], ["Drama", "War"]),
        (["family film"], ["Q29168811"], ["Animation", "Children"]),
        (["comedy film"], ["Q202866"], ["Animation", "Comedy"]),
        ([], ["Q93204"], ["Documentary"]),
        (["biographical film"], [], []),
    ],
)
def test_wikidata_genres_map_to_movielens_genres(labels, classes, expected):
    assert movielens_genres(labels, classes) == expected


def test_wikidata_films_get_stable_ids_above_every_movielens_id():
    assert wikidata_movie_id("Q12345") == WIKIDATA_ID_OFFSET + 12345


def record(**overrides):
    base = {
        "title": "Some Film", "article": "Some Film (2025 film)", "imdb": "tt1234567",
        "dates": [["2025-03-07", 11], ["2025-02-14", 11]], "classes": ["Q11424"], "genres": ["drama film"],
        "directors": ["B", "A", "A"], "cast": ["C"], "runtime": 101, "certification": None,
    }
    return base | overrides


def test_recent_rows_use_the_earliest_release_date():
    (row,) = recent_rows({"Q1": record()}, set(), TODAY)

    assert row["released"] == "2025-02-14"
    assert row["year"] == 2025
    assert row["imdb_id"] == "1234567"
    assert row["directors"] == ["A", "B"]
    assert row["genres"] == ["Drama"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"dates": [["2022-09-01", 11], ["2023-05-01", 11]]},  # first shown before 2023
        {"dates": [["2027-01-01", 11]]},  # not out yet
        {"dates": [["2026-01-01", 9]]},  # only "2026", which may not have happened yet
        {"dates": [["2026-11-01", 10]]},  # only "November 2026"
        {"article": None},  # no English Wikipedia article
        {"dates": []},
    ],
)
def test_recent_rows_skip_films_outside_the_window_or_without_an_article(overrides):
    assert recent_rows({"Q1": record(**overrides)}, set(), TODAY) == []


def test_recent_rows_skip_films_already_in_the_catalog():
    assert recent_rows({"Q1": record()}, {"tt1234567"}, TODAY) == []


def test_films_without_an_imdb_id_are_kept():
    (row,) = recent_rows({"Q1": record(imdb=None)}, set(), TODAY)

    assert row["imdb_id"] == ""


def test_pageview_window_is_the_last_twelve_full_months():
    assert pageview_window(TODAY) == (date(2025, 11, 1), date(2026, 10, 31))
    assert pageview_window(date(2026, 1, 5)) == (date(2025, 1, 1), date(2025, 12, 31))


def test_monthly_views_count_only_months_since_release():
    items = [{"timestamp": f"2026{month:02d}0100", "views": views}
             for month, views in [(8, 10), (9, 3000), (10, 1000)]]

    assert monthly_views(items, None, date(2025, 11, 1)) == pytest.approx((10 + 3000 + 1000) / 3)
    assert monthly_views(items, "2026-09-12", date(2025, 11, 1)) == pytest.approx(2000)
    assert monthly_views([], "2026-09-12", date(2025, 11, 1)) == 0.0


def test_ridge_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4)).astype(np.float32)
    y = (x @ np.array([[0.1], [0.0], [-0.05], [0.02]]) + 0.5).astype(np.float32)

    predicted = themes.predict(themes.fit_ridge(x, y, alpha=0.01), x)

    np.testing.assert_allclose(predicted, np.clip(y, 0, 1), atol=1e-3)


def test_theme_agreement_is_perfect_for_identical_scores():
    rng = np.random.default_rng(1)
    scores = rng.random((20, 30)).astype(np.float32)

    result = themes.agreement(scores, scores)

    assert result["cosine"] == pytest.approx(1.0)
    assert result["top_overlap"] == pytest.approx(1.0)
    assert result["theme_correlation"] == pytest.approx(1.0)


def test_theme_features_add_genres_and_year_to_the_embedding():
    x = themes.features(np.zeros((2, 3), np.float32), [["Drama"], []], [2020, None])

    assert x.shape == (2, 3 + len(themes.GENRE_ORDER) + 1)
    assert x[0, 3 + themes.GENRE_ORDER.index("Drama")] == 1
    assert x[0, -1] == pytest.approx(1.0)
    assert x[1, -1] == 0.0


def test_rescaling_restores_each_themes_spread_and_keeps_the_order():
    real = np.linspace(0, 1, 101, dtype=np.float32)[:, None]
    fitted = 0.5 + (real - 0.5) * 0.1  # regression squeezes scores toward the middle

    rescaled = themes.rescale(fitted, real, np.array([[0.45], [0.5], [0.55]], np.float32))

    assert rescaled[0, 0] < rescaled[1, 0] < rescaled[2, 0]
    assert rescaled[0, 0] == pytest.approx(0.0, abs=0.02)
    assert rescaled[2, 0] == pytest.approx(1.0, abs=0.02)


def test_the_most_common_top_theme_share_spots_identical_predictions():
    same = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    varied = np.eye(10, dtype=np.float32)[:4]

    assert themes.most_common_top_theme(same, top=1) == 1.0
    assert themes.most_common_top_theme(varied, top=1) == 0.25


def test_a_film_without_a_name_takes_its_title_from_the_article():
    (row,) = recent_rows({"Q1": record(title=None, article="Wicked (2024 film)")}, set(), TODAY)

    assert row["title"] == "Wicked"


@pytest.mark.parametrize(
    ("article", "title"),
    [("Dune: Part Two", "Dune: Part Two"), ("Barbie (film)", "Barbie"),
     ("Wicked (2024 film)", "Wicked"), ("Nosferatu (2024 American film)", "Nosferatu"),
     ("Alien: Romulus", "Alien: Romulus")],
)
def test_titles_from_articles_drop_the_film_disambiguation(article, title):
    assert title_from_article(article) == title


def test_imprecise_dates_count_as_the_end_of_their_month_or_year():
    assert latest_release("2025-03-07", 11) == date(2025, 3, 7)
    assert latest_release("2025-02-01", 10) == date(2025, 2, 28)
    assert latest_release("2025-12-01", 10) == date(2025, 12, 31)
    assert latest_release("2025-01-01", 9) == date(2025, 12, 31)


def test_a_year_only_date_counts_once_the_year_is_over():
    (row,) = recent_rows({"Q1": record(dates=[["2025-01-01", 9]])}, set(), TODAY)

    assert row["released"] == "2025-12-31"


def test_english_names_win_over_language_neutral_ones():
    rows = [
        {"film": "F1", "item": "P1", "value": "Denis Villeneuve", "lang": "mul"},
        {"film": "F1", "item": "P2", "value": "Second Name (mul)", "lang": "mul"},
        {"film": "F1", "item": "P2", "value": "Second Name", "lang": "en"},
    ]

    assert preferred_names(rows, "film", "item") == {"F1": ["Denis Villeneuve", "Second Name"]}
