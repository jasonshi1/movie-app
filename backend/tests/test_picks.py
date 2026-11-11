import numpy as np
import pytest

from app.recommend import SAME_SERIES_SIMILARITY, choose_distinct, same_series


def themes(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Two genome vectors with the given cosine similarity."""
    angle = np.arccos(similarity)
    return np.array([1.0, 0.0]), np.array([np.cos(angle), np.sin(angle)])


UNRELATED = themes(0.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Shrek", "Shrek 2"),
        ("The Matrix", "The Matrix Reloaded"),
        ("The Terminator", "Terminator 2: Judgment Day"),
        ("Star Wars: Episode IV - A New Hope", "Star Wars: Episode V - The Empire Strikes Back"),
        ("Harry Potter and the Sorcerer's Stone", "Harry Potter and the Half-Blood Prince"),
        ("Léon: The Professional (a.k.a. The Professional)", "Léon"),
    ],
)
def test_titles_that_start_the_same_way_are_one_series(first, second):
    assert same_series((first, UNRELATED[0]), (second, UNRELATED[1]))


@pytest.mark.parametrize(
    ("first", "second"),
    [("Heat", "Heathers"), ("The Dark Knight", "The Dark Crystal"), ("Alien", "Up")],
)
def test_different_titles_with_different_themes_are_not(first, second):
    assert not same_series((first, UNRELATED[0]), (second, UNRELATED[1]))


def test_almost_identical_themes_mean_one_series_whatever_the_titles():
    near = themes(SAME_SERIES_SIMILARITY + 0.01)
    assert same_series(("Alien", near[0]), ("Aliens", near[1]))


def test_similar_but_different_films_are_kept_apart():
    # Seven and The Silence of the Lambs are 0.90 similar, the most of any two different films.
    close = themes(0.90)
    assert not same_series(("Seven", close[0]), ("The Silence of the Lambs", close[1]))


def film(title: str, axis: int) -> tuple[str, np.ndarray]:
    vector = np.zeros(4)
    vector[axis] = 1.0
    return title, vector


def test_choose_distinct_skips_sequels_and_keeps_the_order():
    candidates = [film("Toy Story 2", 0), film("Toy Story 3", 0), film("Up", 1), film("Heat", 2)]

    assert choose_distinct(candidates, 3) == [0, 2, 3]


def test_choose_distinct_also_avoids_films_still_showing():
    candidates = [film("Shrek 2", 0), film("Heat", 1)]

    assert choose_distinct(candidates, 1, showing=[film("Shrek", 3)]) == [1]


def test_choose_distinct_fills_up_with_skipped_films_when_nothing_else_is_left():
    candidates = [film("Rocky", 0), film("Rocky II", 0)]

    assert choose_distinct(candidates, 3) == [0, 1]
