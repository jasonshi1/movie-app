import numpy as np

from pipeline.evaluate import EvalUser
from pipeline.recent_pick import (
    pool_pick_hits,
    popularity_from_views,
    random_pick_hits,
    standing_within_year,
)


def user(favourites: list[int], seen: list[int], hidden: list[int]) -> EvalUser:
    return EvalUser(np.array(favourites), np.array(seen), np.array(hidden))


def test_the_pick_is_the_best_pool_film_the_user_hadnt_rated():
    pool = np.array([False, False, True, True, True])
    # Film 0 scores best but isn't in the pool. Within it, 4 beats 3, which beats 2.
    scores = np.array([9.0, 0.0, 1.0, 2.0, 3.0])

    def scorer(profiles: list[np.ndarray]) -> np.ndarray:
        return np.tile(scores, (len(profiles), 1))

    users = [
        user([0], [0, 4], [3]),  # had rated 4, so 3 is picked: a hit
        user([0], [0], [2]),  # 4 is picked: a miss
        user([0], [0], [1]),  # liked no pool film, so left out
    ]

    np.testing.assert_array_equal(pool_pick_hits(scorer, users, 1, pool), [1.0, 0.0])
    np.testing.assert_allclose(random_pick_hits(users, pool), [1 / 2, 1 / 3])


def test_popularity_from_views_follows_the_line_through_rated_films_with_views():
    # Views of 9, 99 and 999 are evenly spaced in log1p, and so are their popularities.
    views = np.array([0, 9, 99, 999, 99], dtype=np.float32)
    popularity = np.array([0.9, 0.2, 0.4, 0.6, 0.0])
    fit_on = np.array([True, True, True, True, False])

    result = popularity_from_views(views, popularity, fit_on)

    # The film without views stays out of the fit, and so does the last one.
    np.testing.assert_allclose(result, [0.0, 0.2, 0.4, 0.6, 0.4], atol=1e-4)


def test_standing_is_the_percent_rank_within_each_year():
    values = np.array([5.0, 1.0, 3.0, 3.0, 100.0, 7.0, 2.0])
    years = np.array([2024, 2024, 2024, 2024, 2025, 2026, 2026])

    standing = standing_within_year(values, years)

    # As PostgreSQL's percent_rank(): ties share the lower rank, and a year's only film is 0.
    np.testing.assert_allclose(standing, [1.0, 0.0, 1 / 3, 1 / 3, 0.0, 1.0, 0.0])
