import numpy as np
import pytest
from scripted_claude import ScriptedClaude, message, pick, search, text
from support import EMBEDDING_SIZE, load_sample_catalog, tagged, unit

from pipeline.simulate import (
    MAX_SUGGESTIONS,
    NUDGE,
    Catalog,
    Session,
    SimulatedUser,
    conversation_report,
    converse,
    hits_within,
    paired_difference,
    proportion,
    ranked_lists,
    summarize,
    walk_list,
)

# The same four films as the sample catalog: The Matrix, Blade Runner, Notting Hill, 100% Wolf.
# Tag 2 ("twist ending") is hidden from people, so it's never used in a reason.
CATALOG = Catalog(
    movie_ids=np.array([1, 2, 3, 4]),
    genome=np.stack([
        tagged(t0=1.0, t1=1.0, t2=1.0),
        tagged(t0=1.0, t1=0.9, t2=1.0),
        tagged(t3=1.0),
        tagged(t3=1.0),
    ]),
    tag_positions=np.array([0, 1, 3]),
    tag_names=["dystopia", "cyberpunk", "romance"],
)


def matrix_fan(seen=(0,)) -> SimulatedUser:
    """Named The Matrix as a favourite, and liked Notting Hill later."""
    return SimulatedUser(
        favourites=np.array([0]),
        seen=frozenset(seen),
        liked_later=frozenset({2}),
        taste=CATALOG.genome[[0, 2]].mean(axis=0),
    )


def test_a_simulated_user_accepts_only_films_they_liked_later():
    user = matrix_fan(seen=(0, 3))

    assert user.respond(2, CATALOG).kind == "hit"
    assert user.respond(3, CATALOG).message == "I've already seen that one."
    assert user.respond(1, CATALOG).kind == "reason"


def test_a_reason_names_the_biggest_differences_in_shown_tags():
    # Blade Runner scores higher than this user's taste on dystopia and lower on romance.
    response = matrix_fan().respond(1, CATALOG)

    assert response.message == "Something with less dystopia and more romance, please."


def test_quick_picks_walk_down_the_list_until_a_hit():
    session = walk_list(np.array([3, 1, 2]), matrix_fan(seen=(0, 3)), CATALOG)

    assert session.suggestions == [3, 1, 2]
    assert session.responses == ["seen", "reason", "hit"]
    assert session.hit_at == 3


def test_quick_picks_stop_after_the_last_suggestion():
    never = SimulatedUser(np.array([0]), frozenset(), frozenset(), CATALOG.genome[0])
    ranked = np.tile(np.array([1, 3]), MAX_SUGGESTIONS)

    session = walk_list(ranked, never, CATALOG)

    assert len(session.suggestions) == MAX_SUGGESTIONS and session.hit_at is None


def test_ranked_lists_leave_out_favourites_and_put_the_best_first():
    scores = np.arange(20, dtype=np.float64)[::-1].copy()  # row 0 scores highest
    user = SimulatedUser(np.array([0, 2]), frozenset(), frozenset(), np.zeros(1))

    ranked = ranked_lists(lambda profiles: np.tile(scores, (len(profiles), 1)), [user])

    assert ranked[0].tolist() == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_hit_rates_and_paired_differences():
    sessions = [Session([1], ["hit"], 1), Session([1, 2], ["reason", "hit"], 2),
                Session([1], ["seen"], None), Session([1], ["reason"], None)]

    assert hits_within(sessions, 1).tolist() == [True, False, False, False]
    assert proportion(hits_within(sessions, 2))[0] == 0.5
    summary = summarize(sessions)
    assert summary["seen_share"] == pytest.approx(1 / 5)
    assert summary["hit_within"][10][0] == 0.5
    assert paired_difference(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))[0] == 0.25


def embed(_query):
    return unit(EMBEDDING_SIZE, 1).tolist()


def test_a_conversation_reaches_a_hit_through_the_chat_code_path(empty_catalog):
    load_sample_catalog(empty_catalog)
    claude = ScriptedClaude(
        search("t1"), pick("t2", 2), message(text("Try Blade Runner.")),
        search("t3"), pick("t4", 3), message(text("Then Notting Hill.")),
    )

    session = converse(matrix_fan(), CATALOG, claude, embed)

    assert session.suggestions == [1, 2]
    assert session.responses == ["reason", "hit"]
    assert session.hit_at == 2
    assert session.cost_usd > 0
    assert claude.requests[0][-1]["content"][0]["text"] == (
        "My favourite films: The Matrix (1999).\n\nPick a film for me to watch tonight."
    )
    assert claude.requests[3][-1]["content"][0]["text"] == (
        "Not that one: Blade Runner (1982).\n\n"
        "Something with less dystopia and more romance, please."
    )


def test_a_conversation_without_picks_nudges_once_then_gives_up(empty_catalog):
    load_sample_catalog(empty_catalog)
    claude = ScriptedClaude(message(text("What mood are you in?")), message(text("Any genre?")))

    session = converse(matrix_fan(), CATALOG, claude, embed)

    assert session.suggestions == [] and session.hit_at is None
    assert session.replies == ["What mood are you in?", "Any genre?"]
    assert claude.requests[1][-1]["content"][0]["text"] == NUDGE


def test_the_conversation_report_shows_rates_cost_and_projections():
    claude = [Session([1], ["hit"], 1, cost_usd=0.4), Session([1, 2], ["reason", "reason"], None, 0.6)]
    quick = [Session([3], ["hit"], 1), Session([3], ["hit"], 1)]

    report = conversation_report(claude, quick, seconds=120)

    assert "| Conversation with Claude | 50.0%" in report
    assert "$1.00 in total, $0.500 per user" in report
    assert "200 users would cost about $100 and 1,000 about $500." in report
    assert "-50.0%" in report
