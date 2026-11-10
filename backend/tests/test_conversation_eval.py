import psycopg
import pytest
from scripted_claude import ScriptedClaude, message, pick, search, text
from support import EMBEDDING_SIZE, load_sample_catalog, unit

from app.catalog_tools import CERTIFICATIONS, GENRES
from pipeline.conversation_cases import CASES, Case, Expect, Turn
from pipeline.evaluate_conversation import (
    TurnRecord,
    check_case,
    film_facts,
    resolve_favourites,
    run_case,
)

FACTS = {
    2: {"movie_id": 2, "title": "Blade Runner", "year": 1982, "runtime": 117,
        "genres": ["Action", "Sci-Fi", "Thriller"], "certification": "R"},
    3: {"movie_id": 3, "title": "Notting Hill", "year": 1999, "runtime": 124,
        "genres": ["Comedy", "Romance"], "certification": "PG-13"},
}


def record(pick, reply="Here you go.", reason="It fits."):
    return TurnRecord("msg", reply, pick, reason if pick else None, "done", 0.01, 1.0)


def failures_for(expect: Expect, *records, favourites=(1,)):
    case = Case("c", ("x",), tuple(Turn("m") for _ in records), expect)
    return check_case(case, list(records), FACTS, list(favourites))


def test_a_pick_meeting_every_limit_passes():
    expect = Expect(max_runtime=120, min_year=1980, max_year=1990, genres_any=("Sci-Fi",),
                    genres_none=("Comedy",), certifications=("R",))

    assert failures_for(expect, record(2)) == ([], [])


@pytest.mark.parametrize(
    ("expect", "fragment"),
    [
        (Expect(max_runtime=100), "runtime 117 is over 100"),
        (Expect(min_year=1990), "year 1982 is before 1990"),
        (Expect(max_year=1980), "year 1982 is after 1980"),
        (Expect(genres_any=("Comedy",)), "include none of"),
        (Expect(genres_none=("Thriller",)), "include one of"),
        (Expect(certifications=("PG",)), "rating R"),
    ],
)
def test_each_broken_limit_is_reported(expect, fragment):
    failures, _ = failures_for(expect, record(2))

    assert len(failures) == 1 and fragment in failures[0]


def test_no_pick_is_a_failure_unless_nothing_can_fit():
    assert failures_for(Expect(), record(None))[0] == ["no film was picked"]
    assert failures_for(Expect(no_pick=True), record(None, reply="Nothing after 2014."))[0] == []
    assert failures_for(Expect(no_pick=True), record(2))[0] == [
        "picked a film although nothing can fit"
    ]


def test_invented_favourite_and_repeated_picks_fail():
    assert "isn't in the catalog" in failures_for(Expect(), record(999))[0][0]
    assert failures_for(Expect(), record(2), favourites=(2,))[0] == ["picked one of the favourites"]
    assert "already shown" in failures_for(Expect(), record(2), record(2))[0][0]


def test_a_shorter_pick_is_required_after_too_long():
    expect = Expect(shorter_than_rejected=True)

    assert failures_for(expect, record(3), record(2))[0] == []
    assert failures_for(expect, record(2), record(3))[0] == [
        "the new pick isn't shorter than the rejected one"
    ]


def test_spoiler_words_are_flagged_not_failed():
    failures, flags = failures_for(Expect(), record(2, reason="The twist is great."))

    assert failures == [] and flags == ["possible spoiler: 'twist'"]


def test_the_cases_are_complete_and_valid():
    assert len(CASES) == 75
    for case in CASES:
        expect = case.expect
        assert set(expect.genres_any + expect.genres_none) <= set(GENRES), case.id
        assert set(expect.certifications) <= set(CERTIFICATIONS), case.id
        assert all(turn.reject for turn in case.turns[1:]), case.id


def test_a_case_runs_through_the_chat_code_path(empty_catalog):
    load_sample_catalog(empty_catalog)
    case = Case(
        "sample",
        ("The Matrix",),
        (Turn("Something dark"), Turn("Too long, something shorter.", reject=True)),
        Expect(shorter_than_rejected=True),
    )
    claude = ScriptedClaude(
        search("t1"), pick("t2", 3), message(text("Try Notting Hill.")),
        search("t3"), pick("t4", 2), message(text("Then Blade Runner.")),
    )
    embed = lambda query: unit(EMBEDDING_SIZE, 1).tolist()

    with psycopg.connect(empty_catalog) as conn:
        favourites = resolve_favourites(conn, case.favourites)
        records = run_case(case, favourites, claude, embed)
        facts = film_facts(conn, [r.pick for r in records if r.pick])

    assert favourites == [1]
    assert [r.pick for r in records] == [3, 2]
    assert claude.requests[3][-1]["content"][0]["text"].startswith("Not that one: Notting Hill (1999).")
    assert check_case(case, records, facts, favourites) == ([], [])


def test_an_ambiguous_title_needs_a_year(empty_catalog):
    load_sample_catalog(empty_catalog)
    with psycopg.connect(empty_catalog) as conn:
        conn.execute("UPDATE films SET title = 'The Matrix' WHERE movie_id = 2")
        with pytest.raises(ValueError, match="add the year"):
            resolve_favourites(conn, ("The Matrix",))
        assert resolve_favourites(conn, ("The Matrix (1982)",)) == [2]
