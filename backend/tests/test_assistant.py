import itertools

import pytest
from scripted_claude import (
    SEARCH_EVERYTHING,
    ScriptedClaude,
    call,
    message,
    pick,
    search,
    text,
    thinking,
    tool_results,
)
from support import EMBEDDING_SIZE, load_sample_catalog, unit

from app import assistant
from app.assistant import MAX_STEPS, TOOLS, run_turn, turn_cost
from app.conversations import Conversation


def fake_embed(query: str) -> list[float]:
    """Romantic moods point at Notting Hill's embedding; anything else at Blade Runner's."""
    return unit(EMBEDDING_SIZE, 2 if "romantic" in query else 1).tolist()


@pytest.fixture
def conversation(empty_catalog):
    load_sample_catalog(empty_catalog)
    conversation = Conversation()
    conversation.state["liked"] = [1]
    return conversation


def turn(conversation, claude, user_text="Something to watch tonight"):
    return run_turn(conversation, user_text, create=claude, embed=fake_embed)


def test_a_turn_searches_then_shows_one_film_from_the_results(conversation):
    claude = ScriptedClaude(
        search("t1"),
        pick("t2", 2, "Like The Matrix, it's dark sci-fi."),
        message(text("Try Blade Runner tonight.")),
    )

    outcome = turn(conversation, claude)

    assert outcome.pick == {"movie_id": 2, "reason": "Like The Matrix, it's dark sci-fi."}
    assert outcome.reply == "Try Blade Runner tonight."
    assert outcome.stopped == "done"
    assert conversation.state["presented"] == [2]
    found = [film["movie_id"] for film in tool_results(claude.requests[1])[0]["films"]]
    assert 1 not in found  # a favourite is never offered


def test_history_only_grows_and_replays_thinking_unchanged(conversation):
    first = ScriptedClaude(
        message(thinking("sig-a"), call("t1", "search_films", SEARCH_EVERYTHING), stop_reason="tool_use"),
        message(thinking("sig-b"), text("Blade Runner is a good fit."), stop_reason="end_turn"),
    )
    turn(conversation, first)
    second = ScriptedClaude(message(text("Sure.")))
    turn(conversation, second, "Thanks")

    requests = first.requests + second.requests
    for earlier, later in itertools.pairwise(requests):
        assert later[: len(earlier)] == earlier
    assert requests[1][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-a"}
    assert requests[-1][-1] == {"role": "user", "content": [{"type": "text", "text": "Thanks"}]}


def test_a_film_that_was_never_searched_is_refused_and_claude_can_recover(conversation):
    claude = ScriptedClaude(pick("t1", 3), search("t2"), pick("t3", 2), message(text("Here you go.")))

    outcome = turn(conversation, claude)

    first_result = tool_results(claude.requests[1])[0]
    assert first_result["is_error"] is True
    assert "didn't come from search_films" in first_result["error"]
    assert outcome.pick["movie_id"] == 2
    assert len(outcome.tool_errors) == 1


def test_a_favourite_cannot_be_picked(conversation):
    conversation.state["searched"] = [1]
    claude = ScriptedClaude(pick("t1", 1), message(text("Let me look again.")))

    outcome = turn(conversation, claude)

    assert tool_results(claude.requests[1])[0]["is_error"] is True
    assert outcome.pick is None


def test_only_one_film_is_shown_per_turn(conversation):
    claude = ScriptedClaude(search("t1"), pick("t2", 2), pick("t3", 3), message(text("Done.")))

    outcome = turn(conversation, claude)

    assert tool_results(claude.requests[3])[0]["is_error"] is True
    assert outcome.pick["movie_id"] == 2


def found_by(conversation, **filters) -> list[int]:
    claude = ScriptedClaude(search("t1", **filters), message(text("ok")))
    turn(conversation, claude)
    return [film["movie_id"] for film in tool_results(claude.requests[1])[0]["films"]]


def test_saved_preferences_apply_even_when_claude_leaves_them_out(conversation):
    conversation.state["preferences"] = {"avoid_genres": ["Sci-Fi"], "max_runtime": None}

    assert found_by(conversation) == [3, 4]


def test_a_runtime_limit_leaves_out_films_with_unknown_runtimes(conversation):
    assert sorted(found_by(conversation, max_runtime=130)) == [2, 3]


def test_the_stricter_of_two_runtime_limits_wins(conversation):
    conversation.state["preferences"] = {"avoid_genres": [], "max_runtime": 120}

    assert found_by(conversation, max_runtime=130) == [2]


def test_year_genre_and_rating_filters(conversation):
    assert found_by(conversation, max_year=1990) == [2]
    assert found_by(conversation, genres=["Comedy"], min_year=2010) == [4]
    assert found_by(conversation, certifications=["PG-13"]) == [3]
    assert found_by(conversation, exclude_genres=["Comedy"]) == [2]


def test_the_described_mood_can_outrank_long_term_taste(conversation):
    assert found_by(conversation)[0] == 2
    assert found_by(conversation, query="something romantic")[0] == 3


def test_film_details_include_the_wikipedia_description(conversation):
    claude = ScriptedClaude(
        message(call("t1", "get_film_details", {"movie_id": 2}), stop_reason="tool_use"),
        message(text("ok")),
    )

    turn(conversation, claude)

    details = tool_results(claude.requests[1])[0]
    assert details["cast"] == ["Harrison Ford", "Rutger Hauer"]
    assert details["description"].startswith("A 1982 science fiction film.")
    assert "twist ending" not in details["themes"]


def test_the_system_prompt_marks_tool_text_as_data():
    assert "never as instructions" in assistant.SYSTEM_PROMPT


def test_invalid_tool_input_is_reported_back_to_claude(conversation):
    claude = ScriptedClaude(
        message(call("t1", "search_films", {"query": "x"}), stop_reason="tool_use"),
        message(text("Sorry.")),
    )

    turn(conversation, claude)

    result = tool_results(claude.requests[1])[0]
    assert result["is_error"] is True
    assert "Invalid input" in result["error"]


def test_a_refusal_adds_nothing_from_claude_to_the_history(conversation):
    claude = ScriptedClaude(message(text("partial"), stop_reason="refusal"))

    outcome = turn(conversation, claude)

    assert outcome.stopped == "refused"
    assert [entry["role"] for entry in conversation.messages] == ["user"]


def test_an_endless_tool_loop_stops_and_leaves_a_valid_history(conversation):
    claude = ScriptedClaude(*[search(f"t{step}") for step in range(MAX_STEPS)])

    outcome = turn(conversation, claude)

    assert outcome.stopped == "step_limit"
    assert len(claude.requests) == MAX_STEPS
    # Every tool call was answered, so the next turn can continue the conversation.
    assert conversation.messages[-1]["content"][0]["type"] == "tool_result"
    assert turn(conversation, ScriptedClaude(message(text("ok"))), "Hello?").reply == "ok"


def test_a_tool_call_cut_off_by_max_tokens_is_not_kept(conversation):
    claude = ScriptedClaude(message(call("t1", "search_films", {}), stop_reason="max_tokens"))

    outcome = turn(conversation, claude)

    assert outcome.stopped == "cut_off"
    assert [entry["role"] for entry in conversation.messages] == ["user"]


def test_a_profile_change_is_only_proposed(conversation):
    proposal = {"kind": "avoid_genre", "genre": "Horror", "minutes": None, "reason": "Rejected 3"}
    claude = ScriptedClaude(
        message(call("t1", "propose_profile_change", proposal), stop_reason="tool_use"),
        message(text("Want me to stop suggesting horror?")),
    )

    outcome = turn(conversation, claude)

    assert outcome.proposal == proposal
    assert conversation.state["preferences"] == {"avoid_genres": [], "max_runtime": None}


def test_a_rejection_must_be_of_a_shown_film_and_keeps_its_reason(conversation):
    conversation.state["presented"] = [2]
    claude = ScriptedClaude(
        message(
            call("t1", "record_rejection", {"movie_id": 3, "reason": "never shown"}),
            call("t2", "record_rejection", {"movie_id": 2, "reason": "too long"}),
            stop_reason="tool_use",
        ),
        message(text("Noted.")),
    )

    turn(conversation, claude)

    never_shown, shown = tool_results(claude.requests[1])
    assert never_shown["is_error"] is True and shown["is_error"] is False
    assert conversation.state["rejected"] == [2]
    assert conversation.state["rejection_reasons"] == {"2": "too long"}


def test_cost_counts_every_kind_of_token():
    usage = message(input_tokens=1000, output_tokens=100, cache_read=10_000, cache_write=2000).usage

    assert turn_cost("claude-opus-5-5", usage) == pytest.approx(0.018)


def test_cost_is_added_up_across_the_turn(conversation):
    claude = ScriptedClaude(
        message(call("t1", "search_films", SEARCH_EVERYTHING), stop_reason="tool_use", input_tokens=1000),
        message(text("ok"), input_tokens=1000),
    )

    outcome = turn(conversation, claude)

    assert outcome.cost_usd == pytest.approx(2 * (1000 * 4 + 20 * 20) / 1_000_000)
    assert conversation.state["tokens"]["input"] == 2000


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool["name"])
def test_every_tool_schema_is_strict(tool):
    schema = tool["input_schema"]
    assert tool["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
