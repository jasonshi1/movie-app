import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from scripted_claude import ScriptedClaude, message, pick, search, text
from support import EMBEDDING_SIZE, load_sample_catalog, unit

from app import assistant, conversations, main
from app.config import settings

client = TestClient(main.app)


@pytest.fixture
def chat(empty_catalog, monkeypatch):
    """Enable chat with a fake key and a scripted Claude. Returns a function to set the script."""
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("test-key"))
    monkeypatch.setattr(assistant, "embed_query", lambda query: unit(EMBEDDING_SIZE, 1).tolist())

    def script(*replies) -> ScriptedClaude:
        claude = ScriptedClaude(*replies)
        monkeypatch.setattr(assistant, "claude_create", claude)
        return claude

    return script


def send(**body):
    return client.post("/api/chat", json={"favourites": [1], "message": "Something dark", **body})


def test_chat_is_unavailable_without_an_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    response = send()

    assert response.status_code == 503
    assert client.get("/api/config").json() == {"conversation": False}


def test_config_reports_chat_when_a_key_is_set(chat):
    assert client.get("/api/config").json() == {"conversation": True}


def test_a_chat_turn_returns_the_pick_with_its_details(chat):
    claude = chat(search("t1"), pick("t2", 2, "Dark and futuristic."), message(text("Try this.")))

    response = send()

    assert response.status_code == 200
    body = response.json()
    assert body["pick"]["film"]["title"] == "Blade Runner"
    assert body["pick"]["film"]["runtime"] == 117
    assert body["pick"]["reason"] == "Dark and futuristic."
    assert body["reply"] == "Try this."
    assert body["cost_usd"] > 0
    first_message = claude.requests[0][0]["content"][0]["text"]
    assert first_message == "My favourite films: The Matrix (1999).\n\nSomething dark"


def test_a_saved_conversation_is_replayed_exactly_after_a_rejection(chat):
    first = chat(search("t1"), pick("t2", 2), message(text("Try Blade Runner.")))
    conversation_id = send().json()["conversation_id"]

    second = chat(search("t3"), pick("t4", 3), message(text("Then Notting Hill.")))
    body = send(conversation_id=conversation_id, reject_movie_id=2, message="Lighter").json()

    history = first.requests[-1] + [
        {"role": "assistant", "content": [{"type": "text", "text": "Try Blade Runner."}]}
    ]
    assert second.requests[0][: len(history)] == history
    assert second.requests[0][-1]["content"][0]["text"] == "Not that one: Blade Runner (1982).\n\nLighter"
    assert body["pick"]["film"]["title"] == "Notting Hill"
    saved = conversations.load(conversation_id)
    assert saved.state["rejected"] == [2] and saved.state["presented"] == [2, 3]


def test_changed_favourites_and_preferences_are_added_as_a_new_message(chat):
    chat(message(text("Hi.")))
    conversation_id = send().json()["conversation_id"]
    claude = chat(message(text("Got it.")))

    send(
        conversation_id=conversation_id,
        favourites=[1, 3],
        preferences={"avoid_genres": ["Horror"], "max_runtime": 120},
        message="Updated",
    )

    assert claude.requests[0][-1]["content"][0]["text"] == (
        "My favourite films are now: The Matrix (1999), Notting Hill (1999).\n\n"
        "My saved preferences: avoid Horror; nothing over 120 minutes.\n\nUpdated"
    )


def test_rejecting_a_film_that_was_never_shown_is_refused(chat):
    chat()

    assert send(reject_movie_id=2).status_code == 422


def test_favourites_must_be_in_the_catalog(chat):
    chat()

    assert send(favourites=[999]).status_code == 422


def test_an_unknown_conversation_is_404(chat):
    chat()

    assert send(conversation_id="00000000-0000-0000-0000-000000000000").status_code == 404


def test_a_claude_outage_is_502_and_saves_nothing(chat, monkeypatch):
    def unreachable(_messages):
        raise anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        )

    monkeypatch.setattr(assistant, "claude_create", unreachable)
    saved = []
    monkeypatch.setattr(conversations, "save", lambda conversation: saved.append(conversation))

    assert send().status_code == 502
    assert saved == []


def test_a_full_conversation_must_start_over(chat, monkeypatch):
    chat(message(text("Hi.")))
    conversation_id = send().json()["conversation_id"]
    saved = conversations.load(conversation_id)
    saved.state["user_turns"] = main.MAX_USER_TURNS
    conversations.save(saved)

    assert send(conversation_id=conversation_id).status_code == 409


def test_messages_are_length_limited(chat):
    assert send(message="x" * 501).status_code == 422
