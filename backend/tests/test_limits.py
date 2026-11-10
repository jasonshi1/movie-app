import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from scripted_claude import ScriptedClaude, message, pick, search, text
from support import EMBEDDING_SIZE, load_sample_catalog, unit

from app import assistant, limits, main
from app.config import settings
from app.limits import Limit, RateLimiter

client = TestClient(main.app)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_limiter_allows_a_window_of_requests_then_says_how_long_to_wait():
    clock = Clock()
    limiter = RateLimiter(clock)
    limit = Limit(requests=2, seconds=60)

    assert limiter.wait_seconds("chat", "1.2.3.4", limit) == 0
    clock.now += 10
    assert limiter.wait_seconds("chat", "1.2.3.4", limit) == 0
    assert limiter.wait_seconds("chat", "1.2.3.4", limit) == 50

    clock.now += 50
    assert limiter.wait_seconds("chat", "1.2.3.4", limit) == 0


def test_visitors_and_kinds_of_request_are_counted_separately():
    limiter = RateLimiter(Clock())
    limit = Limit(requests=1, seconds=60)

    assert limiter.wait_seconds("chat", "1.2.3.4", limit) == 0
    assert limiter.wait_seconds("chat", "5.6.7.8", limit) == 0
    assert limiter.wait_seconds("requests", "1.2.3.4", limit) == 0
    assert limiter.wait_seconds("chat", "1.2.3.4", limit) > 0


def test_expired_windows_are_pruned_when_many_visitors_are_tracked(monkeypatch):
    monkeypatch.setattr(limits, "MAX_TRACKED", 3)
    clock = Clock()
    limiter = RateLimiter(clock)
    for address in ("a", "b", "c"):
        limiter.wait_seconds("requests", address, Limit(1, 60))
    clock.now += 61

    limiter.wait_seconds("requests", "d", Limit(1, 60))

    assert set(limiter._windows) == {("requests", "d")}


def test_searches_and_picks_share_a_per_minute_limit(empty_catalog, monkeypatch):
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(settings, "requests_per_minute", 2)

    assert client.get("/api/films/search", params={"q": "matrix"}).status_code == 200
    assert client.post("/api/pick", json={"liked": [1]}).status_code == 200
    response = client.post("/api/pick", json={"liked": [1]})

    assert response.status_code == 429
    assert 0 < int(response.headers["Retry-After"]) <= 60
    assert client.get("/api/health").status_code == 200


@pytest.fixture
def chat(empty_catalog, monkeypatch):
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("test-key"))
    monkeypatch.setattr(assistant, "embed_query", lambda query: unit(EMBEDDING_SIZE, 1).tolist())

    def script(*replies) -> ScriptedClaude:
        claude = ScriptedClaude(*replies)
        monkeypatch.setattr(assistant, "claude_create", claude)
        return claude

    return script


def send():
    return client.post("/api/chat", json={"favourites": [1], "message": "Something dark"})


def test_each_chat_turn_adds_its_cost_to_the_day(chat):
    chat(search("t1"), pick("t2", 2), message(text("Try this.")))

    body = send().json()

    assert limits.spent_today() == pytest.approx(body["cost_usd"], abs=1e-6)


def test_chat_pauses_once_the_day_is_over_budget(chat, monkeypatch):
    monkeypatch.setattr(settings, "daily_spend_limit_usd", 0.50)
    limits.record_spend(0.50)
    claude = chat(message(text("Never sent.")))

    response = send()

    assert response.status_code == 503
    assert "today's budget" in response.json()["detail"]
    assert claude.requests == []
    assert client.post("/api/pick", json={"liked": [1]}).status_code == 200


def test_a_turn_that_fails_partway_still_counts_what_it_spent(chat):
    def overloaded(_messages):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.InternalServerError(
            "overloaded", response=httpx2.Response(529, request=request), body=None
        )

    chat(search("t1"), overloaded)

    assert send().status_code == 502
    assert limits.spent_today() > 0


def test_chat_has_its_own_hourly_limit(chat, monkeypatch):
    monkeypatch.setattr(settings, "chat_messages_per_hour", 1)
    chat(search("t1"), pick("t2", 2), message(text("Try this.")))

    assert send().status_code == 200
    response = send()

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 60


def test_visitors_are_the_connection_address_unless_a_proxy_header_is_trusted(monkeypatch):
    headers = {"fly-client-ip": "203.0.113.7", "x-forwarded-for": "198.51.100.1"}

    assert limits.visitor_address(headers, "10.0.0.2") == "10.0.0.2"

    monkeypatch.setattr(settings, "client_ip_header", "Fly-Client-IP")
    assert limits.visitor_address(headers, "10.0.0.2") == "203.0.113.7"
    assert limits.visitor_address({}, "10.0.0.2") == "10.0.0.2"


def test_rate_limits_follow_the_trusted_proxy_header(empty_catalog, monkeypatch):
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(settings, "client_ip_header", "Fly-Client-IP")
    monkeypatch.setattr(settings, "requests_per_minute", 1)

    def search(address):
        return client.get(
            "/api/films/search", params={"q": "matrix"}, headers={"Fly-Client-IP": address}
        ).status_code

    assert [search("203.0.113.7"), search("203.0.113.8"), search("203.0.113.7")] == [200, 200, 429]
