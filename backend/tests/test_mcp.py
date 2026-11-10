import json

import pytest
from fastapi.testclient import TestClient
from support import EMBEDDING_SIZE, load_sample_catalog, unit

from app import main, mcp_server
from app.config import settings

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def mcp(empty_catalog, monkeypatch):
    """Call MCP methods on the running app. Returns (call, client)."""
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(mcp_server, "embed_query", lambda query: unit(EMBEDDING_SIZE, 1).tolist())
    with TestClient(main.app, base_url="http://localhost:8000") as client:

        def call(method: str, params: dict | None = None, path: str = "/mcp"):
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
            return client.post(path, json=body, headers=HEADERS)

        yield call, client


def tool(call, name: str, **arguments):
    """Returns the raw result and the tool's output: a list's in structuredContent, a dict's
    in the text."""
    result = call("tools/call", {"name": name, "arguments": arguments}).json()["result"]
    if result.get("isError"):
        return result, None
    structured = result.get("structuredContent") or {}
    return result, structured["result"] if "result" in structured else json.loads(
        result["content"][0]["text"]
    )


def test_the_server_introduces_itself_and_lists_its_tools(mcp):
    call, _ = mcp
    initialize = call("initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"},
    })

    assert initialize.status_code == 200
    assert initialize.json()["result"]["serverInfo"]["name"] == "Tonight's Pick"
    names = {entry["name"] for entry in call("tools/list").json()["result"]["tools"]}
    assert names == {"search_titles", "pick_film", "find_films", "film_details"}


def test_the_endpoint_works_with_or_without_a_trailing_slash(mcp):
    call, _ = mcp

    assert call("tools/list", path="/mcp").status_code == 200
    assert call("tools/list", path="/mcp/").status_code == 200


def test_tools_search_pick_and_describe_films(mcp):
    call, _ = mcp

    _, matches = tool(call, "search_titles", title="matrix")
    assert [film["movie_id"] for film in matches] == [1]
    _, picked = tool(call, "pick_film", favourite_ids=[1])
    assert picked["film"]["title"] == "Blade Runner"
    _, found = tool(call, "find_films", favourite_ids=[1], genres=["Comedy"])
    assert {film["title"] for film in found} == {"Notting Hill", "100% Wolf"}
    _, details = tool(call, "film_details", movie_id=2)
    assert details["cast"] == ["Harrison Ford", "Rutger Hauer"]


def test_tools_report_bad_input_as_errors(mcp):
    call, _ = mcp

    missing, _ = tool(call, "film_details", movie_id=999)
    assert missing["isError"] and "no film with ID 999" in missing["content"][0]["text"]
    assert tool(call, "pick_film", favourite_ids=[])[0]["isError"]
    assert tool(call, "find_films", favourite_ids=[1], genres=["Not a genre"])[0]["isError"]


def test_requests_for_other_host_names_are_refused(mcp):
    _, client = mcp
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

    response = client.post("/mcp", json=body, headers={**HEADERS, "Host": "attacker.example"})

    assert response.status_code in (400, 421)


def test_a_configured_public_host_is_accepted(empty_catalog, monkeypatch):
    load_sample_catalog(empty_catalog)
    monkeypatch.setattr(settings, "public_hosts", "pick.example.com")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    with TestClient(main.app, base_url="https://pick.example.com") as client:
        assert client.post("/mcp", json=body, headers=HEADERS).status_code == 200


def test_mcp_shares_the_per_minute_limit(mcp, monkeypatch):
    call, _ = mcp
    monkeypatch.setattr(settings, "requests_per_minute", 1)

    assert call("tools/list").status_code == 200
    assert call("tools/list").status_code == 429
