"""A stand-in for Claude in tests. It replays prepared replies and never calls the API."""

import copy
import json

from anthropic.types.beta import BetaMessage

SEARCH_EVERYTHING = {
    "query": "",
    "genres": [],
    "exclude_genres": [],
    "max_runtime": None,
    "min_year": None,
    "max_year": None,
    "certifications": None,
}


def message(*blocks, stop_reason="end_turn", model="claude-opus-5-5", **usage) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 100),
                "output_tokens": usage.get("output_tokens", 20),
                "cache_read_input_tokens": usage.get("cache_read", 0),
                "cache_creation_input_tokens": usage.get("cache_write", 0),
            },
        }
    )


def thinking(signature: str) -> dict:
    return {"type": "thinking", "thinking": "", "signature": signature}


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def call(tool_id: str, name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def search(tool_id: str, **filters) -> BetaMessage:
    return message(call(tool_id, "search_films", {**SEARCH_EVERYTHING, **filters}), stop_reason="tool_use")


def pick(tool_id: str, movie_id: int, reason: str = "It fits.") -> BetaMessage:
    return message(
        call(tool_id, "present_pick", {"movie_id": movie_id, "reason": reason}), stop_reason="tool_use"
    )


def tool_results(messages: list[dict]) -> list[dict]:
    """The results Claude would read next, each parsed and marked with is_error."""
    return [
        {**json.loads(block["content"]), "is_error": block.get("is_error", False)}
        for block in messages[-1]["content"]
    ]


class ScriptedClaude:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> BetaMessage:
        self.requests.append(copy.deepcopy(messages))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply
