"""Store each conversation's full history so it can be replayed to Claude unchanged."""

import uuid
from dataclasses import dataclass, field

import psycopg
from psycopg.types.json import Jsonb

from app.config import settings


def _new_state() -> dict:
    return {
        "liked": [],
        "rejected": [],
        "rejection_reasons": {},
        "presented": [],
        "searched": [],
        "preferences": {"avoid_genres": [], "max_runtime": None},
        "user_turns": 0,
        "cost_usd": 0.0,
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
    }


@dataclass
class Conversation:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    version: int = 0
    messages: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=_new_state)


def load(conversation_id: uuid.UUID) -> Conversation | None:
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        row = conn.execute(
            "SELECT version, messages, state FROM conversations WHERE id = %s", (conversation_id,)
        ).fetchone()
    if row is None:
        return None
    return Conversation(id=conversation_id, version=row[0], messages=row[1], state=row[2])


def save(conversation: Conversation) -> bool:
    """Write the conversation unless another request saved it first. Returns False if it did."""
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        if conversation.version == 0:
            saved = conn.execute(
                "INSERT INTO conversations (id, version, messages, state) VALUES (%s, 1, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (conversation.id, Jsonb(conversation.messages), Jsonb(conversation.state)),
            ).rowcount
        else:
            saved = conn.execute(
                "UPDATE conversations SET version = version + 1, messages = %s, state = %s,"
                " updated_at = now() WHERE id = %s AND version = %s",
                (
                    Jsonb(conversation.messages),
                    Jsonb(conversation.state),
                    conversation.id,
                    conversation.version,
                ),
            ).rowcount
    if saved:
        conversation.version += 1
    return bool(saved)
