import logging
from typing import Annotated, Literal
from uuid import UUID

import anthropic
import psycopg
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import assistant, conversations
from app.catalog_tools import GENRES, film_cards
from app.config import settings
from app.db import vector_extension_version
from app.recommend import pick_film, search_films

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Tonight's Pick API", version="0.1.0")

MAX_USER_TURNS = 20


class FilmMatch(BaseModel):
    movie_id: int
    title: str
    year: int | None
    directors: list[str]


class PickRequest(BaseModel):
    liked: list[int] = Field(min_length=1, max_length=50)
    rejected: list[int] = Field(default_factory=list, max_length=500)


class PickedFilm(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    directors: list[str]
    runtime: int | None
    certification: str | None
    wikipedia_url: str | None
    where_to_watch_url: str


class PickReason(BaseModel):
    similar_to: str
    themes: list[str]


class PickResponse(BaseModel):
    film: PickedFilm
    reason: PickReason


class Preferences(BaseModel):
    avoid_genres: list[Literal[GENRES]] = Field(default_factory=list)  # type: ignore[valid-type]
    max_runtime: int | None = Field(default=None, ge=30, le=600)


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=500)
    favourites: list[int] = Field(min_length=1, max_length=50)
    preferences: Preferences = Field(default_factory=Preferences)
    reject_movie_id: int | None = None


class ChatPick(BaseModel):
    film: PickedFilm
    reason: str


class ProfileProposal(BaseModel):
    kind: Literal["avoid_genre", "max_runtime"]
    genre: str | None
    minutes: int | None
    reason: str


class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str
    pick: ChatPick | None
    proposal: ProfileProposal | None
    cost_usd: float
    conversation_cost_usd: float


@app.exception_handler(psycopg.OperationalError)
def database_unavailable(_: Request, __: psycopg.OperationalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "The database isn't available."})


@app.get("/api/health")
def health(response: Response) -> dict[str, str | bool | None]:
    try:
        pgvector = vector_extension_version()
    except psycopg.OperationalError:
        response.status_code = 503
        return {"status": "unavailable", "database": False, "pgvector": None}

    if pgvector is None:
        response.status_code = 503
        return {"status": "unavailable", "database": True, "pgvector": None}

    return {"status": "ok", "database": True, "pgvector": pgvector}


@app.get("/api/films/search")
def search(
    q: Annotated[str, Query(max_length=100)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> list[FilmMatch]:
    return search_films(q, limit)


@app.post("/api/pick")
def pick(request: PickRequest) -> PickResponse:
    result = pick_film(request.liked, request.rejected)
    if result is None:
        raise HTTPException(status_code=404, detail="No film is available to pick.")
    return result


@app.get("/api/config")
def config() -> dict[str, bool]:
    return {"conversation": settings.anthropic_api_key is not None}


def _label(card: dict) -> str:
    return f"{card['title']} ({card['year']})" if card["year"] else card["title"]


def _describe_preferences(preferences: dict) -> str:
    parts = []
    if preferences["avoid_genres"]:
        parts.append(f"avoid {', '.join(preferences['avoid_genres'])}")
    if preferences["max_runtime"]:
        parts.append(f"nothing over {preferences['max_runtime']} minutes")
    return f"My saved preferences: {'; '.join(parts)}." if parts else "I have no saved preferences."


def compose_message(conversation: conversations.Conversation, request: ChatRequest) -> str:
    """Turn the request into the person's next message, noting anything that changed.

    Changes go into the new message rather than earlier ones, so the history stays append-only.
    """
    state = conversation.state
    cards = film_cards(request.favourites + ([request.reject_movie_id] if request.reject_movie_id else []))
    favourites = [movie_id for movie_id in dict.fromkeys(request.favourites) if movie_id in cards]
    if not favourites:
        raise HTTPException(status_code=422, detail="None of those favourites are in the catalog.")

    lines = []
    if favourites != state["liked"]:
        names = ", ".join(_label(cards[movie_id]) for movie_id in favourites)
        lines.append(f"My favourite films{' are now' if state['liked'] else ''}: {names}.")
        state["liked"] = favourites

    preferences = request.preferences.model_dump()
    if preferences != state["preferences"]:
        lines.append(_describe_preferences(preferences))
        state["preferences"] = preferences

    if request.reject_movie_id is not None:
        if request.reject_movie_id not in state["presented"]:
            raise HTTPException(status_code=422, detail="You can only turn down a film you were shown.")
        if request.reject_movie_id not in state["rejected"]:
            state["rejected"].append(request.reject_movie_id)
        lines.append(f"Not that one: {_label(cards[request.reject_movie_id])}.")

    lines.append(request.message.strip())
    return "\n\n".join(lines)


@app.post("/api/chat")
def chat(request: ChatRequest) -> ChatResponse:
    if settings.anthropic_api_key is None:
        raise HTTPException(status_code=503, detail="Chat with Claude isn't set up on this server.")

    if request.conversation_id is None:
        conversation = conversations.Conversation()
    else:
        conversation = conversations.load(request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="That conversation no longer exists.")
    if conversation.state["user_turns"] >= MAX_USER_TURNS:
        raise HTTPException(status_code=409, detail="This conversation is full. Start a new one.")

    user_text = compose_message(conversation, request)
    try:
        outcome = assistant.run_turn(
            conversation, user_text, create=assistant.claude_create, embed=assistant.embed_query
        )
    except anthropic.AuthenticationError as error:
        raise HTTPException(status_code=503, detail="The server's Claude API key was rejected.") from error
    except anthropic.RateLimitError as error:
        raise HTTPException(status_code=429, detail="Claude is busy. Try again in a minute.") from error
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
        raise HTTPException(status_code=502, detail="Claude couldn't be reached. Try again.") from error

    if not conversations.save(conversation):
        raise HTTPException(
            status_code=409, detail="This conversation changed in another tab. Reload to continue."
        )

    pick = None
    if outcome.pick is not None:
        card = film_cards([outcome.pick["movie_id"]])[outcome.pick["movie_id"]]
        pick = ChatPick(film=PickedFilm(**card), reason=outcome.pick["reason"])
    return ChatResponse(
        conversation_id=conversation.id,
        reply=outcome.reply,
        pick=pick,
        proposal=ProfileProposal(**outcome.proposal) if outcome.proposal else None,
        cost_usd=round(outcome.cost_usd, 6),
        conversation_cost_usd=round(conversation.state["cost_usd"], 6),
    )
