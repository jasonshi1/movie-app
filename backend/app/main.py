import logging
import math
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import anthropic
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from app import assistant, conversations, limits, mcp_server
from app.catalog_tools import GENRES, film_cards
from app.config import settings
from app.db import newest_film_year, vector_extension_version
from app.recommend import pick_films, search_films

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The app only loads its own files and calls its own API.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'self'"
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        deleted = conversations.delete_older_than(settings.conversation_retention_days)
        logger.info("Deleted %d conversations older than %d days", deleted,
                    settings.conversation_retention_days)
    except psycopg.OperationalError:
        logger.warning("Couldn't delete old conversations: the database isn't available.")
    async with mcp_server.endpoint.running():
        yield


app = FastAPI(title="Tonight's Pick API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next: Callable) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response

MAX_USER_TURNS = 20


def visitor(request: Request) -> str:
    return limits.visitor_address(request.headers, request.client.host if request.client else None)


def rate_limited(bucket: str, limit: Callable[[], limits.Limit]) -> Callable[[Request], None]:
    def check(request: Request) -> None:
        wait = limits.limiter.wait_seconds(bucket, visitor(request), limit())
        if wait > 0:
            raise HTTPException(
                status_code=429,
                detail="You're going a bit fast. Try again in a moment.",
                headers={"Retry-After": str(math.ceil(wait))},
            )

    return check


per_minute = Depends(rate_limited("requests", limits.request_limit))
per_hour_of_chat = Depends(rate_limited("chat", limits.chat_limit))


class FilmMatch(BaseModel):
    movie_id: int
    title: str
    year: int | None
    directors: list[str]


class PickRequest(BaseModel):
    liked: list[int] = Field(min_length=1, max_length=50)
    rejected: list[int] = Field(default_factory=list, max_length=500)
    count: int = Field(default=3, ge=1, le=5)
    # Picks still on screen when one is replaced. They're left out, and so are their sequels.
    showing: list[int] = Field(default_factory=list, max_length=5)
    # How many of the picks, the last ones, come from films released in the catalog's last few
    # years (recommend.RECENT_YEARS).
    recent: int = Field(default=0, ge=0, le=5)

    @model_validator(mode="after")
    def recent_picks_fit_in_count(self) -> "PickRequest":
        if self.recent > self.count:
            raise ValueError("recent can't be more than count")
        return self


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


class Pick(BaseModel):
    film: PickedFilm
    reason: PickReason
    # Released in the catalog's last few years, whichever slot it was picked for.
    recent: bool


class PickResponse(BaseModel):
    # Best first. Fewer than asked for when the catalog runs out.
    picks: list[Pick]


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


@app.get("/api/films/search", dependencies=[per_minute])
def search(
    q: Annotated[str, Query(max_length=100)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> list[FilmMatch]:
    return search_films(q, limit)


@app.post("/api/pick", dependencies=[per_minute])
def pick(request: PickRequest) -> PickResponse:
    picks = pick_films(
        request.liked, request.rejected, request.count, request.showing, request.recent
    )
    if not picks:
        raise HTTPException(status_code=404, detail="No film is available to pick.")
    return PickResponse(picks=picks)


@app.get("/api/config")
def config() -> dict[str, bool | int | None]:
    try:
        newest_year = newest_film_year()
    except psycopg.OperationalError:
        newest_year = None
    return {"conversation": settings.anthropic_api_key is not None, "newest_year": newest_year}


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


@app.post("/api/chat", dependencies=[per_hour_of_chat])
def chat(request: ChatRequest) -> ChatResponse:
    if settings.anthropic_api_key is None:
        raise HTTPException(status_code=503, detail="Chat with Claude isn't set up on this server.")
    if not limits.budget_left():
        raise HTTPException(
            status_code=503,
            detail="Chat has used today's budget and is back at midnight UTC. Quick picks still work.",
        )

    if request.conversation_id is None:
        conversation = conversations.Conversation()
    else:
        conversation = conversations.load(request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="That conversation no longer exists.")
    if conversation.state["user_turns"] >= MAX_USER_TURNS:
        raise HTTPException(status_code=409, detail="This conversation is full. Start a new one.")

    user_text = compose_message(conversation, request)
    cost_before = conversation.state["cost_usd"]
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
    finally:
        # Every finished API call is counted, including those in a turn that later failed.
        limits.record_spend(conversation.state["cost_usd"] - cost_before)

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


def serve_frontend(application: FastAPI, directory: Path) -> bool:
    """Serve the built app from the same address as the API, if it has been built.

    Mounted last, so the API and /mcp take precedence.
    """
    if not (directory / "index.html").is_file():
        return False
    application.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    return True


class McpWithoutSlash:
    """Accept /mcp as well as /mcp/: clients are usually given the address without the slash."""

    def __init__(self, application: ASGIApp):
        self.app = application

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        await self.app(scope, receive, send)


app.add_middleware(McpWithoutSlash)
app.mount("/mcp", mcp_server.endpoint)
serve_frontend(app, Path(settings.static_dir))
