"""Simulated users: how many suggestions does it take to find a film someone likes?

Each simulated user is a real MovieLens user from the test split, which no earlier stage used.
They give the app their first 5 liked films as favourites, then accept a suggestion only if
they rated it 4 or higher later on. Anything else they turn down. A film they rated before
they're shown gets "I've already seen that one", because the app can't know what they've
watched. Any other film gets a reason built from the Tag Genome themes where it differs most
from everything they liked, such as "less violence and more humour".

Two ways of suggesting films are compared on the same users:

- Quick picks: each "Not that one" moves down the ranker's list. Reasons can't be given.
- The conversation: Claude reads each reason and searches again. This needs an API key.

    python -m pipeline.simulate                             # quick picks only, no API calls
    python -m pipeline.simulate --claude --limit 10 --yes   # Claude, on the first 10 users

Unrated films count as misses, so these rates understate how often people would be happy.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import anthropic
import numpy as np
import polars as pl
import psycopg
from fastapi import HTTPException

from app import assistant
from app.config import settings
from app.conversations import Conversation
from app.main import ChatRequest, compose_message
from app.recommend import POPULARITY_WEIGHT
from pipeline.catalog import PROCESSED_DIR, film_popularity
from pipeline.embed import normalize
from pipeline.evaluate import (
    RESULTS_DIR,
    EvalUser,
    Scorer,
    build_users,
    popularity_scorer,
    ranker_scorer,
    similarity_scorer,
)
from pipeline.load_db import load_ranker
from pipeline.tag_policy import is_displayable

FAVOURITES = 5
MAX_SUGGESTIONS = 10
HIT_WITHIN = (1, 3, 5, 10)
HEADLINE_WITHIN = 5
# Run sizes to price after a trial: about ±6 and ±3 points of confidence on the headline rate.
PROJECTED_USERS = (200, 1000)
SIMULATED_USERS = 5000
SEED = 20261110
OPENING = "Pick a film for me to watch tonight."
NUDGE = "Just pick one for me, please."
# Turns in a row without a pick before the simulated user gives up.
MAX_TURNS_WITHOUT_PICK = 2


@dataclass(frozen=True)
class Catalog:
    movie_ids: np.ndarray
    genome: np.ndarray  # raw Tag Genome scores, 0 to 1
    tag_positions: np.ndarray  # columns of the tags that may be shown to people
    tag_names: list[str]
    rows: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.rows.update({movie_id: row for row, movie_id in enumerate(self.movie_ids.tolist())})


@dataclass(frozen=True)
class Response:
    kind: str  # "hit", "seen" or "reason"
    message: str


@dataclass(frozen=True)
class SimulatedUser:
    favourites: np.ndarray  # catalog rows
    seen: frozenset[int]  # rows rated up to the last favourite, favourites included
    liked_later: frozenset[int]  # rows rated 4 or higher after the last favourite
    taste: np.ndarray  # mean Tag Genome scores of every film they liked

    @classmethod
    def from_eval_user(cls, user: EvalUser, catalog: Catalog) -> "SimulatedUser":
        liked = np.concatenate([user.favourites, user.hidden])
        return cls(
            favourites=user.favourites,
            seen=frozenset(user.seen.tolist()),
            liked_later=frozenset(user.hidden.tolist()),
            taste=catalog.genome[liked].mean(axis=0),
        )

    def respond(self, row: int, catalog: Catalog) -> Response:
        if row in self.liked_later:
            return Response("hit", "Yes, that's the one.")
        if row in self.seen:
            return Response("seen", "I've already seen that one.")
        positions = catalog.tag_positions
        difference = catalog.genome[row, positions] - self.taste[positions]
        less = catalog.tag_names[int(difference.argmax())]
        more = catalog.tag_names[int(difference.argmin())]
        return Response("reason", f"Something with less {less} and more {more}, please.")


@dataclass
class Session:
    """What happened for one simulated user. hit_at counts suggestions, starting at 1."""

    suggestions: list[int]
    responses: list[str]
    hit_at: int | None
    cost_usd: float = 0.0
    replies: list[str] = field(default_factory=list)
    error: str | None = None


def walk_list(ranked: np.ndarray, user: SimulatedUser, catalog: Catalog) -> Session:
    """Quick picks: each turned-down film moves to the next one on the list."""
    suggestions, responses = [], []
    for row in ranked[:MAX_SUGGESTIONS].tolist():
        response = user.respond(row, catalog)
        suggestions.append(row)
        responses.append(response.kind)
        if response.kind == "hit":
            return Session(suggestions, responses, hit_at=len(suggestions))
    return Session(suggestions, responses, hit_at=None)


def ranked_lists(scorer: Scorer, users: list[SimulatedUser], batch: int = 256) -> np.ndarray:
    """Each user's top films, best first, leaving out their favourites as the app does."""
    lists = []
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        scores = scorer([user.favourites for user in chunk]).astype(np.float64)
        for index, user in enumerate(chunk):
            scores[index, user.favourites] = -np.inf
        top = np.argpartition(-scores, MAX_SUGGESTIONS, axis=1)[:, :MAX_SUGGESTIONS]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        lists.append(np.take_along_axis(top, order, axis=1))
    return np.concatenate(lists)


def converse(
    user: SimulatedUser,
    catalog: Catalog,
    create: Callable,
    embed: Callable[[str], list[float]],
) -> Session:
    """The conversation, through the same code path the chat API uses."""
    conversation = Conversation()
    favourites = [int(catalog.movie_ids[row]) for row in user.favourites]
    session = Session(suggestions=[], responses=[], hit_at=None)
    message, reject, without_pick = OPENING, None, 0

    while len(session.suggestions) < MAX_SUGGESTIONS:
        request = ChatRequest(message=message, favourites=favourites, reject_movie_id=reject)
        outcome = assistant.run_turn(
            conversation, compose_message(conversation, request), create=create, embed=embed
        )
        session.cost_usd += outcome.cost_usd
        session.replies.append(outcome.reply)
        if outcome.pick is None:
            without_pick += 1
            if without_pick >= MAX_TURNS_WITHOUT_PICK:
                break
            message, reject = NUDGE, None
            continue

        without_pick = 0
        movie_id = outcome.pick["movie_id"]
        response = user.respond(catalog.rows[movie_id], catalog)
        session.suggestions.append(catalog.rows[movie_id])
        session.responses.append(response.kind)
        if response.kind == "hit":
            session.hit_at = len(session.suggestions)
            break
        message, reject = response.message, movie_id
    return session


def proportion(hits: np.ndarray) -> tuple[float, float]:
    """Share of users and its 95% confidence half-width."""
    share = float(hits.mean())
    return share, float(1.96 * np.sqrt(share * (1 - share) / hits.size))


def paired_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Mean per-user difference between two methods and its 95% confidence half-width."""
    difference = first.astype(np.float64) - second.astype(np.float64)
    return float(difference.mean()), float(1.96 * difference.std(ddof=1) / np.sqrt(difference.size))


def hits_within(sessions: list[Session], within: int) -> np.ndarray:
    return np.array([session.hit_at is not None and session.hit_at <= within for session in sessions])


def summarize(sessions: list[Session]) -> dict:
    responses = [kind for session in sessions for kind in session.responses]
    return {
        "users": len(sessions),
        "hit_within": {n: proportion(hits_within(sessions, n)) for n in HIT_WITHIN},
        "seen_share": responses.count("seen") / len(responses) if responses else 0.0,
        "distinct_first_suggestions": len({s.suggestions[0] for s in sessions if s.suggestions}),
        "errors": sum(1 for session in sessions if session.error),
        "cost_usd": sum(session.cost_usd for session in sessions),
    }


def hit_table(summaries: dict[str, dict]) -> str:
    header = "| Method | " + " | ".join(f"Hit within {n}" for n in HIT_WITHIN) + " | Already seen |"
    rows = [
        f"| {name} | "
        + " | ".join(f"{s['hit_within'][n][0]:.1%} ± {s['hit_within'][n][1]:.1%}" for n in HIT_WITHIN)
        + f" | {s['seen_share']:.0%} |"
        for name, s in summaries.items()
    ]
    return "\n".join([header, "|---|" + "---:|" * (len(HIT_WITHIN) + 1), *rows])


def load_catalog() -> tuple[Catalog, np.ndarray, pl.DataFrame]:
    films = pl.read_parquet(PROCESSED_DIR / "films.parquet")
    tags = json.loads((PROCESSED_DIR / "tags.json").read_text(encoding="utf-8"))
    positions = np.array([index for index, tag in enumerate(tags) if is_displayable(tag)])
    catalog = Catalog(
        movie_ids=films["movie_id"].to_numpy(),
        genome=np.load(PROCESSED_DIR / "genome.npy"),
        tag_positions=positions,
        tag_names=[tags[index] for index in positions],
    )
    return catalog, film_popularity(films), films


def simulated_users(catalog: Catalog, limit: int) -> list[SimulatedUser]:
    ratings = pl.read_parquet(PROCESSED_DIR / "ratings.parquet").filter(pl.col("split") == "test")
    users = build_users(ratings, catalog.movie_ids, limit, seed=SEED, profile_size=FAVOURITES)
    return [SimulatedUser.from_eval_user(user, catalog) for user in users]


def list_methods(catalog: Catalog, popularity: np.ndarray) -> dict[str, Scorer]:
    ranker = load_ranker(catalog.movie_ids)
    if ranker is None:
        sys.exit("Train the ranker first: python -m pipeline.ranker")
    return {
        "Quick picks (ranker)": ranker_scorer(ranker),
        "Quick picks (Stage 1 method)": similarity_scorer(
            normalize(catalog.genome), popularity, POPULARITY_WEIGHT
        ),
        "Most popular films": popularity_scorer(popularity),
    }


def run_lists(catalog: Catalog, popularity: np.ndarray, users: list[SimulatedUser]) -> dict:
    return {
        name: [
            walk_list(ranked, user, catalog)
            for ranked, user in zip(ranked_lists(scorer, users), users, strict=True)
        ]
        for name, scorer in list_methods(catalog, popularity).items()
    }


def list_report(sessions: dict[str, list[Session]]) -> str:
    summaries = {name: summarize(runs) for name, runs in sessions.items()}
    ranker_name = next(iter(sessions))
    headline = summaries[ranker_name]["hit_within"][HEADLINE_WITHIN]
    comparisons = [
        f"| {name} | "
        + " | ".join(
            "{:+.1%} ± {:.1%}".format(
                *paired_difference(hits_within(sessions[ranker_name], n), hits_within(runs, n))
            )
            for n in HIT_WITHIN
        )
        + " |"
        for name, runs in sessions.items()
        if name != ranker_name
    ]
    users = summaries[ranker_name]["users"]
    return "\n".join(
        [
            "# Stage 4 simulated users",
            "",
            (f"With quick picks, {headline[0]:.1%} ± {headline[1]:.1%} of simulated users get a film "
             f"they liked within {HEADLINE_WITHIN} suggestions (95% confidence)."),
            "",
            (f"Simulated users: {users:,} real MovieLens users from the test split, which no earlier "
             f"stage used. Each gives their first {FAVOURITES} liked films as favourites and accepts "
             "only a film they rated 4 or higher later on. Films they had already rated count as "
             "turned down, because the app can't know what someone has seen."),
            "",
            "## Share of users with a hit",
            "",
            hit_table(summaries),
            "",
            "Already seen: the share of all suggestions that the user had already rated.",
            "",
            f"## {ranker_name} minus each other method, per user",
            "",
            "| Method | " + " | ".join(f"Hit within {n}" for n in HIT_WITHIN) + " |",
            "|---|" + "---:|" * len(HIT_WITHIN),
            *comparisons,
            "",
            "## Limits",
            "",
            ("- Unrated films count as misses, so every rate understates how often people would be "
            "happy. That penalizes less familiar suggestions most."),
            ("- Simulated users aren't real people. Their reasons come from Tag Genome differences, "
            "and they never change their minds. Some reasons name an actor or author, such as "
             "\"less arnold\", because those are Tag Genome tags too."),
            "",
        ]
    )


def conversation_report(claude: list[Session], quick: list[Session], seconds: float) -> str:
    summaries = {"Conversation with Claude": summarize(claude), "Quick picks (ranker)": summarize(quick)}
    difference = paired_difference(
        hits_within(claude, HEADLINE_WITHIN), hits_within(quick, HEADLINE_WITHIN)
    )
    suggestions = [len(session.suggestions) for session in claude]
    cost_per_user = sum(session.cost_usd for session in claude) / len(claude)
    return "\n".join(
        [
            "# Stage 4 simulated users with Claude",
            "",
            (f"Model: {settings.claude_model}, effort {settings.claude_effort}. Users: {len(claude)}, "
             "the first ones from the quick-picks run, so both methods face the same people."),
            "",
            hit_table(summaries),
            "",
            (f"Conversation minus quick picks, hit within {HEADLINE_WITHIN}: "
             f"{difference[0]:+.1%} ± {difference[1]:.1%} (95% confidence, per user)."),
            "",
            (f"- Cost: ${summaries['Conversation with Claude']['cost_usd']:.2f} in total, "
            f"${cost_per_user:.3f} per user"),
            f"- Suggestions per user: {np.mean(suggestions):.1f} on average",
            f"- Errors: {summaries['Conversation with Claude']['errors']}",
            f"- Time: {seconds / 60:.0f} minutes",
            (f"- At this rate, {PROJECTED_USERS[0]:,} users would cost about "
             f"${cost_per_user * PROJECTED_USERS[0]:,.0f} and {PROJECTED_USERS[1]:,} about "
             f"${cost_per_user * PROJECTED_USERS[1]:,.0f}."),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claude", action="store_true", help="also run the Claude conversation")
    parser.add_argument("--limit", type=int, default=10, help="users for the Claude run")
    parser.add_argument("--yes", action="store_true", help="confirm that this spends API credit")
    args = parser.parse_args()

    if args.claude and settings.anthropic_api_key is None:
        sys.exit("ANTHROPIC_API_KEY isn't set, so there is no Claude to simulate against.")
    if args.claude and not args.yes:
        sys.exit(f"This runs {args.limit} conversations with {settings.claude_model}, up to "
                 f"{MAX_SUGGESTIONS} suggestions each, and spends API credit. "
                 "Re-run with --yes to start.")

    catalog, popularity, _ = load_catalog()
    users = simulated_users(catalog, SIMULATED_USERS)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.claude:
        sessions = run_lists(catalog, popularity, users)
        report = list_report(sessions)
        (RESULTS_DIR / "stage4_simulation.md").write_text(report, encoding="utf-8")
        (RESULTS_DIR / "stage4_simulation.json").write_text(
            json.dumps({name: summarize(runs) for name, runs in sessions.items()}, indent=2),
            encoding="utf-8",
        )
        print(report)
        return

    users = users[: args.limit]
    started = time.perf_counter()
    claude = []
    for index, user in enumerate(users, start=1):
        try:
            session = converse(user, catalog, assistant.claude_create, assistant.embed_query)
        # One failed conversation shouldn't lose the rest of the run; a bug still stops it.
        except (anthropic.APIError, psycopg.Error, HTTPException) as error:
            session = Session([], [], None, error=f"{type(error).__name__}: {error}")
        claude.append(session)
        print(f"user {index}: hit at {session.hit_at} after {len(session.suggestions)} suggestions "
              f"(${session.cost_usd:.3f}){' ' + session.error if session.error else ''}", flush=True)

    # Saved before anything else, so a later problem can't lose a paid run.
    (RESULTS_DIR / "stage4_conversation.json").write_text(
        json.dumps({"sessions": [asdict(session) for session in claude]}, indent=2),
        encoding="utf-8",
    )
    quick = run_lists(catalog, popularity, users)["Quick picks (ranker)"]
    report = conversation_report(claude, quick, time.perf_counter() - started)
    (RESULTS_DIR / "stage4_conversation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
