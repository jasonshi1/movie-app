"""Run the 75 written requests against Claude and check every pick against the person's rules.

This sends real requests and spends API credit, so it only starts with --yes.

    python -m pipeline.evaluate_conversation --limit 5 --yes   # try a few cases first
    python -m pipeline.evaluate_conversation --yes             # all 75
"""

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import anthropic
import psycopg
from fastapi import HTTPException

from app import assistant
from app.config import settings
from app.conversations import Conversation
from app.main import ChatRequest, compose_message
from pipeline.conversation_cases import CASES, Case
from pipeline.evaluate import RESULTS_DIR

# Words that often give away a plot. A hit is flagged for a person to read, not failed.
SPOILER_WORDS = ("twist", "ending", " dies", "death of", "turns out", "killer is", "spoiler")
_TITLE_WITH_YEAR = re.compile(r"^(?P<title>.+) \((?P<year>\d{4})\)$")


@dataclass
class TurnRecord:
    message: str
    reply: str
    pick: int | None
    reason: str | None
    stopped: str
    cost_usd: float
    seconds: float
    tool_errors: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    turns: list[TurnRecord]
    failures: list[str]
    flags: list[str]
    error: str | None = None


def resolve_favourites(conn: psycopg.Connection, titles: tuple[str, ...]) -> list[int]:
    ids = []
    for name in titles:
        match = _TITLE_WITH_YEAR.match(name)
        title, year = (match["title"], int(match["year"])) if match else (name, None)
        rows = conn.execute(
            "SELECT movie_id FROM films WHERE title = %s AND (%s::integer IS NULL OR year = %s)",
            (title, year, year),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(f"{name!r} matches {len(rows)} films; add the year to the case")
        ids.append(rows[0][0])
    return ids


def film_facts(conn: psycopg.Connection, movie_ids: list[int]) -> dict[int, dict]:
    rows = conn.execute(
        "SELECT movie_id, title, year, runtime, genres, certification FROM films"
        " WHERE movie_id = ANY(%s)",
        (movie_ids,),
    ).fetchall()
    return {
        row[0]: dict(zip(("movie_id", "title", "year", "runtime", "genres", "certification"), row))
        for row in rows
    }


def run_case(
    case: Case,
    favourites: list[int],
    create: Callable,
    embed: Callable[[str], list[float]],
) -> list[TurnRecord]:
    """Play the case's messages through the same code path the chat API uses."""
    conversation = Conversation()
    records: list[TurnRecord] = []
    for turn in case.turns:
        previous = records[-1].pick if records else None
        request = ChatRequest(
            message=turn.message,
            favourites=favourites,
            reject_movie_id=previous if turn.reject and previous is not None else None,
        )
        text = compose_message(conversation, request)
        started = time.perf_counter()
        outcome = assistant.run_turn(conversation, text, create=create, embed=embed)
        records.append(
            TurnRecord(
                message=turn.message,
                reply=outcome.reply,
                pick=outcome.pick["movie_id"] if outcome.pick else None,
                reason=outcome.pick["reason"] if outcome.pick else None,
                stopped=outcome.stopped,
                cost_usd=outcome.cost_usd,
                seconds=time.perf_counter() - started,
                tool_errors=outcome.tool_errors,
            )
        )
    return records


def check_case(
    case: Case, records: list[TurnRecord], facts: dict[int, dict], favourites: list[int]
) -> tuple[list[str], list[str]]:
    """Return (failed rules, flags for a person to review). No failures means the case passed."""
    expect = case.expect
    final = records[-1]
    failures: list[str] = []
    flags = [
        f"possible spoiler: {word.strip()!r}"
        for word in SPOILER_WORDS
        if word in f" {final.reason or ''} {final.reply}".lower()
    ]

    if expect.no_pick:
        if final.pick is not None:
            failures.append("picked a film although nothing can fit")
        if not final.reply.strip():
            failures.append("gave no explanation")
        return failures, flags

    if final.pick is None:
        return ["no film was picked"], flags
    film = facts.get(final.pick)
    if film is None:
        return [f"picked {final.pick}, which isn't in the catalog"], flags

    if final.pick in favourites:
        failures.append("picked one of the favourites")
    if final.pick in [record.pick for record in records[:-1]]:
        failures.append("showed a film that was already shown")
    if expect.max_runtime is not None and (film["runtime"] or 10**6) > expect.max_runtime:
        failures.append(f"runtime {film['runtime']} is over {expect.max_runtime}")
    if expect.min_year is not None and (film["year"] or 0) < expect.min_year:
        failures.append(f"year {film['year']} is before {expect.min_year}")
    if expect.max_year is not None and (film["year"] or 10**6) > expect.max_year:
        failures.append(f"year {film['year']} is after {expect.max_year}")
    if expect.genres_any and not set(expect.genres_any) & set(film["genres"]):
        failures.append(f"genres {film['genres']} include none of {list(expect.genres_any)}")
    if expect.genres_none and set(expect.genres_none) & set(film["genres"]):
        failures.append(f"genres {film['genres']} include one of {list(expect.genres_none)}")
    if expect.certifications and film["certification"] not in expect.certifications:
        failures.append(f"rating {film['certification']} isn't one of {list(expect.certifications)}")
    if expect.shorter_than_rejected:
        rejected = facts.get(records[-2].pick) if records[-2].pick else None
        if rejected is None or (film["runtime"] or 10**6) >= (rejected["runtime"] or 0):
            failures.append("the new pick isn't shorter than the rejected one")
    return failures, flags


def summarize(results: list[CaseResult]) -> dict:
    turns = [turn for result in results for turn in result.turns]
    total_cost = sum(turn.cost_usd for turn in turns)
    return {
        "cases": len(results),
        "passed": sum(1 for result in results if not result.failures and not result.error),
        "errors": sum(1 for result in results if result.error),
        "flagged": sum(1 for result in results if result.flags),
        "invented_films": sum(
            1 for result in results for failure in result.failures if "isn't in the catalog" in failure
        ),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_case_usd": round(total_cost / len(results), 4) if results else 0.0,
        "median_turn_seconds": round(sorted(t.seconds for t in turns)[len(turns) // 2], 2) if turns else 0.0,
    }


def report(results: list[CaseResult], summary: dict) -> str:
    lines = [
        "# Stage 2 conversation results",
        "",
        f"Model: {settings.claude_model}, effort {settings.claude_effort}.",
        "",
        f"- Passed every rule: **{summary['passed']} of {summary['cases']}**",
        f"- Errors: {summary['errors']}",
        f"- Invented films: {summary['invented_films']}",
        f"- Flagged for possible spoilers: {summary['flagged']}",
        f"- Total cost: ${summary['total_cost_usd']:.2f} (${summary['cost_per_case_usd']:.3f} per case)",
        f"- Median time per turn: {summary['median_turn_seconds']} s",
        "",
        "## Failures and flags",
        "",
    ]
    problems = [result for result in results if result.failures or result.flags or result.error]
    for result in problems:
        detail = result.error or "; ".join(result.failures + result.flags)
        lines.append(f"- `{result.id}`: {detail}")
    if not problems:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--case", action="append", help="run only this case id (repeatable)")
    parser.add_argument("--yes", action="store_true", help="confirm that this spends API credit")
    args = parser.parse_args()

    if settings.anthropic_api_key is None:
        sys.exit("ANTHROPIC_API_KEY isn't set, so there is nothing to evaluate against.")
    cases = [case for case in CASES if not args.case or case.id in args.case][: args.limit]
    if not args.yes:
        sys.exit(f"This sends {len(cases)} cases to {settings.claude_model}, usually 2 to 4 "
                 "requests each, and spends API credit. Re-run with --yes to start.")

    results = []
    with psycopg.connect(settings.database_url) as conn:
        for case in cases:
            favourites = resolve_favourites(conn, case.favourites)
            try:
                records = run_case(case, favourites, assistant.claude_create, assistant.embed_query)
            # One failed case shouldn't lose the rest of the run; a bug still stops it.
            except (anthropic.APIError, psycopg.Error, HTTPException) as error:
                results.append(CaseResult(case.id, [], [], [], error=f"{type(error).__name__}: {error}"))
                print(f"{case.id}: error {error}", flush=True)
                continue
            facts = film_facts(conn, [record.pick for record in records if record.pick])
            failures, flags = check_case(case, records, facts, favourites)
            results.append(CaseResult(case.id, records, failures, flags))
            status = "pass" if not failures else "FAIL " + "; ".join(failures)
            print(f"{case.id}: {status} (${sum(r.cost_usd for r in records):.3f})", flush=True)

    summary = summarize(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    text = report(results, summary)
    (RESULTS_DIR / "stage2_conversation.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "stage2_conversation.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
