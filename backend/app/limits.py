"""Keep a public server affordable: a daily cap on Claude spending, and per-visitor rate limits.

The spending total lives in the database, so it holds across restarts and server copies. Rate
limits are counted in memory by IP address, so nothing about visitors is stored. That means
each server copy counts separately and a restart clears the counts, which is fine for one
small server.
"""

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import psycopg

from app.config import settings

# Entries are pruned once this many visitors are being tracked.
MAX_TRACKED = 10_000


def spent_today() -> float:
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        row = conn.execute(
            "SELECT cost_usd FROM daily_spend WHERE day = (now() AT TIME ZONE 'utc')::date"
        ).fetchone()
    return row[0] if row else 0.0


def budget_left() -> bool:
    return spent_today() < settings.daily_spend_limit_usd


def record_spend(cost_usd: float) -> None:
    if cost_usd <= 0:
        return
    with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
        conn.execute(
            "INSERT INTO daily_spend (day, cost_usd) VALUES ((now() AT TIME ZONE 'utc')::date, %s)"
            " ON CONFLICT (day) DO UPDATE SET cost_usd = daily_spend.cost_usd + EXCLUDED.cost_usd",
            (cost_usd,),
        )


@dataclass(frozen=True)
class Limit:
    requests: int
    seconds: int


class RateLimiter:
    """Fixed-window request counts per visitor and kind of request."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, str], tuple[float, int]] = {}

    def wait_seconds(self, bucket: str, visitor: str, limit: Limit) -> float:
        """Count one request. Returns 0 if it's allowed, or how long until the window resets."""
        now = self._clock()
        key = (bucket, visitor)
        with self._lock:
            resets_at, count = self._windows.get(key, (now + limit.seconds, 0))
            if now >= resets_at:
                resets_at, count = now + limit.seconds, 0
            if count >= limit.requests:
                return resets_at - now
            self._windows[key] = (resets_at, count + 1)
            if len(self._windows) > MAX_TRACKED:
                self._windows = {k: v for k, v in self._windows.items() if v[0] > now}
        return 0.0

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()


limiter = RateLimiter()


def visitor_address(headers: Mapping[str, str], connection_host: str | None) -> str:
    """Who a request counts against: the trusted proxy header if one is set, else the
    connection's address."""
    if settings.client_ip_header:
        forwarded = headers.get(settings.client_ip_header.lower(), "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return connection_host or "unknown"


def chat_limit() -> Limit:
    return Limit(settings.chat_messages_per_hour, 3600)


def request_limit() -> Limit:
    return Limit(settings.requests_per_minute, 60)
