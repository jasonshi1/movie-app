-- What Claude has cost per UTC day, checked before every chat turn.
CREATE TABLE IF NOT EXISTS daily_spend (
    day date PRIMARY KEY,
    cost_usd double precision NOT NULL DEFAULT 0
);
