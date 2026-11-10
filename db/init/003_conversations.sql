ALTER TABLE films ADD COLUMN IF NOT EXISTS description text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY,
    version integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    messages jsonb NOT NULL,
    state jsonb NOT NULL
);
