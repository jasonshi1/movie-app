CREATE TABLE IF NOT EXISTS films (
    movie_id integer PRIMARY KEY,
    imdb_id text NOT NULL,
    title text NOT NULL,
    year integer,
    genres text[] NOT NULL,
    directors text[] NOT NULL,
    cast_members text[] NOT NULL,
    runtime integer,
    certification text,
    wikipedia_title text,
    popularity real NOT NULL,
    genome vector(1084) NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS films_title_trgm ON films USING gin (title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS genome_tags (
    position integer PRIMARY KEY,
    tag text NOT NULL,
    displayable boolean NOT NULL
);
