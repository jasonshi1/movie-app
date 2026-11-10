-- The trained ranker, filled by pipeline.load_db when data/processed/ranker.npz exists.
-- ranker_output holds a film's output vector with its bias appended, so its inner product
-- with [taste vector, 1] is the film's score. Without a ranker, picks use the Stage 1 method.
ALTER TABLE films ADD COLUMN IF NOT EXISTS ranker_input vector;
ALTER TABLE films ADD COLUMN IF NOT EXISTS ranker_output vector;

CREATE TABLE IF NOT EXISTS ranker (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    encoder bytea NOT NULL
);
