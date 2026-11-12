-- Each film's IMDb rating out of 10 and Rotten Tomatoes Tomatometer score in percent, with its
-- Rotten Tomatoes page, such as m/matrix, from pipeline.review_scores. NULL when unknown.
ALTER TABLE films ADD COLUMN IF NOT EXISTS imdb_rating real;
ALTER TABLE films ADD COLUMN IF NOT EXISTS rotten_tomatoes smallint;
ALTER TABLE films ADD COLUMN IF NOT EXISTS rotten_tomatoes_id text;
