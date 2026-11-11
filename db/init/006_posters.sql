-- Each film's poster, the lead image of its English Wikipedia article, from pipeline.posters.
-- NULL when the article has no image or the film has no article.
ALTER TABLE films ADD COLUMN IF NOT EXISTS poster_url text;
