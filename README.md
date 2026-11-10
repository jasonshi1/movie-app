# Tonight's Pick

A movie picker for one person. Name a few films you love and get one film to watch tonight, with the reason it was chosen. Say "not that one" and it tries again.

This project is in development. The current version recommends without an LLM. A Claude conversation comes next.

## Run it

You need Docker. From the project root:

```bash
docker compose up --build
```

- App: http://localhost:5173
- API health check: http://localhost:8000/api/health

The catalog starts empty. To fill it, build the data once with Python 3.13. This downloads about 2.2 GB and takes around 20 minutes:

```bash
docker compose up -d --wait db
cd backend
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,pipeline]"
.venv/Scripts/python -m pipeline.build
```

## How picks are chosen

Your favourites are averaged into a taste profile across 1,084 Tag Genome themes. Each film is scored by its similarity to that profile, plus a weighted popularity score.

To reproduce the comparison of methods:

```bash
.venv/Scripts/python -m pipeline.evaluate
```

Each test user's first 20 liked films are the possible favourites, and everything they rated afterwards is hidden. Results on 5,000 users who played no part in choosing the method, at 5 favourites:

| Method | NDCG@10 | Distinct first picks |
|---|---:|---:|
| Tag Genome + popularity (used) | 0.298 | 82 |
| Text embedding + popularity | 0.291 | 59 |
| Popularity alone | 0.275 | 15 |
| Tag Genome alone | 0.161 | 660 |
| Text embedding alone | 0.117 | 829 |
| Genre match | 0.073 | 628 |

Full results are in [results/stage1_baselines.md](results/stage1_baselines.md).

### Known limits

- With only one favourite, popularity alone scores higher (0.275 against 0.257).
- First picks lean toward well-known films, because people mostly rate films they've heard of and that rewards popularity in this kind of test.
- The catalog only covers films up to 2014, and age ratings are known for just 23% of films.

## Develop

Run the tests with the database up:

```bash
cd backend
.venv/Scripts/ruff check .
.venv/Scripts/pytest
```

```bash
cd frontend
npm ci
npm run typecheck
npm test
```

## Data and credits

This is a free, non-commercial project.

- [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and the [Tag Genome 2021](https://grouplens.org/datasets/movielens/tag-genome-2021/) by GroupLens. Harper and Konstan (2015), The MovieLens Datasets: History and Context, ACM TiiS 5(4). Kotkov et al. (2021), Revisiting the tag relevance prediction problem, SIGIR. Vig et al. (2012), The tag genome, ACM TiiS 2(3). The Tag Genome is licensed under CC BY-NC 3.0.
- [Wikidata](https://www.wikidata.org/) (CC0) and [Wikipedia](https://en.wikipedia.org/) (CC BY-SA).

GroupLens and the University of Minnesota do not endorse this project.

## Roadmap

- [x] Stage 0: Foundation
- [x] Stage 1: Catalog and baseline recommendations, no LLM
- [ ] Stage 2: Claude conversation
- [ ] Stage 3: PyTorch ranking model
- [ ] Stage 4: Simulated-user evaluation
- [ ] Stage 5: Public launch
