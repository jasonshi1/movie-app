# Tonight's Pick

A movie picker for one person. Name a few films you love and get one film to watch tonight, with the reason it was chosen. Say "not that one" and it tries again.

This project is in development. Quick picks come from a ranking model trained with PyTorch, with no LLM involved. A conversation with Claude is built and switches on when the server has an Anthropic API key. It hasn't been evaluated against the real model yet.

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

Then train the ranking model and load it. Install PyTorch first: the command below is for an NVIDIA GPU, and training also works on a CPU, just more slowly. On an RTX 3060 it takes a few minutes:

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev,pipeline,train]"
.venv/Scripts/python -m pipeline.ranker
.venv/Scripts/python -m pipeline.load_db
```

Without a trained model, quick picks use the Stage 1 method described below.

## Chat with Claude

To switch the conversation on, copy `.env.example` to `.env`, add your key after `ANTHROPIC_API_KEY=`, and restart with `docker compose up --build`. Set a spending limit on the key in the Anthropic Console first. Without a key, the page says chat isn't set up and quick picks keep working.

How it works:

- You describe what you want, and Claude (Claude Opus 5.5, low effort) searches the catalog with tools. Hard limits such as runtime, years, genres, age ratings, films you've seen and films you've turned down are enforced in SQL, so Claude can only choose from films that already meet them.
- Claude can only show a film that its search returned, and only one per turn. Your favourites can never be picked.
- Text from Wikipedia reaches Claude only as tool results, which it is told to treat as facts about films, never as instructions.
- If you keep turning films down for the same reason, Claude can suggest saving a preference. Nothing is saved unless you accept, and saved preferences stay in your browser, where you can remove them.
- Each conversation's full history is stored and replayed to Claude unchanged, which Claude Opus 5.5 needs to keep its reasoning valid between turns. Safety-classifier declines are retried on a fallback model.
- The cost of every turn is logged.

### Evaluating it

`pipeline/conversation_cases.py` holds 75 written requests, each with the limits its pick must meet. The runner sends them to Claude and checks every pick against the database. It spends API credit, so it only starts with `--yes`. Try a few cases first:

```bash
.venv/Scripts/python -m pipeline.evaluate_conversation --limit 5 --yes
```

Results go to `results/stage2_conversation.md`.

The weight given to the described mood was measured offline: see [results/stage2_mood_weight.md](results/stage2_mood_weight.md).

## How picks are chosen

A ranking model built with PyTorch turns your favourites into a taste vector, with no retraining for each person. Each film has two learned vectors, built from its 1,084 Tag Genome theme scores, its text embedding and a per-film correction, plus a learned bias for how widely liked it is. Your favourites' vectors are averaged, and a small network adjusts the average into your taste vector. Every film is scored against it in one database query.

The model is trained in PyTorch on a GPU. The app runs it with numpy and pgvector, so the server needs neither PyTorch nor a GPU. The reason shown with a pick still comes from Tag Genome themes, because people can read those.

Each test user's first 20 liked films are the possible favourites, and everything they rated afterwards is hidden. The model trained on 102,061 users and stopped early based on 5,000 other training users. The results below are for 5,000 validation users who played no part in training or choosing the method, at 5 favourites:

| Method | NDCG@10 | Distinct first picks |
|---|---:|---:|
| Ranking model (used) | 0.348 | 349 |
| Tag Genome + popularity (Stage 1) | 0.298 | 82 |
| Text embedding + popularity | 0.291 | 59 |
| Popularity alone | 0.275 | 15 |
| Tag Genome alone | 0.161 | 660 |
| Text embedding alone | 0.117 | 829 |
| Genre match | 0.073 | 628 |

Compared user by user, the model beats Stage 1 by 0.049 ± 0.006 at 5 favourites (95% confidence). Its lead grows with more favourites, from 0.015 at 1 favourite to 0.104 at 20. It also offers 4 times as many different first picks, so people see fewer of the same famous films.

To reproduce the results:

```bash
.venv/Scripts/python -m pipeline.evaluate
.venv/Scripts/python -m pipeline.ranker
```

Full results are in [results/stage3_ranker.md](results/stage3_ranker.md) and [results/stage1_baselines.md](results/stage1_baselines.md).

### Known limits

- With only one favourite, the model (0.272) is no better than recommending the most popular films (0.275).
- The conversation with Claude still searches with the Stage 1 taste profile. Switching it to the model means re-measuring how much the described mood should count.
- The catalog only covers films up to 2014, and age ratings are known for just 23% of films.

## Develop

Run the tests with the database up. The ranking model's tests need PyTorch, installed as shown above:

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
- [ ] Stage 2: Claude conversation (built and tested with a stand-in for Claude; waiting on a real evaluation)
- [x] Stage 3: PyTorch ranking model
- [ ] Stage 4: Simulated-user evaluation
- [ ] Stage 5: Public launch
