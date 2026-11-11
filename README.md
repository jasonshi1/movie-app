# Tonight's Pick

A movie picker for one person. Name a few films you love and get one film to watch tonight, with the reason it was chosen. Say "not that one" and it tries again.

This project is in development. Quick picks come from a ranking model trained with PyTorch, with no LLM involved. A conversation with Claude is built and switches on when the server has an Anthropic API key. It hasn't been evaluated against the real model yet.

## Run it

You need Docker. From the project root:

```bash
docker compose up --build
```

- App, with live reloading for development: http://localhost:5173
- App as it runs when hosted, from the production image: http://localhost:8000
- API health check: http://localhost:8000/api/health
- MCP server: http://localhost:8000/mcp

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
- Chat pauses for the rest of the UTC day once Claude has cost `DAILY_SPEND_LIMIT_USD` ($5 by default). Quick picks keep working. A turn that's already running can go past the limit, so keep a spending limit on the key as well.
- Each visitor can send 30 chat messages an hour and 120 other requests a minute. Visitors are counted by IP address in memory, so nothing about them is stored.

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

## Simulated users

**With quick picks, 77.0% ± 1.2% of simulated users get a film they liked within 5 suggestions** (95% confidence, 5,000 users).

Each simulated user is a real MovieLens user from the test split, which no earlier stage touched:

- They give their first 5 liked films as favourites.
- They accept a suggestion only if they rated it 4 or higher later on.
- They turn down a film they'd already rated with "I've already seen that one", because the app can't know what someone has seen.
- They turn down anything else with a reason built from the Tag Genome themes where it differs most from everything they liked, such as "less dinosaurs and more military".

| Method | Hit within 1 | Hit within 5 | Hit within 10 |
|---|---:|---:|---:|
| Quick picks (ranking model) | 43.6% | 77.0% | 86.6% |
| Quick picks (Stage 1 method) | 40.1% | 73.1% | 83.5% |
| Most popular films | 34.5% | 68.2% | 78.3% |

Compared user by user, the ranking model gets 4.0 ± 1.3 more points within 5 suggestions than Stage 1. Full results are in [results/stage4_simulation.md](results/stage4_simulation.md). To reproduce them, with no API calls:

```bash
.venv/Scripts/python -m pipeline.simulate
```

The same simulated users can talk to Claude, which reads each reason and searches again. This spends API credit, so try 10 users first. The report shows the cost per user and what 200 or 1,000 users would cost:

```bash
.venv/Scripts/python -m pipeline.simulate --claude --limit 10 --yes
```

Unrated films count as misses, so these rates understate how often people would be happy. Simulated users aren't real people, so treat the numbers as a simulation.

## Use it from an AI assistant (MCP)

The app is also an MCP server at `/mcp`, using streamable HTTP with no sign-in. Any assistant that supports remote MCP servers can use its tools:

- `search_titles`: find films by title, allowing for typos, to get their IDs.
- `pick_film`: one pick from a list of favourite IDs, with the reason, using the ranking model.
- `find_films`: up to 8 films that meet limits on runtime, years, genres and age rating, ranked by fit with the favourites and a described mood.
- `film_details`: a film's details, themes and a short Wikipedia description.

The tools run the same code as the app and never call Claude, so they cost nothing to run. They share the per-minute rate limit.

## Host it

The image built from the root `Dockerfile` serves the app, the API and the MCP server on one port. To host it you need:

1. **A PostgreSQL database with the pgvector extension.** Most managed PostgreSQL services offer it. The catalog takes about 100 MB. Load it from the machine where you built the data:

   ```bash
   DATABASE_URL=<hosted database URL> .venv/Scripts/python -m pipeline.load_db
   ```

2. **A container host** that runs the image with at least 1 GB of memory, for the embedding model. Set these environment variables:

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | The hosted database |
   | `ANTHROPIC_API_KEY` | Your key, kept on the server. Set a spending limit on it in the Anthropic Console |
   | `PUBLIC_HOSTS` | The app's host name, such as `tonights-pick.example.com`. The MCP endpoint refuses other names |
   | `CLIENT_IP_HEADER` | The header your host's proxy sets to the visitor's address, such as `Fly-Client-IP`, so rate limits see each visitor. Never `X-Forwarded-For`, which visitors can add to |
   | `DAILY_SPEND_LIMIT_USD` | Optional, $5 by default |

   The server listens on `PORT`, which most hosts set, or 8000.

### Launch on Fly.io and Neon

`fly.toml` runs one always-on machine with 1 GB of memory in Toronto, so the link opens without waiting for the model to load. Neon hosts the database. You create both accounts and add any payment method yourself. Run these commands in your own terminal, and never paste keys into a chat.

1. In Neon, create a project in a region near Toronto, such as AWS US East. Put its connection strings in `.env.local` at the project root: `neon init` from Neon's CLI writes them there. Or create the file yourself with `DATABASE_URL` (pooled) and `DATABASE_URL_UNPOOLED` (connection pooling turned off). Git and Docker ignore the file, and the app never reads it, so tests and local Docker keep using the local database.
2. Load the catalog into it, from `backend`. The loader creates the tables too. It uses the unpooled string, and the parentheses keep Neon's address out of your shell, so a later test run, which empties tables, can't reach it:

   ```bash
   (. ../.env.local; DATABASE_URL="$DATABASE_URL_UNPOOLED" .venv/Scripts/python -m pipeline.load_db)
   ```

3. Install [flyctl](https://fly.io/docs/flyctl/install/) and sign in with `fly auth login`.
4. Create the app. App names are global, so if `tonights-pick` is taken, choose another and change it in `fly.toml`, in both `app` and `PUBLIC_HOSTS`:

   ```bash
   fly apps create tonights-pick
   ```

5. Store the secrets on Fly.io, from the project root. The app gets the pooled string, which suits its short connections:

   ```bash
   (. ./.env.local; fly secrets set DATABASE_URL="$DATABASE_URL" ANTHROPIC_API_KEY="<your key>")
   ```

6. Deploy from the project root. The first build takes a few minutes:

   ```bash
   fly deploy
   ```

7. Open https://tonights-pick.fly.dev and try a quick pick and a chat. The MCP server is at https://tonights-pick.fly.dev/mcp.
8. Check that visitors can't get around the rate limits by claiming another address. This sends 125 searches, each claiming a different one. A few should come back `429`, because Fly.io replaces the header with your real address:

   ```bash
   for i in $(seq 1 125); do curl -s -o /dev/null -w "%{http_code}\n" -H "Fly-Client-IP: 203.0.113.$i" "https://tonights-pick.fly.dev/api/films/search?q=matrix"; done | sort | uniq -c
   ```

   If all 125 come back `200`, visitors can forge the header. Remove `CLIENT_IP_HEADER` from `fly.toml` and redeploy. Every visitor then shares one set of limits, which is safe but strict, while you look for another way to identify visitors.

### Privacy

There are no accounts, cookies or analytics. Favourites and saved preferences stay in the visitor's browser. Chat messages are sent to Anthropic to get Claude's replies, and each conversation is stored so it can continue. Conversations are deleted after 30 days without a new message. Rate limits count visitors by IP address in memory only.

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
- [ ] Stage 4: Simulated-user evaluation (quick picks measured; the Claude comparison is waiting on an API key)
- [ ] Stage 5: Public launch (ready to host, with spending and rate limits and an MCP server; not hosted yet)
