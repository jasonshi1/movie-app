# Tonight's Pick

A movie picker for one person. Name a few films you love or hate, say what you're in the mood for, and get one pick with a reason. Say "not that one" and why, and it tries again.

This project is in early development. Only the foundation exists so far.

## Run it

You need Docker. From the project root:

```bash
docker compose up --build
```

- App: http://localhost:5173
- API health check: http://localhost:8000/api/health

## Develop

Backend (Python 3.13, with the database running):

```bash
docker compose up -d --wait db
cd backend
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/ruff check .
.venv/Scripts/pytest
```

Frontend (Node 24):

```bash
cd frontend
npm ci
npm run typecheck
npm test
```

## Roadmap

- [ ] Stage 0: Foundation (waiting on the first CI run)
- [ ] Stage 1: Catalog and baseline recommendations, no LLM
- [ ] Stage 2: Claude conversation
- [ ] Stage 3: PyTorch ranking model
- [ ] Stage 4: Simulated-user evaluation
- [ ] Stage 5: Public launch
