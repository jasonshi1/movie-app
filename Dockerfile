# The hosted app in one image: the API, the MCP endpoint at /mcp and the built frontend.

FROM node:24-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /srv
ENV EMBEDDING_MODEL_DIR=/srv/models \
    STATIC_DIR=/srv/static \
    PYTHONUNBUFFERED=1
COPY backend/pyproject.toml ./
COPY backend/app ./app
RUN pip install --no-cache-dir .
# Fetch the embedding model at build time so the first mood search doesn't wait for a download.
RUN python -c "from app.embedding import load_model; load_model()"
COPY --from=frontend /frontend/dist ./static

RUN useradd --create-home appuser && chown -R appuser /srv/models
USER appuser

EXPOSE 8000
# Hosting platforms set PORT. Behind their proxy, also set CLIENT_IP_HEADER so the rate limits
# see each visitor's address instead of the proxy's.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
