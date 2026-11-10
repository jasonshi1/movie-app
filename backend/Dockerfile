FROM python:3.13-slim

WORKDIR /srv
ENV EMBEDDING_MODEL_DIR=/srv/models
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .
# Fetch the embedding model at build time so the first mood search doesn't wait for a download.
RUN python -c "from app.embedding import load_model; load_model()"

RUN useradd --create-home appuser && chown -R appuser /srv/models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
