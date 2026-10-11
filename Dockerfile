# A13: the API image -- the whole query pipeline (search, reranker, LLM call) behind FastAPI.
# Built by docker-compose.yml (service "api"); the chat screen has its own image (ui/Dockerfile).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Packages first, code second: Docker reuses the slow package layer as long as
# pyproject.toml and uv.lock do not change, so a code edit rebuilds in seconds.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY app ./app
COPY data/raw ./data/raw

# Models (embedding + reranker, ~200 MB) are not baked in: docker-compose.yml mounts
# .cache/models, so they are downloaded once and survive rebuilds.
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
