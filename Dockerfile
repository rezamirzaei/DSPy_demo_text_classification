# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    DSPY_CACHEDIR=/app/.dspy_cache

FROM base AS builder
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip uv
COPY requirements-runtime.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-runtime.txt

FROM base AS production
RUN addgroup --system app && adduser --system --ingroup app app
COPY --from=builder /usr/local /usr/local
COPY app/ ./app/
COPY config.py run.py wsgi.py ./
RUN mkdir -p /app/data /app/.cache /app/.dspy_cache && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60", "wsgi:app"]

FROM production AS dev
USER root
COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements-dev.txt
COPY tests/ ./tests/
COPY pyproject.toml ./
USER app
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
