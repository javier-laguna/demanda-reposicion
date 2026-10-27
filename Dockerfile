FROM python:3.13-slim

# LightGBM necesita OpenMP en tiempo de ejecución
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Primero solo las dependencias, para que la capa se reutilice mientras no cambie uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY notebooks ./notebooks
COPY data/manifest.sha256 ./data/manifest.sha256
RUN uv sync --locked --no-dev

ENTRYPOINT ["demanda"]
CMD ["--help"]
