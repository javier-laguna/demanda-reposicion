# Dos etapas: la primera instala con uv y la final copia solo el entorno y el código, sin uv, sin su caché y sin
# las dependencias de desarrollo. Torch sale del índice de CPU de PyTorch (ver [tool.uv.sources]).
FROM python:3.13-slim AS base
# LightGBM necesita OpenMP en tiempo de ejecución
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS construccion
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
# Primero solo las dependencias, para que la capa se reutilice mientras no cambie uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
# El proyecto queda instalado en modo editable sobre /app/src: config.py ubica la raíz del repo desde ahí
COPY src ./src
RUN uv sync --locked --no-dev

FROM base
WORKDIR /app
COPY --from=construccion /opt/venv /opt/venv
COPY --from=construccion /app/src ./src
COPY pyproject.toml README.md ./
COPY notebooks ./notebooks
COPY data/manifest.sha256 ./data/manifest.sha256
# La caché de insights (para correr sin key) y el diagrama que muestra el tablero
COPY insights ./insights
COPY .archify ./.archify
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/hf \
    PYDANTIC_AI_NO_BANNER=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
ENTRYPOINT ["demanda"]
CMD ["--help"]
