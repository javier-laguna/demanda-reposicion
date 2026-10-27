.DEFAULT_GOAL := help
.PHONY: help setup data calidad eda ajuste backtest seleccion calibracion decision drift causal insights pipeline mcp app lint format types test validate docker-build up

DATA_SRC ?=
MODO ?=
DESDE ?=

help:  ## lista los comandos
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## crea el ambiente con uv e instala los hooks de pre-commit
	uv sync --locked
	uv run pre-commit install

data:  ## copia los CSV a data/raw/ (DATA_SRC=carpeta o zip) y verifica sus hashes
	uv run demanda data $(if $(DATA_SRC),--src "$(DATA_SRC)")

calidad:  ## valida contratos y escribe data/interim/panel.parquet y artifacts/calidad/reporte.json
	uv run demanda calidad

eda:  ## corre el notebook de EDA: reports/eda.html y artifacts/eda/resumen.json
	uv run demanda eda

ajuste:  ## ajusta LightGBM en orígenes previos al backtest: artifacts/modelos/ajuste_lightgbm.json
	uv run demanda ajustar

backtest:  ## backtest de origen rodante: artifacts/backtest/ y un run de MLflow por modelo
	uv run demanda backtest

seleccion:  ## regla de campeón sobre el backtest: artifacts/modelos/seleccion.json
	uv run demanda seleccion

calibracion:  ## campeón y retador en orígenes previos al backtest: artifacts/backtest/calibracion.parquet
	uv run demanda calibracion

decision:  ## newsvendor: políticas de reposición y su impacto, artifacts/decision/impacto.json
	uv run demanda decision

drift:  ## monitor de drift semana a semana: artifacts/drift/monitor.json
	uv run demanda drift

causal:  ## experimento de promociones: artifacts/causal/promociones.json
	uv run demanda causal

insights:  ## agentes, caché o plantillas (MODO=auto|vivo|cache|plantilla): artifacts/insights/insights.json
	uv run demanda insights $(if $(MODO),--modo $(MODO))

pipeline:  ## de calidad a insights, en orden (requiere make data; DESDE=etapa para retomar, MODO= de insights)
	uv run demanda pipeline $(if $(DESDE),--desde $(DESDE)) $(if $(MODO),--modo $(MODO))

mcp:  ## servidor MCP de solo lectura por stdio (HTTP en el 8765: uv run demanda mcp --http)
	uv run demanda mcp

app:  ## tablero en Streamlit en http://127.0.0.1:8501 (solo lee artifacts/)
	uv run demanda app

lint:  ## ruff: lint y formato (sin modificar)
	uv run ruff check .
	uv run ruff format --check .

format:  ## ruff: aplica formato y correcciones automáticas
	uv run ruff check --fix .
	uv run ruff format .

types:  ## pyrefly
	uv run pyrefly check

test:  ## pytest
	uv run pytest

validate: lint types test  ## lint + tipos + tests (requisito para cada commit)

docker-build:  ## construye la imagen
	docker compose build

up:  ## tablero, MCP y MLflow con docker compose (leen los artefactos: corre antes el pipeline)
	docker compose up
