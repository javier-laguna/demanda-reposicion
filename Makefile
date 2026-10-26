.DEFAULT_GOAL := help
.PHONY: help setup data calidad eda backtest lint format types test validate docker-build up

DATA_SRC ?=

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

backtest:  ## backtest de origen rodante: artifacts/backtest/ y un run de MLflow por modelo
	uv run demanda backtest

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

up:  ## corre el servicio con docker compose (requiere make data antes)
	docker compose up
