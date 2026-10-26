# Pronóstico de demanda para reposición, con capa de agentes

Solución a una prueba técnica de Data Science (retail, México). En construcción: el plan y el estado de cada etapa
están en [`docs/specs/README.md`](docs/specs/README.md) y el proceso en [`PROCESS.md`](PROCESS.md).

## Cómo correrlo (hasta ahora)

Requisitos: [uv](https://docs.astral.sh/uv/) y GNU Make.

```bash
make setup                                  # ambiente (uv sync) + hooks de pre-commit
make data DATA_SRC="ruta/al/zip-o-carpeta"  # copia los 3 CSV a data/raw/ y verifica sus hashes
make validate                               # lint + tipos + tests
```

Los datos **no** se versionan: se colocan localmente con `make data` a partir del material de la prueba.
