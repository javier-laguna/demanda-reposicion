# Pronóstico de demanda para reposición, con capa de agentes

Solución a una prueba técnica de Data Science sobre una cadena de retail con 80 tiendas en México. El problema de
negocio que elegí resolver es este: **¿cuánto reponer por tienda y categoría cada semana, y con qué evidencia?**

La respuesta tiene cinco piezas, todas reproducibles con un comando:

1. **Pronóstico** de las unidades por tienda × categoría a 7 días. Lo compara contra el sistema de reposición actual,
   cuatro baselines, LightGBM y Chronos-2, en un backtest de 17 semanas.
2. **Regla de decisión:** el nivel de inventario por categoría (newsvendor), con su impacto en costo, quiebre y
   sobreinventario.
3. **Monitor de drift:** cada semana, si cambiaron los datos, las entradas o el desempeño, y qué hacer.
4. **Experimento natural** con las promociones, que se asignaron al azar: ¿venden más?, ¿se pagan?
5. **Capa de agentes** (Pydantic AI, Anthropic por defecto) que interpreta los resultados vía un servidor MCP de solo
   lectura. Un validador comprueba cada cifra que escriben. Todo se ve en un tablero de Streamlit.

El proceso, incluido cómo se usó la IA, está en [`PROCESS.md`](PROCESS.md). El plan, las specs pre-registradas y el
resultado de cada etapa están en [`docs/specs/`](docs/specs/README.md).

## Resultados principales

| Pregunta | Respuesta | Detalle |
|---|---|---|
| ¿El sistema actual es honesto? | No del todo: su señal de reposición mira un día al futuro | Sin esa fuga, su WAPE semanal sube de 18.6 a 20.1 %. [E1](docs/specs/E1.md) |
| ¿Un modelo nuevo es mejor? | Todavía no se puede afirmar | LightGBM baja el WAPE de 27.3 a 20.3 %, pero la prueba Diebold-Mariano con 17 cortes da p = 0.17. Sigue el sistema actual, con LightGBM en vigilancia. [E4](docs/specs/E4.md) |
| ¿Dónde está el valor? | En la regla de decisión | Un colchón calibrado (newsvendor) sobre el pronóstico de hoy cuesta 40 % menos que reponer al pronóstico puntual (p = 0.019). LightGBM agrega 9 %, sin significancia. [E5](docs/specs/E5.md) |
| ¿El monitor funciona? | Sí, con un límite conocido | Detecta los tres escenarios de drift inyectado y una falla real de captura. Confunde la estacionalidad anual con drift en noviembre y enero. [E5](docs/specs/E5.md) |
| ¿Las promociones venden más? | No | +0.05 % en unidades (IC95 de −0.25 a +0.34 %); solo Bebidas responde (+1.45 %). Ningún descuento de 5 a 20 % se paga. [E6](docs/specs/E6.md), [diseño del próximo A/B](docs/causal_promos.md) |
| ¿Hay algo raro en los datos? | Sí | Los «conteos inconsistentes» del ~20 % de las filas son exactamente los días con promoción: `total_transactions` cuenta registros sin pago. [E6](docs/specs/E6.md) |

Más detalle en la [model card](docs/model_card.md). Lo que cambiaría para producción está en
[`docs/despliegue.md`](docs/despliegue.md).

## Cómo correrlo

Los datos de la prueba **no** están en el repo. Se colocan localmente a partir del zip o la carpeta de la prueba, y
se verifican contra `data/manifest.sha256`.

### Con uv y make

Requisitos: [uv](https://docs.astral.sh/uv/) y GNU Make. Python 3.13 lo instala uv.

```bash
make setup                              # ambiente (uv sync --locked) + hooks de pre-commit
make data DATA_SRC="ruta/al/zip"        # copia los 3 CSV a data/raw/ y verifica sus hashes
make pipeline                           # todas las etapas, de calidad a insights (~25 min)
make app                                # tablero en http://127.0.0.1:8501
```

`make pipeline` corre, en orden:

| Etapa | Comando suelto | Tiempo aprox. | Escribe |
|---|---|---|---|
| Contratos y calidad | `make calidad` | segundos | `data/interim/panel.parquet`, `artifacts/calidad/` |
| EDA (marimo) | `make eda` | ~1 min | `reports/eda.html`, `artifacts/eda/` |
| Backtest | `make backtest` | ~19 min | `artifacts/backtest/` y MLflow |
| Selección | `make seleccion` | segundos | `artifacts/modelos/seleccion.json` |
| Calibración | `make calibracion` | ~4 min | `artifacts/backtest/calibracion.*` |
| Decisión | `make decision` | segundos | `artifacts/decision/impacto.json` |
| Drift | `make drift` | ~15 s | `artifacts/drift/monitor.json` |
| Promociones | `make causal` | ~20 s | `artifacts/causal/promociones.json` |
| Insights | `make insights` | segundos sin key | `artifacts/insights/insights.json` |

- Si una etapa falla, el pipeline se detiene y dice cómo retomar: `make pipeline DESDE=<etapa>`.
- La primera corrida del backtest baja de Hugging Face los pesos de Chronos-2. Sin red, el backtest se detiene; para
  correr sin Chronos-2: `uv run demanda pipeline --modelos sistema_actual,naive_estacional,ma7,autoets,lightgbm`.
- `make validate` corre lint, tipos y tests (sin datos reales) y es el requisito de cada commit.
- `uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db` muestra los runs del backtest.

### Con Docker

Requisitos: Docker con Compose. Los volúmenes comparten `data/`, `artifacts/` e `insights/` con el host.

```bash
docker compose build
docker compose run --rm -v "$PWD/ruta/al/zip:/tmp/datos.zip:ro" pipeline data --src /tmp/datos.zip
docker compose run --rm pipeline        # todas las etapas
docker compose up app mcp mlflow        # tablero :8501, MCP por HTTP :8765, MLflow :5000
```

En macOS, el receptor de AirPlay ocupa el puerto 5000. Si MLflow no arranca, usa otro puerto del host:
`MLFLOW_PORT=5001 docker compose up mlflow`.

La imagen pesa ~3.4 GB, la mayor parte por torch (solo lo usa Chronos-2). Los pesos de Chronos-2 quedan en un volumen
con nombre, así que se bajan una sola vez. Docker usa su propia base de MLflow (`artifacts/mlflow-docker.db`), para no
mezclar sus runs con los de la laptop.

### Sin key y con key

La capa de agentes funciona sin key, y así corren CI y Docker por defecto. Hay tres modos:

- **Sin key:** `make insights` usa la caché versionada (`insights/cache.json`), que viene de una corrida real con
  Claude Opus 5.5. La caché vale solo si sus cifras siguen en los artefactos actuales y los prompts no cambiaron. Si
  no vale, se usan plantillas deterministas.
- **Con key:** copia `.env.example` a `.env`, pon `ANTHROPIC_API_KEY` y corre `make insights MODO=vivo`. Cuesta
  ~USD 0.4, tarda ~2 min y renueva la caché. El código frena en USD 1 por corrida y USD 20 acumulados; el límite
  duro va en la consola del proveedor.
- **Otro proveedor:** `LLM_MODEL=openrouter:<modelo>` con `OPENROUTER_API_KEY`, o cualquier gateway compatible con la
  API de OpenAI con `LLM_BASE_URL`.

El servidor MCP también sirve a clientes externos: `make mcp` (stdio) o `uv run demanda mcp --http` (HTTP en
127.0.0.1:8765).

## Estructura

```
├── src/demanda/
│   ├── datos/         contratos (Pandera), panel y reporte de calidad
│   ├── features/      matriz de features sin fugas
│   ├── backtest/      protocolo de cortes, bucle, métricas, calibración y MLflow
│   ├── modelos/       baselines, LightGBM y Chronos-2 (en un proceso aparte)
│   ├── seleccion/     Diebold-Mariano y regla de campeón
│   ├── decision/      newsvendor y supuestos de costo
│   ├── drift/         monitor y escenarios de drift inyectado
│   ├── causal/        experimento de promociones
│   ├── mcp_server/    servidor MCP de solo lectura
│   ├── agentes/       agentes, validador, caché, plantillas y presupuesto
│   ├── app/           tablero Streamlit
│   └── cli.py         `demanda <etapa>`, el punto de entrada de make y Docker
├── notebooks/eda.py   EDA en marimo
├── tests/             pruebas con datos sintéticos; CI nunca ve datos reales
├── insights/          la caché versionada de la capa de agentes
├── docs/              specs por etapa, model card, despliegue y diseño del A/B
├── .archify/          diagrama de arquitectura (se ve en la página Arquitectura del tablero)
└── .claude/           subagentes de desarrollo (opcional, ver abajo)
```

## Sobre `.claude/`

`.claude/agents/` tiene los subagentes de Claude Code que se usaron para desarrollar el repo (EDA, forecast, causal y
revisión). Solo funcionan dentro de Claude Code y quedan como evidencia del proceso. **No hacen falta para reproducir
nada:** la solución corre solo con uv y make, o con Docker.

## Reproducibilidad

- Las dependencias están fijadas en `uv.lock`. Torch viene del índice de CPU de PyTorch en Linux y de PyPI en macOS.
- Cada artefacto guarda el hash de las fuentes de las que salió, y las etapas siguientes se niegan a correr sobre una
  corrida de otra fuente.
- Dos corridas en la misma máquina dan artefactos idénticos. Entre macOS y Linux, los pronósticos difieren en ~1e-6
  relativo por las librerías numéricas. Las conclusiones no cambian, y la caché de insights sigue valiendo porque su
  huella redondea a 3 cifras significativas.
- Secretos solo en `.env` (gitignored), leídos como `SecretStr`. gitleaks corre en pre-commit y en CI.
