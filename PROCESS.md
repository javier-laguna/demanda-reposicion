# Proceso de trabajo

Este documento registra cómo se construyó la solución: las decisiones, las herramientas y, en particular, cómo se
usó la IA. Se escribió al cerrar cada etapa del plan (`docs/specs/README.md`).

## Herramientas

| Ámbito | Herramienta |
|---|---|
| Ambiente y calidad | uv (Python 3.13 fijado, `uv.lock`), ruff, pyrefly, pytest, pre-commit, gitleaks, Make |
| Datos | Polars, Pandera _(a partir de E1)_ |
| Modelado | LightGBM vía mlforecast, statsforecast, Chronos-2, statsmodels _(E3–E6)_ |
| Seguimiento | MLflow local con SQLite _(E3)_ |
| Capa de IA | Pydantic AI, con Anthropic por defecto y proveedor configurable; servidor MCP con FastMCP _(E7)_ |
| Entrega | Streamlit _(E8)_, Docker Compose, GitHub Actions |
| Asistente de desarrollo | Claude Code (Claude Opus 5.5) con subagentes, skills y servidores MCP de documentación |

## Cómo se usó la IA

La IA aparece en dos lugares distintos, y conviene no confundirlos:

1. **En el desarrollo.** Claude Code construyó el repo conmigo:
   - exploración de datos y lectura de documentación;
   - código y tests;
   - revisión de cada etapa con subagentes de `.claude/agents/`;
   - diagramas con la skill archify.

   Las decisiones de enfoque y alcance fueron mías: la IA propuso opciones y yo elegí o corregí. Los criterios de
   validación los escribió Claude Code en la spec de cada etapa, a partir del plan que aprobé. Nada del producto
   depende de Claude Code.
2. **En el producto.** Una capa de agentes (Pydantic AI) interpreta los resultados del pipeline y redacta
   conclusiones de negocio. Tiene tres límites: no calcula ni decide, solo lee vía un servidor MCP, y un validador
   determinista comprueba cada cifra que cita contra los artefactos. Funciona con cualquier proveedor y también sin
   key: en ese caso usa la caché versionada o conclusiones generadas con plantillas.

## Bitácora

### Antes de E0 · Exploración, investigación y plan

- **Perfilado de los datos con scripts desechables**, antes de proponer nada. De ahí salieron los hallazgos que
  definieron el enfoque, y que hoy están en `CLAUDE.md`:
  - la fuga de `replenishment_signal`;
  - el 20 % de conteos inconsistentes;
  - que los huecos se reducen a 42 filas;
  - la aleatorización de las promociones;
  - los efectos de calendario;
  - la falta de señal en el pago en efectivo.
- **Investigación de herramientas 2025–2026 con tres agentes en paralelo:**
  - servidores MCP del ecosistema de ciencia de datos;
  - modelos y librerías de forecasting, tabular y causal;
  - herramientas de trabajo.

  Las versiones de 19 paquetes clave se verificaron después contra PyPI. De ahí salió el stack de la tabla de
  arriba.
- **Definición del problema.** La IA propuso varios problemas de negocio y se eligió el pronóstico de demanda para
  reposición, por tres razones:
  - es el único con un sistema actual contra el cual medir valor;
  - `total_transactions` tiene conteos inconsistentes;
  - el efectivo no tiene señal.

  El experimento de promociones se sumó porque la asignación aleatoria permite medir causalidad.
- **Correcciones mías a la propuesta inicial:**
  - pedí que los agentes por etapa fueran parte del sistema replicable, no solo una herramienta de desarrollo;
  - exigí que la capa de IA fuera agnóstica de proveedor;
  - pedí un monitor de drift explícito.
- **Diagrama de arquitectura con archify.** La versión final, con los resultados, se ve en la página Arquitectura
  del tablero.

### E0 · Base del repo y gobierno

- **Qué se hizo:**
  - ambiente uv con Python 3.13 fijado;
  - Makefile como punto de entrada;
  - pre-commit con ruff, pyrefly, gitleaks y un bloqueo de archivos de datos;
  - configuración con `SecretStr`;
  - `demanda data`, que coloca los CSV y verifica sus hashes (los datos no se versionan);
  - esqueletos de Docker y CI;
  - specs por etapa.
- **Decisión:** los datos de la prueba no se suben al repo. Quien lo reproduzca los coloca con
  `make data DATA_SRC=...`, y el manifiesto SHA-256 garantiza que son los originales.

### E1 · Datos: contratos y calidad

- **Qué se hizo:**
  - contratos Pandera (backend Polars) para las 3 tablas y sus uniones;
  - reporte de calidad en JSON;
  - panel sobre la malla completa con banderas de calidad;
  - `make calidad`;
  - 29 tests sobre datos sintéticos.
- **Criterio de diseño:** el contrato solo exige lo que debe cumplirse siempre (tipos, dominios, llaves, montos que
  cuadran). Los problemas conocidos, como los conteos inconsistentes o los nulos, no lo rompen: se miden y se marcan.
  Si un archivo nuevo rompe el contrato, la etapa no escribe nada.
- **Qué aportó la IA:** al diseñar la prueba de fuga no se quedó con la cifra de la exploración (MA7 que incluye el
  día, correlación 0.98). Probó ventanas que terminan después del día y una regresión por desfase, y así descubrió
  que la señal incluye las ventas de t+1. Eso cerró el pendiente "falta verificar si usa información futura" y
  endureció la regla de uso de la señal: con origen T, solo valores fechados en T−1 o antes. También implica que el
  WAPE del sistema actual medido en la exploración era optimista.
- **Decisión:** el panel incluye las filas faltantes, con medidas nulas y una bandera, en vez de omitirlas. Así los
  lags por `shift` corresponden a días de calendario y el backtest sabe qué fechas no evaluar.
